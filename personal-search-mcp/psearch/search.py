"""Retrieval: BM25 (FTS5) + dense vectors, fused with Reciprocal Rank Fusion, optional rerank."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .db import get_meta
from .embeddings import get_embedder, get_reranker

# Conversational filler: dropped from the BM25 query so "find that explanation of X I read
# last week" behaves like "explanation X".
STOPWORDS = set(
    "a an and are as at be but by for from has have i in into is it its of on or so that the their "
    "then there these they this to was were what when where which who why will with you your me my "
    "find show about read last week weeks month months ago some any can could would should recent "
    "recently yesterday today".split()
)


# ------------------------------------------------------------------ filters
def parse_when(value: str | None, end: bool = False) -> float | None:
    """ISO date/datetime -> epoch seconds (local time). Date-only `end` values are inclusive."""
    if not value:
        return None
    dt = datetime.fromisoformat(value.strip())
    if end and len(value.strip()) <= 10:
        dt += timedelta(days=1)
    return dt.timestamp()


@dataclass
class Filters:
    since: float | None = None
    until: float | None = None
    doc_type: str | None = None
    path_contains: str | None = None

    def sql(self) -> tuple[str, list]:
        parts, params = [], []
        if self.since is not None:
            parts.append("d.mtime >= ?"); params.append(self.since)
        if self.until is not None:
            parts.append("d.mtime < ?"); params.append(self.until)
        if self.doc_type:
            parts.append("d.type = ?"); params.append(self.doc_type)
        if self.path_contains:
            parts.append("d.path LIKE ?"); params.append(f"%{self.path_contains}%")
        return ("".join(f" AND {p}" for p in parts), params)

    def active(self) -> bool:
        return bool(self.sql()[0])


# ------------------------------------------------------------------ BM25
def build_match(query: str) -> str:
    toks = list(dict.fromkeys(re.findall(r"\w+", query.lower())))
    keep = [t for t in toks if t not in STOPWORDS and len(t) > 1] or toks
    return " OR ".join(f'"{t}"' for t in keep)


def bm25_search(conn, query: str, filters: Filters, limit: int) -> list[tuple[int, float]]:
    match = build_match(query)
    if not match:
        return []
    frag, params = filters.sql()
    rows = conn.execute(
        f"""SELECT c.id AS id, bm25(chunks_fts, 1.0, 2.0) AS s   -- title column weighted 2x
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ?{frag}
            ORDER BY s LIMIT ?""",
        [match, *params, limit],
    ).fetchall()
    return [(r["id"], -r["s"]) for r in rows]  # FTS5 bm25 is lower=better; flip sign


# ------------------------------------------------------------------ vectors
class _VectorIndex:
    def __init__(self, key, ids, mat):
        self.key, self.ids, self.mat = key, ids, mat


_cache: _VectorIndex | None = None


def get_vector_index(conn) -> _VectorIndex:
    """All embeddings as one float32 matrix in RAM. Brute-force cosine is milliseconds for
    ~100k chunks, and avoids native SQLite extensions that are painful to install."""
    global _cache
    dbfile = conn.execute("PRAGMA database_list").fetchone()[2]
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    key = (dbfile, get_meta(conn, "index_version", "0"), count)
    if _cache is not None and _cache.key == key:
        return _cache
    rows = conn.execute(
        "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id"
    ).fetchall()
    if rows:
        dim = len(rows[0]["embedding"]) // 4
        mat = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32).reshape(len(rows), dim)
        ids = np.array([r["id"] for r in rows], dtype=np.int64)
    else:
        mat, ids = np.zeros((0, 0), np.float32), np.zeros(0, np.int64)
    _cache = _VectorIndex(key, ids, mat)
    return _cache


def vector_search(conn, embedder, query: str, filters: Filters, limit: int) -> list[tuple[int, float]]:
    idx = get_vector_index(conn)
    if len(idx.ids) == 0:
        return []
    q = embedder.embed_query(query)
    if q.shape[0] != idx.mat.shape[1]:
        raise RuntimeError("Embedding dimension mismatch - run: psearch index --rebuild")
    scores = idx.mat @ q  # cosine, since both sides are L2-normalised
    if filters.active():
        frag, params = filters.sql()
        allowed = [r[0] for r in conn.execute(
            f"SELECT c.id FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE 1=1{frag}", params)]
        scores = np.where(np.isin(idx.ids, np.array(allowed, dtype=np.int64)), scores, -np.inf)
    top = np.argsort(-scores)[:limit]
    return [(int(idx.ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]


# ------------------------------------------------------------------ fusion
def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score = sum_i w_i / (k + rank_i). Rank-based, so BM25 and cosine
    scores (different scales) combine without normalisation."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ------------------------------------------------------------------ results
def make_snippet(text: str, query: str, width: int = 350) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    low = flat.lower()
    pos = -1
    for t in re.findall(r"\w+", query.lower()):
        if t in STOPWORDS or len(t) < 2:
            continue
        i = low.find(t if len(t) <= 5 else t[:5])
        if i != -1 and (pos == -1 or i < pos):
            pos = i
    start = 0 if pos == -1 else max(0, pos - width // 4)
    snippet = flat[start:start + width]
    return ("… " if start > 0 else "") + snippet + (" …" if start + width < len(flat) else "")


@dataclass
class Result:
    chunk_id: int
    doc_id: int
    path: str
    title: str
    type: str
    mtime: float
    page: int | None
    heading: str | None
    text: str
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None

    def to_dict(self, query: str = "", include_text: bool = False) -> dict:
        d = {
            "title": self.title,
            "path": self.path,
            "type": self.type,
            "modified": datetime.fromtimestamp(self.mtime).isoformat(timespec="minutes"),
            "page": self.page,
            "heading": self.heading,
            "snippet": make_snippet(self.text, query),
            "score": round(self.score, 5),
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
        }
        if include_text:
            d["text"] = self.text
        return d


def _hydrate(conn, ids: list[int]) -> dict[int, object]:
    if not ids:
        return {}
    q = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""SELECT c.id AS chunk_id, c.doc_id, c.text, c.page, c.heading,
                   d.path, d.title, d.type, d.mtime
            FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({q})""",
        ids,
    ).fetchall()
    return {r["chunk_id"]: r for r in rows}


def search(
    conn,
    cfg,
    query: str,
    *,
    k: int = 5,
    mode: str = "hybrid",          # "bm25" | "vector" | "hybrid"
    filters: Filters | None = None,
    rerank: bool = False,
    embedder=None,
    per_doc: int = 2,              # max chunks returned per document (diversity)
    pool: int = 50,                # candidates pulled from each retriever
) -> list[Result]:
    if mode not in ("bm25", "vector", "hybrid"):
        raise ValueError("mode must be bm25, vector or hybrid")
    filters = filters or Filters()
    rankings: list[list[int]] = []
    bm_rank: dict[int, int] = {}
    vec_rank: dict[int, int] = {}

    if mode in ("bm25", "hybrid"):
        ids = [i for i, _ in bm25_search(conn, query, filters, pool)]
        rankings.append(ids)
        bm_rank = {i: r + 1 for r, i in enumerate(ids)}
    if mode in ("vector", "hybrid"):
        embedder = embedder or get_embedder(cfg.embed_model)
        ids = [i for i, _ in vector_search(conn, embedder, query, filters, pool)]
        rankings.append(ids)
        vec_rank = {i: r + 1 for r, i in enumerate(ids)}

    fused = rrf(rankings)
    if not fused:
        return []
    take = max(pool // 2, k * 5) if rerank else max(k * per_doc * 3, 30)
    order = [cid for cid, _ in fused[:take]]
    scores = dict(fused)
    rows = _hydrate(conn, order)
    order = [c for c in order if c in rows]

    if rerank and order:
        pairs = [(query, f"{rows[c]['title']}\n{rows[c]['text']}") for c in order]
        rr = get_reranker(cfg.rerank_model).predict(pairs)
        scores = dict(zip(order, rr))
        order.sort(key=lambda c: scores[c], reverse=True)

    results: list[Result] = []
    per: dict[int, int] = {}
    for cid in order:
        r = rows[cid]
        if per.get(r["doc_id"], 0) >= per_doc:
            continue
        per[r["doc_id"]] = per.get(r["doc_id"], 0) + 1
        results.append(Result(
            chunk_id=cid, doc_id=r["doc_id"], path=r["path"], title=r["title"], type=r["type"],
            mtime=r["mtime"], page=r["page"], heading=r["heading"], text=r["text"],
            score=scores[cid], bm25_rank=bm_rank.get(cid), vector_rank=vec_rank.get(cid),
        ))
        if len(results) >= k:
            break
    return results
