"""Measure retrieval quality so ranking changes are decisions, not vibes.

Case format (JSON list): {"query": "...", "relevant": ["substring of the right file's path", ...]}
A query is a hit@k if any of the top-k *documents* has a path containing a relevant substring.
"""
from __future__ import annotations

from .embeddings import get_embedder
from .search import search


def evaluate(conn, cfg, cases: list[dict], k: int = 5, rerank: bool = False) -> list[dict]:
    embedder = get_embedder(cfg.embed_model)
    setups = [("bm25", "bm25", False), ("vector", "vector", False), ("hybrid", "hybrid", False)]
    if rerank:
        setups.append(("hybrid+rerank", "hybrid", True))

    rows = []
    for name, mode, rr in setups:
        hits, rr_sum = 0, 0.0
        for case in cases:
            results = search(conn, cfg, case["query"], k=k, mode=mode, rerank=rr,
                             embedder=embedder, per_doc=1)
            rel = [r.lower() for r in case["relevant"]]
            rank = next((i + 1 for i, res in enumerate(results)
                         if any(x in res.path.lower() for x in rel)), None)
            if rank:
                hits += 1
                rr_sum += 1.0 / rank
        n = max(len(cases), 1)
        rows.append({"method": name, f"recall@{k}": hits / n, "mrr": rr_sum / n})
    return rows


def format_table(rows: list[dict]) -> str:
    keys = list(rows[0].keys())
    out = ["  ".join(f"{k:<14}" for k in keys)]
    for r in rows:
        out.append("  ".join(f"{(v if isinstance(v, str) else format(v, '.3f')):<14}" for v in r.values()))
    return "\n".join(out)
