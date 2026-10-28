"""Incremental indexing: scan roots, skip unchanged files, chunk, embed, store."""
from __future__ import annotations

import fnmatch
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bookmarks import fetch_page_text, parse_bookmarks
from .chunking import Chunk, chunk_sections
from .config import Config
from .db import bump_version, connect, delete_document, get_meta, set_meta
from .embeddings import get_embedder
from .extractors import Section, extract


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)  # never stdout: it would corrupt MCP stdio


@dataclass
class Stats:
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0

    def summary(self) -> str:
        return (f"new={self.new} updated={self.updated} unchanged={self.unchanged} "
                f"skipped={self.skipped} failed={self.failed} removed={self.removed}")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _excluded(name: str, cfg: Config) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in cfg.exclude)


def is_indexable(path: Path, cfg: Config) -> bool:
    """True if `path` lives under a root, has a supported extension and isn't hidden/excluded."""
    if path.suffix.lower() not in cfg.extensions:
        return False
    p = path.resolve()
    for root in cfg.roots:
        try:
            rel = p.relative_to(Path(root).resolve())
        except ValueError:
            continue
        return not any(part.startswith(".") or _excluded(part, cfg) for part in rel.parts)
    return False


class Indexer:
    def __init__(self, cfg: Config, conn=None, embedder=None):
        self.cfg = cfg
        self.conn = conn or connect(cfg.db_path)
        self.embedder = embedder or get_embedder(cfg.embed_model)
        self._check_model()

    # ------------------------------------------------------------ model bookkeeping
    def _check_model(self) -> None:
        stored = get_meta(self.conn, "embed_model")
        n = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if stored and stored != self.embedder.name and n:
            raise RuntimeError(
                f"Index was built with embedding model '{stored}' but config uses "
                f"'{self.embedder.name}'. Run: psearch index --rebuild"
            )
        if stored != self.embedder.name:
            with self.conn:
                set_meta(self.conn, "embed_model", self.embedder.name)

    def reset(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chunks_fts")
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM documents")
            set_meta(self.conn, "embed_model", self.embedder.name)
            bump_version(self.conn)

    # ------------------------------------------------------------ scanning
    def _walk(self):
        exts = set(self.cfg.extensions)
        for root in self.cfg.roots:
            r = Path(root)
            if not r.exists():
                log(f"[warn] root does not exist: {r}")
                continue
            for dirpath, dirnames, filenames in os.walk(r):
                dirnames[:] = sorted(
                    d for d in dirnames if not d.startswith(".") and not _excluded(d, self.cfg)
                )
                for fn in sorted(filenames):
                    if fn.startswith(".") or _excluded(fn, self.cfg):
                        continue
                    p = Path(dirpath) / fn
                    if p.suffix.lower() in exts:
                        yield p

    def index_all(self, rebuild: bool = False) -> Stats:
        stats = Stats()
        if rebuild:
            self.reset()
        for path in self._walk():
            status = self.index_file(path)
            setattr(stats, status, getattr(stats, status) + 1)
            if status in ("new", "updated"):
                log(f"[{status}] {path}")
        stats.removed = self.prune()
        return stats

    def prune(self) -> int:
        """Drop documents whose file vanished or is no longer under a configured root."""
        if any(not Path(r).exists() for r in self.cfg.roots):
            log("[warn] a root is missing (unmounted drive?) - skipping prune")
            return 0
        removed = 0
        rows = self.conn.execute("SELECT id, path FROM documents WHERE type != 'bookmark'").fetchall()
        for r in rows:
            p = Path(r["path"])
            if not p.exists() or not is_indexable(p, self.cfg):
                self._remove_id(r["id"])
                removed += 1
        return removed

    def remove_path(self, path: str | Path) -> bool:
        key = str(Path(path).resolve())
        row = self.conn.execute("SELECT id FROM documents WHERE path=?", (key,)).fetchone()
        if row:
            self._remove_id(row["id"])
            return True
        return False

    def _remove_id(self, doc_id: int) -> None:
        with self.conn:
            delete_document(self.conn, doc_id)
            bump_version(self.conn)

    # ------------------------------------------------------------ single file
    def index_file(self, path: str | Path, force: bool = False) -> str:
        path = Path(path)
        try:
            st = path.stat()
            if st.st_size > self.cfg.max_file_mb * 1024 * 1024:
                return "skipped"
            key = str(path.resolve())
            row = self.conn.execute(
                "SELECT id, mtime, size, content_hash FROM documents WHERE path=?", (key,)
            ).fetchone()
            # Fast path: same size + mtime -> unchanged, no hashing needed.
            if row and not force and row["size"] == st.st_size and abs(row["mtime"] - st.st_mtime) < 1e-3:
                return "unchanged"
            digest = file_hash(path)
            if row and not force and row["content_hash"] == digest:
                with self.conn:
                    self.conn.execute(
                        "UPDATE documents SET mtime=?, size=? WHERE id=?",
                        (st.st_mtime, st.st_size, row["id"]),
                    )
                return "unchanged"

            ex = extract(path, self.cfg)
            sections = [s for s in (ex.sections if ex else []) if s.text.strip()]
            if not sections:
                if row:
                    self._remove_id(row["id"])
                return "skipped"
            stored = self._store(
                key=key, doc_type=ex.doc_type, title=ex.title, mtime=st.st_mtime, size=st.st_size,
                content_hash=digest, sections=sections, existing_id=row["id"] if row else None,
            )
            if not stored:
                return "skipped"
            return "updated" if row else "new"
        except Exception as e:  # one bad file must not abort the whole run
            log(f"[failed] {path}: {e}")
            return "failed"

    # ------------------------------------------------------------ bookmarks
    def index_bookmarks(self, html_file: str | Path, fetch: bool = False, workers: int = 8) -> Stats:
        stats = Stats()
        html = Path(html_file).read_text(encoding="utf-8", errors="replace")
        todo = []
        for b in parse_bookmarks(html):
            base = "\n\n".join(x for x in [b.title, f"Folder: {b.folder}" if b.folder else "", b.url] if x)
            row = self.conn.execute(
                "SELECT id, content FROM documents WHERE path=?", (b.url,)
            ).fetchone()
            if row and (not fetch or len(row["content"] or "") > len(base) + 200):
                stats.unchanged += 1  # exists (and already fetched if --fetch)
                continue
            todo.append((b, base, row["id"] if row else None))

        def get_text(item):
            return fetch_page_text(item[0].url) if fetch else ""

        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched = list(pool.map(get_text, todo))

        for (b, base, existing_id), page_text in zip(todo, fetched):
            text = base + ("\n\n" + page_text if page_text else "")
            digest = hashlib.sha256(text.encode()).hexdigest()
            try:
                ok = self._store(
                    key=b.url, doc_type="bookmark", title=b.title, mtime=b.add_date or time.time(),
                    size=len(text), content_hash=digest, sections=[Section(text)],
                    existing_id=existing_id,
                )
            except Exception as e:
                log(f"[failed] {b.url}: {e}")
                stats.failed += 1
                continue
            if ok:
                stats.updated += existing_id is not None
                stats.new += existing_id is None
            else:
                stats.skipped += 1
        return stats

    # ------------------------------------------------------------ storage
    @staticmethod
    def _embed_text(title: str, c: Chunk) -> str:
        head = " - ".join(x for x in (title, c.heading) if x)
        return f"{head}\n{c.text}" if head else c.text

    def _store(self, *, key, doc_type, title, mtime, size, content_hash, sections, existing_id) -> bool:
        chunks = chunk_sections(sections, self.cfg.chunk_words, self.cfg.overlap_words)
        if not chunks:
            if existing_id:
                self._remove_id(existing_id)
            return False
        embs = self.embedder.embed_documents([self._embed_text(title, c) for c in chunks])
        content = "\n\n".join(s.text for s in sections)
        with self.conn:
            if existing_id:
                delete_document(self.conn, existing_id)
            cur = self.conn.execute(
                "INSERT INTO documents(path,type,title,mtime,size,content_hash,content,indexed_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (key, doc_type, title, mtime, size, content_hash, content, time.time()),
            )
            doc_id = cur.lastrowid
            for i, (c, e) in enumerate(zip(chunks, embs)):
                cur = self.conn.execute(
                    "INSERT INTO chunks(doc_id,chunk_index,text,page,heading,embedding) VALUES (?,?,?,?,?,?)",
                    (doc_id, i, c.text, c.page, c.heading, e.astype(np.float32).tobytes()),
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts(rowid,text,title) VALUES (?,?,?)",
                    (cur.lastrowid, c.text, f"{title} {c.heading or ''}".strip()),
                )
            bump_version(self.conn)
        return True
