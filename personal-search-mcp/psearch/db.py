from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
    id           INTEGER PRIMARY KEY,
    path         TEXT UNIQUE NOT NULL,   -- absolute file path, or URL for bookmarks
    type         TEXT NOT NULL,          -- pdf | note | web | docx | screenshot | bookmark
    title        TEXT,
    mtime        REAL NOT NULL,          -- drives "last week" style filters
    size         INTEGER,
    content_hash TEXT,
    content      TEXT,                   -- full extracted text (for read_document)
    indexed_at   REAL
);
CREATE TABLE IF NOT EXISTS chunks(
    id          INTEGER PRIMARY KEY,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    page        INTEGER,
    heading     TEXT,
    embedding   BLOB                     -- float32, L2-normalised
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_mtime ON documents(mtime);
CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(type);
-- rowid of chunks_fts == chunks.id. Porter stemming: "searching" matches "search".
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, title, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def open_db(path: str | Path):
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def get_meta(conn, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn, key: str, value: str) -> None:
    """Does not commit; callers decide the transaction boundary."""
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def bump_version(conn) -> None:
    set_meta(conn, "index_version", str(int(get_meta(conn, "index_version", "0")) + 1))


def delete_document(conn, doc_id: int) -> None:
    conn.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id=?)", (doc_id,)
    )
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))  # chunks cascade
