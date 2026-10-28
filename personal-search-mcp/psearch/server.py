"""MCP server (stdio). IMPORTANT: never print() to stdout here - it would corrupt the protocol."""
from __future__ import annotations

import logging
import sys
import threading
from datetime import date

from mcp.server.fastmcp import FastMCP

from .config import load_config
from .db import get_meta, open_db
from .embeddings import get_embedder
from .search import Filters, parse_when, search

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP(
    "personal-search",
    instructions=(
        "Search the user's own indexed PDFs, notes, web pages, screenshots (OCR) and bookmarks. "
        "Use `search` for any request to find or recall something the user saved or read. "
        "Convert relative dates ('last week', 'in March') into ISO `since`/`until` yourself using "
        "today's date. Use `read_document` to open the full text of a promising hit."
    ),
)
cfg = load_config()


@mcp.tool()
def get_current_date() -> str:
    """Today's date (ISO). Call this if you need it to turn 'last week' into since/until."""
    return date.today().isoformat()


@mcp.tool()
def search_documents(
    query: str,
    k: int = 5,
    since: str | None = None,
    until: str | None = None,
    doc_type: str | None = None,
    rerank: bool = False,
) -> list[dict]:
    """Hybrid (keyword + semantic) search over the user's personal documents.

    Args:
        query: natural-language description of what to find, e.g. "explanation of hybrid search".
            Do NOT put date words in the query; use since/until instead.
        k: number of results (1-20).
        since: only items modified/added on or after this ISO date (YYYY-MM-DD).
        until: only items modified/added on or before this ISO date (inclusive).
        doc_type: restrict to one of: pdf, note, web, docx, screenshot, bookmark.
        rerank: use a slower cross-encoder for higher precision (first use downloads a model).

    Returns ranked chunks with title, path, type, modified date, page/heading and a snippet.
    """
    k = max(1, min(int(k), 20))
    filters = Filters(parse_when(since), parse_when(until, end=True), doc_type)
    with open_db(cfg.db_path) as conn:
        results = search(conn, cfg, query, k=k, filters=filters, rerank=rerank)
    if not results:
        return [{"info": "No matches. Try broader wording or remove date/type filters."}]
    return [r.to_dict(query) for r in results]


@mcp.tool()
def list_documents(
    since: str | None = None,
    until: str | None = None,
    doc_type: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List indexed documents by recency (newest first). Good for 'what did I save last week?'."""
    f = Filters(parse_when(since), parse_when(until, end=True), doc_type)
    frag, params = f.sql()
    with open_db(cfg.db_path) as conn:
        rows = conn.execute(
            f"SELECT d.title, d.path, d.type, d.mtime FROM documents d WHERE 1=1{frag} "
            "ORDER BY d.mtime DESC LIMIT ?",
            [*params, max(1, min(int(limit), 100))],
        ).fetchall()
    from datetime import datetime

    return [{"title": r["title"], "path": r["path"], "type": r["type"],
             "modified": datetime.fromtimestamp(r["mtime"]).isoformat(timespec="minutes")} for r in rows]


@mcp.tool()
def read_document(path: str, offset: int = 0, max_chars: int = 8000) -> dict:
    """Return the extracted full text of an indexed document (path or URL from a search hit).
    Use offset to page through long documents."""
    with open_db(cfg.db_path) as conn:
        rows = conn.execute(
            "SELECT title, path, type, content FROM documents WHERE path = ? OR path LIKE ? LIMIT 5",
            (path, f"%{path}%"),
        ).fetchall()
    if not rows:
        raise ValueError(f"No indexed document matches: {path}")
    exact = [r for r in rows if r["path"] == path]
    if len(rows) > 1 and not exact:
        raise ValueError("Ambiguous path; matches: " + "; ".join(r["path"] for r in rows))
    r = (exact or rows)[0]
    content = r["content"] or ""
    max_chars = max(500, min(int(max_chars), 50000))
    return {
        "title": r["title"], "path": r["path"], "type": r["type"],
        "total_chars": len(content), "offset": offset,
        "text": content[offset: offset + max_chars],
        "truncated": offset + max_chars < len(content),
    }


@mcp.tool()
def index_status() -> dict:
    """Index size, document types and embedding model in use."""
    with open_db(cfg.db_path) as conn:
        by_type = {r["type"]: r["n"] for r in conn.execute(
            "SELECT type, COUNT(*) n FROM documents GROUP BY type")}
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        model = get_meta(conn, "embed_model")
    return {"documents": sum(by_type.values()), "chunks": chunks, "by_type": by_type,
            "embed_model": model, "roots": cfg.roots}


@mcp.tool()
def reindex() -> dict:
    """Pick up new/changed/deleted files in the configured folders (incremental)."""
    from .indexer import Indexer

    stats = Indexer(cfg).index_all()
    return stats.__dict__


def main() -> None:
    # Warm the embedding model in the background so the first query isn't slow.
    threading.Thread(target=lambda: get_embedder(cfg.embed_model).embed_query("warmup"), daemon=True).start()
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
