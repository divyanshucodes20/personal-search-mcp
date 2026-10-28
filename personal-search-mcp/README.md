# personal-search-mcp

Search your own PDFs, notes, web pages, screenshots (OCR) and bookmarks in natural language,
from the terminal **or from Claude** (via MCP). Everything runs locally; nothing leaves your machine.

```
files ──► extract ──► chunk ──► embed ──► SQLite ─┬─ FTS5 / BM25 (keywords) ─┐
                                                  └─ float32 vectors (meaning) ─┴─► RRF fusion ─► (rerank) ─► results
```

## 1. Install

Requires Python 3.11+.

```bash
cd personal-search-mcp
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .                     # core
pip install -e ".[ocr,docx]"         # optional: screenshots/scanned PDFs OCR, .docx files
```

The first real index downloads the embedding model (~130 MB, once).

## 2. Point it at your folders and index

```bash
psearch init --root ~/Documents --root ~/Notes --root ~/Downloads/papers
psearch index          # first run embeds everything; later runs only touch changed files
psearch stats
```

Bookmarks: export from your browser (Chrome/Edge: Bookmark manager → ⋮ → *Export bookmarks*;
Firefox: *Library → Import and Backup → Export Bookmarks to HTML*), then:

```bash
psearch import-bookmarks ~/Downloads/bookmarks.html          # titles + folders + URLs
psearch import-bookmarks ~/Downloads/bookmarks.html --fetch  # also download each page's text (slower)
```

Keep it fresh automatically: `psearch watch` (leave it running), or just run `psearch index` now and then.
Files deleted from disk are removed from the index on the next run.

## 3. Search from the terminal

```bash
psearch search "explanation of hybrid search"
psearch search "kubernetes networking" --type pdf --since 2026-10-11
psearch search "rank fusion" --mode bm25      # or vector / hybrid (default)
psearch search "rank fusion" --rerank         # cross-encoder, slower, more precise
```

## 4. Use it from Claude (MCP)

Find the absolute path of the server command:

```bash
which psearch-mcp                    # Windows: where psearch-mcp
```

**Claude Desktop** – edit the config file
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`,
Windows: `%APPDATA%\Claude\claude_desktop_config.json`) and add:

```json
{
  "mcpServers": {
    "personal-search": {
      "command": "/ABSOLUTE/PATH/TO/personal-search-mcp/.venv/bin/psearch-mcp"
    }
  }
}
```

Fully quit and reopen Claude Desktop. You should see the `personal-search` tools.

**Claude Code**:

```bash
claude mcp add personal-search -- /ABSOLUTE/PATH/TO/personal-search-mcp/.venv/bin/psearch-mcp
```

Then just ask: *"Find that explanation of hybrid search I read last week."* Claude turns "last week"
into `since`/`until` dates itself and calls `search_documents`; it can then open a hit with `read_document`.

Tools: `search_documents`, `list_documents`, `read_document`, `index_status`, `reindex`, `get_current_date`.

If you use a custom config location, add `"env": {"PSEARCH_HOME": "/path"}` to the server entry.

## 5. Learn by measuring

Write 20–30 queries with the file that should come back, then compare retrievers:

```bash
cp eval/queries.example.json eval/mine.json     # edit: {"query": "...", "relevant": ["part-of-filename"]}
psearch eval eval/mine.json --rerank
```

You get recall@k and MRR for BM25-only, vector-only, hybrid and hybrid+rerank. Then experiment:
`chunk_words` / `overlap_words` in `~/.personal-search/config.toml` (re-run `psearch index --rebuild`),
a different `embed_model` (e.g. `BAAI/bge-base-en-v1.5`), RRF `k`/weights in `psearch/search.py`.

## Where to read the code (learning map)

| Topic | File |
|---|---|
| Full-text search (FTS5, BM25, stemming, query building) | `psearch/search.py` → `bm25_search`, `build_match` |
| Chunking | `psearch/chunking.py` |
| Embeddings + normalisation | `psearch/embeddings.py` |
| Vector search (brute-force cosine in NumPy) | `psearch/search.py` → `vector_search` |
| Ranking / fusion / reranking | `psearch/search.py` → `rrf`, `search` |
| Incremental indexing (size+mtime, content hash) | `psearch/indexer.py` |
| MCP tool design | `psearch/server.py` |

## Config (`~/.personal-search/config.toml`)

`roots`, `db_path`, `embed_model` (`"hash"` = offline test embedder), `rerank_model`, `chunk_words`,
`overlap_words`, `extensions`, `exclude`, `ocr`, `max_file_mb`. Env overrides: `PSEARCH_HOME`,
`PSEARCH_DB`, `PSEARCH_ROOTS`, `PSEARCH_EMBED_MODEL`.

## Troubleshooting

- **Tools don't appear in Claude Desktop** – use the *absolute* path to `psearch-mcp` inside the venv; check the
  MCP log (macOS `~/Library/Logs/Claude/mcp*.log`). Stdout is reserved for the protocol; logs go to stderr.
- **First search is slow** – the model loads on first use (the server warms it in the background at startup).
- **"Index was built with embedding model X"** – you changed `embed_model`; run `psearch index --rebuild`.
- **Scanned PDFs / screenshots return nothing** – install the OCR extra: `pip install -e ".[ocr]"`.
- **`no such module: fts5`** – your Python's SQLite lacks FTS5; use python.org / Homebrew / conda Python.
- **Why not sqlite-vec?** Vectors are stored as BLOBs and searched with NumPy: no native extension to install,
  and brute force is milliseconds up to ~100k chunks. Swap `vector_search` if you outgrow it.

## Tests

```bash
pip install -e ".[dev]" && pytest
```
