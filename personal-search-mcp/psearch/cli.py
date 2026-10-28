from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .config import CONFIG_PATH, load_config, save_config
from .db import open_db
from .search import Filters, parse_when, search


def _need_roots(cfg) -> None:
    if not cfg.roots:
        sys.exit("No folders configured. Run:  psearch init --root ~/Documents --root ~/Notes")


def cmd_init(args):
    cfg = load_config()
    for r in args.root:
        p = str(Path(r).expanduser().resolve())
        if p not in cfg.roots:
            cfg.roots.append(p)
    if args.embed_model:
        cfg.embed_model = args.embed_model
    path = save_config(cfg)
    print(f"Config written to {path}\nRoots: {cfg.roots}\nEmbedding model: {cfg.embed_model}")


def cmd_index(args):
    from .indexer import Indexer

    cfg = load_config()
    _need_roots(cfg)
    stats = Indexer(cfg).index_all(rebuild=args.rebuild)
    print(stats.summary())


def cmd_bookmarks(args):
    from .indexer import Indexer

    cfg = load_config()
    stats = Indexer(cfg).index_bookmarks(args.file, fetch=args.fetch)
    print(stats.summary())


def cmd_search(args):
    cfg = load_config()
    filters = Filters(parse_when(args.since), parse_when(args.until, end=True), args.type, args.path)
    with open_db(cfg.db_path) as conn:
        results = search(conn, cfg, args.query, k=args.k, mode=args.mode,
                         filters=filters, rerank=args.rerank)
    if args.json:
        print(json.dumps([r.to_dict(args.query) for r in results], indent=2, ensure_ascii=False))
        return
    if not results:
        print("No results.")
    for i, r in enumerate(results, 1):
        d = r.to_dict(args.query)
        where = f"p.{r.page}" if r.page else (r.heading or "")
        print(f"{i}. {r.title}  [{r.type}{' ' + where if where else ''}]  {d['modified'][:10]}")
        print(f"   {r.path}")
        print(f"   {d['snippet']}")
        print(f"   score={r.score:.4f} bm25#{r.bm25_rank} vec#{r.vector_rank}\n")


def cmd_watch(args):
    from .watcher import watch

    cfg = load_config()
    _need_roots(cfg)
    watch(cfg)


def cmd_stats(args):
    cfg = load_config()
    with open_db(cfg.db_path) as conn:
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        by_type = conn.execute("SELECT type, COUNT(*) n FROM documents GROUP BY type ORDER BY n DESC").fetchall()
        last = conn.execute("SELECT MAX(indexed_at) FROM documents").fetchone()[0]
    print(f"config: {CONFIG_PATH}\ndb:     {cfg.db_path}\nroots:  {cfg.roots}")
    print(f"documents={docs} chunks={chunks}")
    for r in by_type:
        print(f"  {r['type']}: {r['n']}")
    if last:
        print("last indexed:", datetime.fromtimestamp(last).isoformat(timespec="seconds"))


def cmd_eval(args):
    from .evaluate import evaluate, format_table

    cfg = load_config()
    cases = json.loads(Path(args.file).read_text(encoding="utf-8"))
    with open_db(cfg.db_path) as conn:
        print(format_table(evaluate(conn, cfg, cases, k=args.k, rerank=args.rerank)))


def cmd_serve(args):
    from .server import main as serve

    serve()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="psearch", description="Personal hybrid search engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create/update config")
    p.add_argument("--root", action="append", default=[], help="folder to index (repeatable)")
    p.add_argument("--embed-model", help="e.g. BAAI/bge-small-en-v1.5 or 'hash' for offline test mode")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("index", help="index/refresh configured folders")
    p.add_argument("--rebuild", action="store_true", help="wipe and re-embed everything")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("import-bookmarks", help="index a browser bookmarks.html export")
    p.add_argument("file")
    p.add_argument("--fetch", action="store_true", help="also download and index each page's text")
    p.set_defaults(fn=cmd_bookmarks)

    p = sub.add_parser("search", help="search from the terminal")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    p.add_argument("--since", help="ISO date, e.g. 2026-10-11")
    p.add_argument("--until", help="ISO date (inclusive)")
    p.add_argument("--type", help="pdf | note | web | docx | screenshot | bookmark")
    p.add_argument("--path", help="only paths containing this text")
    p.add_argument("--rerank", action="store_true", help="cross-encoder rerank (downloads a model)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("watch", help="re-index automatically when files change")
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("stats", help="show index statistics")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("eval", help="compare bm25 / vector / hybrid on labelled queries")
    p.add_argument("file", help="JSON file of {query, relevant[]} cases")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--rerank", action="store_true")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("serve", help="run the MCP server over stdio")
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
