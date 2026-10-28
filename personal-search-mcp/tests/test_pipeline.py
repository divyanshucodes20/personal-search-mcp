import os
import time
from datetime import datetime, timedelta

import pytest

from psearch.bookmarks import parse_bookmarks
from psearch.config import Config
from psearch.db import connect
from psearch.indexer import Indexer
from psearch.search import Filters, parse_when, search

DOCS = {
    "hybrid.md": (
        "# Hybrid search\n\nHybrid search combines BM25 keyword matching with dense vector "
        "embeddings. Results from both retrievers are merged with reciprocal rank fusion, which "
        "uses ranks rather than raw scores so the two scales never need normalising. " * 3
    ),
    "pasta.md": (
        "# Pasta night\n\nBoil salted water, cook the spaghetti until al dente, then toss it with "
        "garlic, olive oil and chilli flakes. Reserve a cup of the starchy pasta water. " * 3
    ),
    "k8s.txt": (
        "Kubernetes schedules containers onto nodes. A deployment manages replica sets and rolling "
        "updates, while services provide stable networking for pods. " * 3
    ),
}


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    for name, text in DOCS.items():
        (root / name).write_text(text)
    cfg = Config(roots=[str(root)], db_path=str(tmp_path / "db.sqlite"), embed_model="hash")
    conn = connect(cfg.db_path)
    return cfg, conn, Indexer(cfg, conn=conn), root


def test_index_and_search(env):
    cfg, conn, idx, root = env
    stats = idx.index_all()
    assert stats.new == 3 and stats.failed == 0
    for mode in ("bm25", "vector", "hybrid"):
        top = search(conn, cfg, "how does reciprocal rank fusion merge results", mode=mode)[0]
        assert top.path.endswith("hybrid.md"), mode
    assert search(conn, cfg, "spaghetti recipe")[0].path.endswith("pasta.md")


def test_stemming_in_bm25(env):
    cfg, conn, idx, _ = env
    idx.index_all()
    assert search(conn, cfg, "merging retrievers", mode="bm25")[0].path.endswith("hybrid.md")


def test_incremental_and_update(env):
    cfg, conn, idx, root = env
    idx.index_all()
    assert idx.index_all().unchanged == 3
    (root / "pasta.md").write_text("# Pasta night\n\nNow it is only about risotto and arborio rice. " * 10)
    stats = idx.index_all()
    assert stats.updated == 1 and stats.unchanged == 2
    assert search(conn, cfg, "arborio risotto", mode="bm25")[0].path.endswith("pasta.md")
    assert not search(conn, cfg, "spaghetti", mode="bm25")


def test_prune_deleted_file(env):
    cfg, conn, idx, root = env
    idx.index_all()
    (root / "k8s.txt").unlink()
    assert idx.index_all().removed == 1
    assert not search(conn, cfg, "kubernetes deployment", mode="bm25")


def test_date_filters(env):
    cfg, conn, idx, root = env
    old = time.time() - 30 * 86400
    os.utime(root / "pasta.md", (old, old))
    idx.index_all()
    week_ago = (datetime.now() - timedelta(days=7)).date().isoformat()
    f = Filters(since=parse_when(week_ago))
    for mode in ("bm25", "vector"):
        paths = [r.path for r in search(conn, cfg, "pasta spaghetti water", mode=mode, filters=f)]
        assert not any(p.endswith("pasta.md") for p in paths), mode
    f_old = Filters(until=parse_when(week_ago, end=True))
    assert search(conn, cfg, "pasta spaghetti", filters=f_old)[0].path.endswith("pasta.md")


def test_type_filter_and_per_doc(env):
    cfg, conn, idx, _ = env
    idx.index_all()
    assert all(r.type == "note" for r in search(conn, cfg, "search", filters=Filters(doc_type="note")))
    res = search(conn, cfg, "hybrid search", k=10, per_doc=1)
    assert len({r.doc_id for r in res}) == len(res)


def test_model_mismatch_requires_rebuild(env):
    cfg, conn, idx, _ = env
    idx.index_all()
    from psearch.embeddings import STEmbedder

    with pytest.raises(RuntimeError, match="--rebuild"):
        Indexer(cfg, conn=conn, embedder=STEmbedder("some/other-model"))


def test_bookmarks(env, tmp_path):
    cfg, conn, idx, _ = env
    html = """<DL><p><DT><H3>Dev</H3><DL><p>
    <DT><A HREF="https://example.com/rrf" ADD_DATE="1760000000">Reciprocal Rank Fusion explained</A>
    </DL><p><DT><A HREF="https://example.org/x">Top level</A></DL><p>"""
    items = parse_bookmarks(html)
    assert [(b.url, b.folder) for b in items] == [("https://example.com/rrf", "Dev"), ("https://example.org/x", "")]
    f = tmp_path / "bookmarks.html"
    f.write_text(html)
    assert idx.index_bookmarks(f).new == 2
    assert search(conn, cfg, "reciprocal rank fusion", filters=Filters(doc_type="bookmark"))[0].path == "https://example.com/rrf"
    assert idx.index_bookmarks(f).unchanged == 2
