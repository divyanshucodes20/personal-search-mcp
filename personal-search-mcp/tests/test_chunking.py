from psearch.chunking import chunk_sections, chunk_text
from psearch.extractors import Section


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_text_single_chunk():
    assert len(chunk_text("hello world", 220, 40)) == 1


def test_long_text_splits_with_overlap():
    text = "\n\n".join(words(60, f"p{i}_") for i in range(10))  # 600 words, 10 paragraphs
    chunks = chunk_text(text, max_words=200, overlap=30)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 200 + 30 for c in chunks)
    tail = chunks[0].split()[-5:]
    assert all(t in chunks[1] for t in tail)  # overlap carried into the next chunk


def test_giant_paragraph_is_split():
    assert len(chunk_text(words(1000), max_words=200, overlap=20)) >= 5


def test_tiny_sections_merge():
    secs = [Section("short one", page=1), Section(words(80), page=2)]
    chunks = chunk_sections(secs, 220, 40)
    assert len(chunks) == 1 and chunks[0].page == 1
