"""Turn extracted sections into overlapping, paragraph-aware chunks (sized in words)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .extractors import Section

MIN_SECTION_WORDS = 60  # tiny sections (e.g. a short PDF page) get merged with neighbours


@dataclass
class Chunk:
    text: str
    page: int | None = None
    heading: str | None = None


def _split_long(paragraph: str, max_words: int) -> list[str]:
    out, cur, n = [], [], 0
    for sent in re.split(r"(?<=[.!?])\s+", paragraph):
        w = len(sent.split())
        if w > max_words:  # one giant "sentence": hard split by words
            if cur:
                out.append(" ".join(cur)); cur, n = [], 0
            words = sent.split()
            out.extend(" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words))
            continue
        if n + w > max_words and cur:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(sent); n += w
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_text(text: str, max_words: int = 220, overlap: int = 40) -> list[str]:
    units: list[str] = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if not p:
            continue
        units.extend(_split_long(p, max_words) if len(p.split()) > max_words else [p])

    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    real_units = 0  # units in `cur` beyond the carried-over overlap tail
    for u in units:
        w = len(u.split())
        if n + w > max_words and real_units:
            chunks.append("\n\n".join(cur))
            tail = " ".join(" ".join(cur).split()[-overlap:]) if overlap else ""
            cur = [tail] if tail else []
            n = len(tail.split())
            real_units = 0
        cur.append(u); n += w; real_units += 1
    if real_units:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_sections(sections: list[Section], max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    merged: list[Section] = []
    carry: Section | None = None
    for s in sections:
        if carry:
            s = Section(carry.text + "\n\n" + s.text, page=carry.page or s.page,
                        heading=carry.heading or s.heading)
            carry = None
        if len(s.text.split()) < MIN_SECTION_WORDS:
            carry = s
            continue
        merged.append(s)
    if carry:
        if merged:
            last = merged[-1]
            merged[-1] = Section(last.text + "\n\n" + carry.text, last.page, last.heading)
        else:
            merged.append(carry)

    out: list[Chunk] = []
    for s in merged:
        for c in chunk_text(s.text, max_words, overlap):
            out.append(Chunk(c, s.page, s.heading))
    return out
