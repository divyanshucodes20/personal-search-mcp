"""Text extraction for each supported file type. Returns sections with page/heading metadata."""
from __future__ import annotations

import io
import re
import sys
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


@dataclass
class Section:
    text: str
    page: int | None = None
    heading: str | None = None


@dataclass
class Extracted:
    title: str
    doc_type: str
    sections: list[Section]


TYPE_BY_EXT = {
    ".pdf": "pdf",
    ".md": "note", ".markdown": "note", ".txt": "note", ".rst": "note",
    ".html": "web", ".htm": "web",
    ".docx": "docx",
    ".png": "screenshot", ".jpg": "screenshot", ".jpeg": "screenshot", ".webp": "screenshot",
}


def warn(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- OCR
_ocr_engine = None
_ocr_warned = False


def ocr_image(source: str | bytes) -> str:
    """OCR a path or raw image bytes. Uses rapidocr if installed, else pytesseract."""
    global _ocr_engine, _ocr_warned
    try:
        from rapidocr_onnxruntime import RapidOCR

        if _ocr_engine is None:
            _ocr_engine = RapidOCR()
        result, _ = _ocr_engine(source)
        return "\n".join(r[1] for r in (result or []))
    except ImportError:
        pass
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
        return pytesseract.image_to_string(img)
    except ImportError:
        if not _ocr_warned:
            warn("OCR unavailable: install with  pip install 'personal-search-mcp[ocr]'")
            _ocr_warned = True
        return ""


# ---------------------------------------------------------------- HTML
class _TextParser(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n\n" if tag != "br" else "\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    try:
        import trafilatura

        text = trafilatura.extract(raw, include_comments=False, include_tables=True)
        if text:
            return text
    except Exception:
        pass
    p = _TextParser()
    p.feed(raw)
    return re.sub(r"\n{3,}", "\n\n", "".join(p.parts)).strip()


def _extract_html(path: Path) -> Extracted:
    raw = read_text(path)
    m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
    title = unescape(m.group(1)).strip() if m else path.stem
    return Extracted(title or path.stem, "web", [Section(html_to_text(raw))])


# ---------------------------------------------------------------- Markdown / text
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def _extract_markdown(path: Path) -> Extracted:
    raw = read_text(path)
    title = None
    m = _FRONT_MATTER.match(raw)
    if m:
        for line in m.group(1).splitlines():
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip().strip("\"'")
        raw = raw[m.end():]

    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []
    in_fence = False

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(text, heading=heading))
        buf.clear()

    for line in raw.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and re.match(r"#{1,6}\s+\S", line):
            flush()
            heading = line.lstrip("#").strip()
            if title is None and line.startswith("# "):
                title = heading
            continue
        buf.append(line)
    flush()
    return Extracted(title or path.stem, "note", sections)


def _extract_text(path: Path) -> Extracted:
    return Extracted(path.stem, "note", [Section(read_text(path))])


# ---------------------------------------------------------------- PDF
def _extract_pdf(path: Path, cfg) -> Extracted | None:
    import pymupdf  # (the old `fitz` alias prints a warning to stdout, which breaks MCP stdio)

    doc = pymupdf.open(str(path))
    title = (doc.metadata or {}).get("title") or path.stem
    sections = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            sections.append(Section(text, page=i))
    if sum(len(s.text) for s in sections) < 50 and (cfg is None or cfg.ocr):
        sections = []  # likely a scanned PDF -> OCR each page
        for i, page in enumerate(doc, start=1):
            text = ocr_image(page.get_pixmap(dpi=200).tobytes("png")).strip()
            if text:
                sections.append(Section(text, page=i))
    doc.close()
    return Extracted(title, "pdf", sections)


# ---------------------------------------------------------------- DOCX
def _extract_docx(path: Path) -> Extracted | None:
    try:
        import docx
    except ImportError:
        warn("python-docx missing: pip install 'personal-search-mcp[docx]'")
        return None
    d = docx.Document(str(path))
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []

    def flush():
        text = "\n\n".join(b for b in buf if b.strip())
        if text.strip():
            sections.append(Section(text, heading=heading))
        buf.clear()

    for p in d.paragraphs:
        style = p.style.name if p.style is not None and p.style.name else ""
        if style.startswith("Heading") and p.text.strip():
            flush()
            heading = p.text.strip()
        else:
            buf.append(p.text)
    flush()
    title = (d.core_properties.title or "").strip() or path.stem
    return Extracted(title, "docx", sections)


# ---------------------------------------------------------------- images
def _extract_image(path: Path, cfg) -> Extracted | None:
    if cfg is not None and not cfg.ocr:
        return None
    text = ocr_image(str(path)).strip()
    return Extracted(path.stem, "screenshot", [Section(text)] if text else [])


# ---------------------------------------------------------------- dispatcher
def extract(path: Path, cfg=None) -> Extracted | None:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path, cfg)
    if ext in {".md", ".markdown"}:
        return _extract_markdown(path)
    if ext in {".txt", ".rst"}:
        return _extract_text(path)
    if ext in {".html", ".htm"}:
        return _extract_html(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return _extract_image(path, cfg)
    return None
