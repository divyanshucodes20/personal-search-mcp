"""Import browser bookmark exports (Netscape HTML format: Chrome, Edge, Firefox, Safari)."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

from .extractors import html_to_text


@dataclass
class Bookmark:
    url: str
    title: str
    add_date: float | None
    folder: str


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items: list[Bookmark] = []
        self.stack: list[str] = []
        self._pending_folder = ""
        self._in_h3 = False
        self._in_a = False
        self._attrs: dict = {}
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): v for k, v in attrs}
        if tag == "h3":
            self._in_h3, self._buf = True, []
        elif tag == "a" and a.get("href"):
            self._in_a, self._attrs, self._buf = True, a, []
        elif tag == "dl":
            self.stack.append(self._pending_folder)
            self._pending_folder = ""

    def handle_endtag(self, tag):
        if tag == "h3" and self._in_h3:
            self._pending_folder = "".join(self._buf).strip()
            self._in_h3 = False
        elif tag == "a" and self._in_a:
            url = self._attrs.get("href", "")
            if url.startswith(("http://", "https://")):
                try:
                    added = float(self._attrs.get("add_date") or 0) or None
                except ValueError:
                    added = None
                title = "".join(self._buf).strip() or url
                folder = " / ".join(f for f in self.stack if f)
                self.items.append(Bookmark(url, title, added, folder))
            self._in_a = False
        elif tag == "dl" and self.stack:
            self.stack.pop()

    def handle_data(self, data):
        if self._in_h3 or self._in_a:
            self._buf.append(data)


def parse_bookmarks(html: str) -> list[Bookmark]:
    p = _Parser()
    p.feed(html)
    return p.items


def fetch_page_text(url: str, timeout: float = 10, max_bytes: int = 2_000_000) -> str:
    """Best-effort download + readable-text extraction. Returns '' on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 personal-search-mcp"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if "html" not in (resp.headers.get("Content-Type") or "").lower():
                return ""
            charset = resp.headers.get_content_charset() or "utf-8"
            raw = resp.read(max_bytes).decode(charset, errors="replace")
        return html_to_text(raw)
    except Exception:
        return ""
