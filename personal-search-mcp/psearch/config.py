from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

HOME_DIR = Path(os.environ.get("PSEARCH_HOME", str(Path.home() / ".personal-search")))
CONFIG_PATH = HOME_DIR / "config.toml"

DEFAULT_EXTENSIONS = [
    ".pdf", ".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".docx",
    ".png", ".jpg", ".jpeg", ".webp",
]
DEFAULT_EXCLUDES = ["node_modules", "__pycache__", ".git", ".obsidian", ".venv", "*.tmp"]


@dataclass
class Config:
    roots: list[str] = field(default_factory=list)
    db_path: str = str(HOME_DIR / "index.db")
    embed_model: str = "BAAI/bge-small-en-v1.5"   # use "hash" for a tiny offline test embedder
    rerank_model: str = "BAAI/bge-reranker-base"
    chunk_words: int = 220
    overlap_words: int = 40
    extensions: list[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    ocr: bool = True
    max_file_mb: int = 100


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else CONFIG_PATH
    if p.exists():
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        names = {f.name for f in fields(Config)}
        for k, v in data.items():
            if k in names:
                setattr(cfg, k, v)
    if os.environ.get("PSEARCH_DB"):
        cfg.db_path = os.environ["PSEARCH_DB"]
    if os.environ.get("PSEARCH_EMBED_MODEL"):
        cfg.embed_model = os.environ["PSEARCH_EMBED_MODEL"]
    if os.environ.get("PSEARCH_ROOTS"):
        cfg.roots = [r for r in os.environ["PSEARCH_ROOTS"].split(os.pathsep) if r]
    cfg.roots = [str(Path(r).expanduser()) for r in cfg.roots]
    cfg.db_path = str(Path(cfg.db_path).expanduser())
    return cfg


def save_config(cfg: Config, path: str | Path | None = None) -> Path:
    p = Path(path) if path else CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# personal-search-mcp configuration", ""]
    for key, value in asdict(cfg).items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        else:
            lines.append(f"{key} = {json.dumps(value)}")  # valid TOML for str/int/list[str]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
