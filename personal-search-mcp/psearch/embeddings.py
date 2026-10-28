"""Embedding + reranking models. Heavy imports are lazy so `hash` mode needs only numpy."""
from __future__ import annotations

import re
import threading
import zlib

import numpy as np


class Embedder:
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Deterministic bag-of-words(+bigrams) hashing. Not semantic; for tests / offline smoke runs."""

    name = "hash-384"
    dim = 384

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = re.findall(r"\w+", text.lower())
        for f in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            v[zlib.crc32(f.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return self._vec(text)


class STEmbedder(Embedder):
    """sentence-transformers model (default: BAAI/bge-small-en-v1.5, ~130MB, runs fine on CPU)."""

    def __init__(self, name: str):
        self.name = name
        self._model = None
        self._lock = threading.Lock()
        low = name.lower()
        if "bge" in low:
            self.q_prefix, self.d_prefix = "Represent this sentence for searching relevant passages: ", ""
        elif "e5" in low:
            self.q_prefix, self.d_prefix = "query: ", "passage: "
        else:
            self.q_prefix = self.d_prefix = ""

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name)
        return self._model

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._load().encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts):
        return self._encode([self.d_prefix + t for t in texts])

    def embed_query(self, text):
        return self._encode([self.q_prefix + text])[0]


_embedders: dict[str, Embedder] = {}


def get_embedder(name: str) -> Embedder:
    if name not in _embedders:
        _embedders[name] = HashEmbedder() if name in ("hash", "hash-384") else STEmbedder(name)
    return _embedders[name]


class Reranker:
    """Cross-encoder: scores (query, passage) pairs jointly. Slower but more precise than bi-encoders."""

    def __init__(self, name: str):
        self.name = name
        self._model = None
        self._lock = threading.Lock()

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.name)
        return [float(s) for s in self._model.predict(pairs)]


_rerankers: dict[str, Reranker] = {}


def get_reranker(name: str) -> Reranker:
    if name not in _rerankers:
        _rerankers[name] = Reranker(name)
    return _rerankers[name]
