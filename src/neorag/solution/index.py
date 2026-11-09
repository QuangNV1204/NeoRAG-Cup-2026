from __future__ import annotations

import contextlib
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

EmbedFn = Callable[[list[str]], np.ndarray]

def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)

class DenseIndex:
    def __init__(
        self,
        embed: EmbedFn,
        cache_dir: str | Path,
        embedding_model: str = "bge-m3",
        batch_size: int = 32,
    ):
        self.embed = embed
        self.cache_dir = Path(cache_dir)
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.matrix = np.zeros((0, 0), dtype=np.float32)

    def _cache_path(self, texts: Sequence[str], name: str) -> Path:
        raw = json.dumps(
            [self.embedding_model, name, list(texts)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / "embeddings" / f"{name}-{digest}.npz"

    def fit(self, texts: Sequence[str], name: str = "index") -> DenseIndex:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(texts, name)
        if path.exists():
            try:
                loaded = np.load(path)
                matrix = loaded["matrix"].astype(np.float32)
                if matrix.shape[0] == len(texts):
                    self.matrix = _normalize_rows(matrix)
                    return self
            except (OSError, KeyError, ValueError):
                pass

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors.append(np.asarray(self.embed(batch), dtype=np.float32))
        matrix = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
        self.matrix = _normalize_rows(matrix)
        with contextlib.suppress(OSError):
            np.savez_compressed(path, matrix=self.matrix)
        return self

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.asarray(self.embed([text]), dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        vector = _normalize_rows(vector)[0]
        return vector

    def scores(self, query_vector: np.ndarray) -> np.ndarray:
        if self.matrix.size == 0:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.matrix @ query_vector, dtype=np.float32)

class BM25Index:
    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: Callable[[str], list[str]],
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.docs = [tokenizer(text) for text in texts]
        self.lengths = np.asarray([len(doc) for doc in self.docs], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if len(self.lengths) else 1.0
        self.tfs = [Counter(doc) for doc in self.docs]

        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n_docs = max(1, len(self.docs))
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        query_terms = self.tokenizer(query)
        result = np.zeros(len(self.docs), dtype=np.float32)
        if not query_terms or not self.docs:
            return result
        for index, (tf, length) in enumerate(zip(self.tfs, self.lengths, strict=False)):
            score = 0.0
            denominator_norm = self.k1 * (
                1.0 - self.b + self.b * float(length) / max(self.avg_length, 1e-9)
            )
            for term in query_terms:
                frequency = tf.get(term)
                if not frequency:
                    continue
                score += (
                    self.idf.get(term, 0.0)
                    * frequency
                    * (self.k1 + 1.0)
                    / (frequency + denominator_norm)
                )
            result[index] = score
        return result

def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)

__all__ = ["BM25Index", "DenseIndex", "normalize01"]
