"""Embedder for the Reasoning Bank: remote API with deterministic fallback.

When the embedding endpoint is unreachable (or in mock/offline mode), a
deterministic hash-based vector is used so retrieval still works consistently
without network access.
"""

from __future__ import annotations

import hashlib
import re

from loguru import logger

from polyglot_lc.config import EmbedConfig
from polyglot_lc.llm.client import LLMClient

_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder:
    """Embed texts via the configured endpoint, or a hash fallback offline."""

    def __init__(self, config: EmbedConfig) -> None:
        self.config = config
        self.client = LLMClient(config)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch; falls back to hash vectors on any failure."""
        if not texts:
            return []
        try:
            resp = self.client.embed(inputs=texts)
            return resp.vectors
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.warning("Embedding endpoint failed ({}); using hash fallback", exc)
            return [self._hash_vector(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    @staticmethod
    def _hash_vector(text: str) -> list[float]:
        """Deterministic feature-hashed vector (stable, overlap-sensitive).

        Words and character trigrams are hashed into a fixed-dim bag vector,
        so texts sharing vocabulary get higher cosine similarity — good
        enough for offline retrieval demos and tests.
        """
        vec = [0.0] * _DIM
        tokens = _TOKEN_RE.findall(text.lower())
        features: list[str] = []
        for tok in tokens:
            features.append(tok)
            if len(tok) > 3:
                features.extend(tok[i : i + 3] for i in range(len(tok) - 2))
        for feat in features:
            h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % _DIM
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]
