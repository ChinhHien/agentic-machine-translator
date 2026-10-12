"""Translation memory: exact + fuzzy reuse within one run."""

from __future__ import annotations

import threading
from difflib import SequenceMatcher

Pairs = list[tuple[str, str]]


class TranslationMemory:
    """Simple in-memory TM; seeded from the Reasoning Bank's few-shots.

    Chunk processing can run concurrently (``MAX_CONCURRENCY``), and a single
    TM instance is shared across those threads, so all access is locked.
    """

    def __init__(self, min_similarity: float = 0.85) -> None:
        self.min_similarity = min_similarity
        self._store: dict[str, Pairs] = {}
        self._lock = threading.Lock()

    def store(self, src: str, tgt: str, src_lang: str = "", tgt_lang: str = "") -> None:
        """Store a source/target pair under the given language pair."""
        if not src.strip() or not tgt.strip():
            return
        key = f"{src_lang or 'auto'}->{tgt_lang or 'auto'}"
        with self._lock:
            self._store.setdefault(key, []).append((src.strip(), tgt.strip()))

    def lookup(self, src: str, src_lang: str = "", tgt_lang: str = "", top: int = 2) -> Pairs:
        """Return up to *top* matches (exact first, then fuzzy >= threshold)."""
        key = f"{src_lang or 'auto'}->{tgt_lang or 'auto'}"
        with self._lock:
            pairs = list(self._store.get(key, []))
        if not pairs:
            return []
        exact = [p for p in pairs if p[0] == src.strip()]
        if exact:
            return exact[:top]
        fuzzy: list[tuple[float, tuple[str, str]]] = []
        for p in pairs:
            ratio = SequenceMatcher(None, src[:400], p[0][:400]).ratio()
            if ratio >= self.min_similarity:
                fuzzy.append((ratio, p))
        fuzzy.sort(key=lambda x: -x[0])
        return [p for _, p in fuzzy[:top]]

    def __len__(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._store.values())
