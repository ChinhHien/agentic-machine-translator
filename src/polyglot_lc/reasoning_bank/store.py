"""Reasoning Bank store on ChromaDB (persistent, local)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from loguru import logger

from polyglot_lc.reasoning_bank.embedder import Embedder
from polyglot_lc.reasoning_bank.schema import MIN_UPSERT_SCORE, ReasoningTrace


class ReasoningBank:
    """Persistent trace store: retrieve similar strategies, upsert successes."""

    def __init__(self, path: str | Path, embedder: Embedder) -> None:
        self.path = Path(path)
        self.embedder = embedder
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._col = self._client.get_or_create_collection(
            name="reasoning_traces",
            metadata={"hnsw:space": "cosine"},
        )
        self._reset_if_dimension_mismatch()

    def _reset_if_dimension_mismatch(self) -> None:
        """Recreate the collection when the embedding dimension changed.

        Happens e.g. after switching EMBED_MODEL or moving to/from mock mode.
        Silently keeping stale vectors would break every query, so the bank is
        reset (logged as a warning) instead of failing mid-run.
        """
        try:
            if self._col.count() == 0:
                return
            probe = self.embedder.embed_one("dimension probe")
            # A cheap upsert/retrieve roundtrip on a throwaway id.
            self._col.query(query_embeddings=[probe], n_results=1)
        except Exception as exc:  # noqa: BLE001 - dimension/type mismatches
            logger.warning(
                "Reasoning Bank incompatible with current embeddings ({}) — "
                "resetting collection at {}",
                exc,
                self.path,
            )
            self._client.delete_collection("reasoning_traces")
            self._col = self._client.get_or_create_collection(
                name="reasoning_traces",
                metadata={"hnsw:space": "cosine"},
            )

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        where: dict[str, Any] | None = None,
    ) -> list[ReasoningTrace]:
        """Return the *top_k* most similar traces for *query*."""
        if self._col.count() == 0:
            return []
        qvec = self.embedder.embed_one(query)
        res = self._col.query(
            query_embeddings=[qvec],
            n_results=min(top_k, self._col.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        traces: list[ReasoningTrace] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, trace_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            doc = docs[i] if i < len(docs) else ""
            dist = dists[i] if i < len(dists) else None
            traces.append(
                ReasoningTrace.from_metadata(
                    id=trace_id, document=doc, metadata=meta, distance=dist
                )
            )
        return traces

    def upsert(self, trace: ReasoningTrace) -> bool:
        """Upsert *trace*; only traces with score >= 0.75 are accepted.

        Returns True when stored, False when rejected by the score gate.
        """
        if trace.score < MIN_UPSERT_SCORE:
            logger.debug(
                "Trace {} rejected (score {:.2f} < {})",
                trace.id,
                trace.score,
                MIN_UPSERT_SCORE,
            )
            return False
        vec = self.embedder.embed_one(trace.document_text())
        self._col.upsert(
            ids=[trace.id],
            embeddings=[vec],
            documents=[trace.document_text()],
            metadatas=[trace.metadata()],
        )
        logger.debug("Trace {} upserted (score {:.2f})", trace.id, trace.score)
        return True

    def count(self) -> int:
        """Number of stored traces."""
        return self._col.count()

    def clear(self) -> None:
        """Delete all traces (used by tests)."""
        self._client.delete_collection("reasoning_traces")
        self._col = self._client.get_or_create_collection(
            name="reasoning_traces",
            metadata={"hnsw:space": "cosine"},
        )
