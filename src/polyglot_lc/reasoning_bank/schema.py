"""Reasoning Bank schema: a trace of a successful (or failed) translation run."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

MIN_UPSERT_SCORE = 0.75


class ReasoningTrace(BaseModel):
    """One retrieval-ready reasoning record stored in ChromaDB."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    domain: str = "general"
    src_lang: str = "auto"
    tgt_lang: str = "vi"
    strategy: str = ""
    few_shot: list[str] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    score: float = 0.0  # 0..1 quality gate for upsert
    ts: float = Field(default_factory=time.time)
    meta: dict[str, Any] = Field(default_factory=dict)

    def document_text(self) -> str:
        """Text used for embedding the trace."""
        return f"[{self.domain}] {self.src_lang}->{self.tgt_lang}: {self.strategy}"

    def metadata(self) -> dict[str, Any]:
        """Flattened metadata for ChromaDB (lists joined; chroma-safe types)."""
        return {
            "domain": self.domain,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "strategy": self.strategy[:500],
            "few_shot": " ||| ".join(self.few_shot)[:2000],
            "pitfalls": " ||| ".join(self.pitfalls)[:1000],
            "score": float(self.score),
            "ts": float(self.ts),
        }

    @classmethod
    def from_metadata(
        cls,
        id: str,  # name mandated by chromadb's query result row
        document: str,
        metadata: dict[str, Any],
        distance: float | None = None,
    ) -> ReasoningTrace:
        """Rebuild a trace from a chroma query result row."""
        split = lambda s: [p for p in (s or "").split(" ||| ") if p]
        return cls(
            id=id,
            domain=metadata.get("domain", "general"),
            src_lang=metadata.get("src_lang", "auto"),
            tgt_lang=metadata.get("tgt_lang", ""),
            strategy=document or metadata.get("strategy", ""),
            few_shot=split(metadata.get("few_shot", "")),
            pitfalls=split(metadata.get("pitfalls", "")),
            score=float(metadata.get("score", 0.0)),
            ts=float(metadata.get("ts", 0.0)),
            meta={"distance": distance} if distance is not None else {},
        )
