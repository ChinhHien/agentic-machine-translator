"""Per-chunk state flowing through the chunk-level LangGraph."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class ChunkState(TypedDict, total=False):
    """State for one chunk: Translate -> Critic -> [Refiner -> Critic] -> Verify.

    Document-level values (strategy, glossary, language pair) are copied into
    each chunk's state so independent chunks can run concurrently without
    sharing mutable state.
    """

    # Document-level context (read-only during chunk processing)
    chunk: Any                       # segment.chunker.Chunk
    src_lang: str
    tgt_lang: str
    domain: str
    strategy: str
    few_shot: list[str]
    pitfalls: list[str]
    glossary_md: str

    # Per-chunk working set
    draft: str
    critique: Any                    # Critique | None
    refine_round: int
    verify_result: Any               # VerifyResult | None
    retranslate_count: int           # semantic retranslate retries (bounded)

    # Outputs
    translated: str                  # final draft for this chunk
    trace: dict[str, Any]            # reasoning-bank trace (for scoring)
    stored: bool                     # whether the trace passed the RB gate
