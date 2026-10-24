"""Per-chunk state flowing through the chunk-level LangGraph."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class ChunkState(TypedDict, total=False):
    """State for one chunk: Translate -> [Critic -> Refiner] -> Verify -> Collect.

    Document-level values (strategy, glossary, language pair) are copied into
    each chunk's state so independent chunks can run concurrently without
    sharing mutable state. The adaptive routing fields are per chunk: the
    coordinator computes ``complexity`` before dispatch, and the Translator node
    derives ``draft_score`` / ``fast_path`` from the draft it just produced.
    """

    # Document-level context (read-only during chunk processing)
    chunk: Any  # segment.chunker.Chunk
    src_lang: str
    tgt_lang: str
    strategy: str
    few_shot: list[str]
    pitfalls: list[str]
    glossary_md: str
    #: Promoted SkillOpt skill for this pair (dict) or None. Document-level,
    #: copied into every chunk: LangGraph drops keys this state does not carry.
    skill: dict[str, Any] | None

    # Adaptive routing (per chunk)
    complexity: float | None  # 0..1 difficulty; None disables the fast path
    section_note: str  # Planner guidance for this chunk's section
    draft_score: float  # deterministic draft quality, 0-10
    draft_issues: list[str]
    #: Deterministic hard-check problems for ``draft``, computed once when the
    #: draft is prepared (see ``make_chunk_graph.prepare_draft``) and reused by
    #: the faster and the LLM verify path. ``None`` means "not computed".
    draft_hard_issues: list[str]
    fast_path: bool  # True -> skip Critic/Refiner, verify directly
    skip_llm_verify: bool  # True -> deterministic hard checks only

    # Per-chunk working set
    draft: str
    #: The draft a failed attempt already produced, handed to the next attempt
    #: so a chunk that is run again is not translated again. Declared here
    #: because LangGraph drops any key this state does not carry.
    kept_draft: str
    critique: Any  # Critique | None
    refine_round: int
    verify_result: Any  # VerifyResult | None
    retranslate_count: int  # semantic retranslate retries (bounded)

    # Outputs
    translated: str  # final draft for this chunk
    path: str  # fast | standard | deep (which route ran)
    trace: dict[str, Any]  # reasoning-bank trace (for scoring)
    stored: bool  # whether the trace passed the RB gate
