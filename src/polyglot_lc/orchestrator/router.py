"""Conditional routing rules for the translation graph."""

from __future__ import annotations

from typing import Any


def after_critic(
    state: dict[str, Any],
    threshold: float,
    max_rounds: int,
    refine_min_score: float | None = None,
) -> str:
    """critic -> refiner (score < threshold, rounds left) | verifier.

    Adaptive refiner gate: when ``refine_min_score`` is set and the draft scores
    at or above it, skip the refiner and go straight to the verifier. Default
    ``None`` preserves the original threshold-only behaviour.
    """
    critique = state.get("critique")
    score = getattr(critique, "score", 10.0)
    if state.get("refine_round", 0) >= max_rounds:
        return "verify"
    if refine_min_score is not None and score >= refine_min_score:
        return "verify"
    if score < threshold:
        return "refine"
    return "verify"


def after_verifier(state: dict[str, Any], max_retries: int = 1) -> str:
    """verifier -> retranslate (with critique, bounded) | next_chunk/assemble."""
    vr = state.get("verify_result")
    ok = bool(vr) if vr is not None else True
    if not ok and state.get("retranslate_count", 0) < max_retries:
        return "retranslate"
    return "next"


def has_next_chunk(state: dict[str, Any]) -> str:
    """next_chunk -> translate (more chunks) | assemble (done)."""
    if state.get("current_chunk", 0) < state.get("total_chunks", 0):
        return "translate"
    return "assemble"
