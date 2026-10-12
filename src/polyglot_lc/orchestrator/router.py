"""Conditional routing rules for the translation graph."""

from __future__ import annotations

from typing import Any

#: Route names returned to the LangGraph conditional edges.
ROUTE_CRITIC = "critic"
ROUTE_VERIFY = "verify"


def after_translate(state: dict[str, Any]) -> str:
    """translator -> verifier (fast path) | critic (full quality path).

    The Translator node decides ``fast_path`` from the chunk's complexity and
    from deterministic draft signals; the edge only reads that decision, so the
    routing rule stays a pure function of state.
    """
    return ROUTE_VERIFY if state.get("fast_path") else ROUTE_CRITIC


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
    """verifier -> retranslate (bounded) | collect.

    There is no document-level edge here: each chunk has its own graph, and the
    coordinator assembles the finished chunks, so a chunk never routes to the
    next one.
    """
    vr = state.get("verify_result")
    ok = bool(vr) if vr is not None else True
    if not ok and state.get("retranslate_count", 0) < max_retries:
        return "retranslate"
    return "next"
