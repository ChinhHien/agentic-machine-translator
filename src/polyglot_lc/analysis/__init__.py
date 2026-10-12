"""Document- and chunk-level analysis: planning, complexity, draft signals.

Everything in this package is either a single bounded Planner request
(:mod:`polyglot_lc.analysis.unified_planner`) or purely local computation
(:mod:`polyglot_lc.analysis.complexity`).
"""

from polyglot_lc.analysis.complexity import (
    Complexity,
    DraftSignals,
    draft_signals,
    heuristic_document_complexity,
    score_chunk,
)
from polyglot_lc.analysis.unified_planner import (
    DocumentPlan,
    GlossaryPair,
    analyze_document,
    build_outline,
    merge_glossary_md,
    structural_summary,
)

__all__ = [
    "Complexity",
    "DocumentPlan",
    "DraftSignals",
    "GlossaryPair",
    "analyze_document",
    "build_outline",
    "draft_signals",
    "heuristic_document_complexity",
    "merge_glossary_md",
    "score_chunk",
    "structural_summary",
]
