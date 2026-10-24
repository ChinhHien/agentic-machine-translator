"""Section helpers shared with the document planner.

This module used to build a summary tree with the Planner LLM (one request per
section, plus one for the document), which the pipeline then fed to a second
planning request. The unified Planner now produces the whole document plan in
one request, so the tree itself is gone; what remains is the way both the old
path and the new one describe a document cheaply: grouping chunks by section and
previewing text without a model call.
"""

from __future__ import annotations

from polyglot_lc.segment.chunker import Chunk


def build_sections(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by their section path (preserving order)."""
    sections: dict[str, list[Chunk]] = {}
    for c in chunks:
        sections.setdefault(c.section_path or "(root)", []).append(c)
    return sections


def preview_text(text: str, limit: int = 160) -> str:
    """Cheap non-LLM single-line preview, truncated with an ellipsis.

    Used by the unified planner's section outline, so every section is shown to
    the Planner the same way.
    """
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")
