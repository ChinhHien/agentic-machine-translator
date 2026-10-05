"""Summary tree: doc -> section -> chunk, summarized by the Planner LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from polyglot_lc.segment.chunker import Chunk

if TYPE_CHECKING:
    from polyglot_lc.agents.base import Agent


@dataclass
class SummaryNode:
    """A node in the document summary tree."""

    level: str  # "doc" | "section" | "chunk"
    title: str
    summary: str = ""
    children: list["SummaryNode"] = field(default_factory=list)
    meta: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}

    def render(self, indent: int = 0) -> str:
        """ASCII render used by the Planner prompt and debugging."""
        pad = "  " * indent
        lines = [f"{pad}[{self.level}] {self.title}" + (f": {self.summary}" if self.summary else "")]
        for ch in self.children:
            lines.append(ch.render(indent + 1))
        return "\n".join(lines)


def build_sections(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by their section path (preserving order)."""
    sections: dict[str, list[Chunk]] = {}
    for c in chunks:
        sections.setdefault(c.section_path or "(root)", []).append(c)
    return sections


def _chunk_preview(text: str, limit: int = 160) -> str:
    """Cheap non-LLM preview used when per-chunk summaries are disabled."""
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def summarize_with_planner(
    chunks: list[Chunk],
    planner: "Agent",
    doc_title: str = "Document",
    chunk_summaries: bool | None = None,
) -> SummaryNode:
    """Build the summary tree, calling the Planner LLM for node summaries.

    Only section- and document-level summaries feed the pipeline; per-chunk
    summaries are never used downstream, so by default no LLM request is made
    for them (one fewer request per chunk). Set ``SUMMARY_TREE_CHUNK_SUMMARIES=1``
    (or pass ``chunk_summaries=True``) to restore the old per-chunk behaviour.
    """
    if chunk_summaries is None:
        try:
            from polyglot_lc.config import get_app_config

            chunk_summaries = get_app_config().summary_tree_chunk_summaries
        except Exception:  # noqa: BLE001 - config is best-effort here
            chunk_summaries = False

    root = SummaryNode(level="doc", title=doc_title)
    sections = build_sections(chunks)

    for sec_name, sec_chunks in sections.items():
        sec_node = SummaryNode(level="section", title=sec_name)
        for c in sec_chunks:
            if chunk_summaries:
                prompt = (
                    "Summarize in 1-2 short sentences for a translator's context. "
                    "Reply with the summary only.\n\n"
                    f"Section: {sec_name}\nText:\n{c.text[:2000]}"
                )
                summary = planner.chat(
                    [{"role": "user", "content": prompt}]
                ).text.strip()
            else:
                summary = _chunk_preview(c.text)
            node = SummaryNode(
                level="chunk",
                title=c.id,
                summary=summary,
                meta={"chunk_id": c.id, "tokens": c.token_count},
            )
            sec_node.children.append(node)
        if sec_chunks:
            sec_prompt = (
                "Summarize this section in one sentence for a translator. "
                "Reply with the summary only.\n\n"
                + "\n\n".join(c.text[:400] for c in sec_chunks[:5])
            )
            resp = planner.chat([{"role": "user", "content": sec_prompt}])
            sec_node.summary = resp.text.strip()
        root.children.append(sec_node)

    doc_prompt = (
        f"Summarize the document '{doc_title}' in 2-3 sentences for translators. "
        "Reply with the summary only.\n\n"
        + "\n".join(f"- {s.title}: {s.summary}" for s in root.children)
    )
    root.summary = planner.chat([{"role": "user", "content": doc_prompt}]).text.strip()
    return root
