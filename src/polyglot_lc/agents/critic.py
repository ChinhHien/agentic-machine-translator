"""Critic agent: scores drafts 0-10 with structured, actionable feedback."""

from __future__ import annotations

from pydantic import BaseModel, Field

from polyglot_lc.agents.base import Agent
from polyglot_lc.agents.context import source_context_messages


class Critique(BaseModel):
    """Structured critique result for one draft."""

    score: float = Field(ge=0, le=10)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    @property
    def feedback_text(self) -> str:
        """Compact feedback block fed back to the translator/refiner."""
        lines = [f"Score: {self.score:.1f}/10"]
        if self.issues:
            lines.append("Issues:\n" + "\n".join(f"- {i}" for i in self.issues))
        if self.suggestions:
            lines.append("Suggestions:\n" + "\n".join(f"- {s}" for s in self.suggestions))
        return "\n".join(lines)


class CriticAgent(Agent):
    """Evaluates draft translations; returns a Critique."""

    name = "CRITIC"

    def review(
        self,
        source: str,
        draft: str,
        src_lang: str,
        tgt_lang: str,
        glossary_md: str = "",
        strategy: str = "",
        pitfalls: list[str] | None = None,
        section_note: str = "",
    ) -> Critique:
        """Score *draft* (0-10) against *source* on fidelity, fluency, terms.

        The source chunk is delivered through the shared read-source prefix
        (cacheable); only the draft being reviewed is dynamic.
        """
        messages = [
            source_context_messages(
                src_lang,
                tgt_lang,
                strategy,
                pitfalls,
                glossary_md,
                source,
                section_note,
            )
        ]
        dynamic = (
            "You are a meticulous translation critic.\n"
            f"Translation ({tgt_lang}):\n{draft[:2500]}\n\n"
            "Score fidelity, fluency, terminology and structure from 0 to 10 "
            "against this rubric:\n"
            "- 9-10: publishable as-is; nothing a careful editor would touch.\n"
            "- 7-8: good, only minor nits; safe to ship.\n"
            "- 4-6: one real defect (mistranslation, dropped or added "
            "content, wrong term, broken structure) that must be fixed.\n"
            "- 0-3: unusable.\n"
            "Be strict about two things: the draft must keep the source's "
            "headings, lists, placeholders and Markdown table rows (report a "
            "flattened or rewritten table as an issue), and it must use the "
            "glossary exactly. Below 7, every issue must name the exact text "
            "at fault so it can be fixed without re-reading the whole chunk.\n"
            "Reply ONLY with JSON:\n"
            '{"score": 8.5, "issues": ["..."], "suggestions": ["..."]}'
        )
        messages.append({"role": "user", "content": dynamic})
        data = self.chat_json(messages, cache_prefix=True)
        try:
            return Critique(
                score=float(data.get("score", 5.0)),
                issues=[str(x) for x in data.get("issues", [])],
                suggestions=[str(x) for x in data.get("suggestions", [])],
            )
        except (TypeError, ValueError):
            # Malformed score -> neutral-pass so the pipeline is not stuck.
            return Critique(score=7.0, issues=["critic reply unparseable"])
