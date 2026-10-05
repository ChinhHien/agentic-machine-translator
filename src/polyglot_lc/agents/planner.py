"""Planner agent: global strategy, summary tree, trace drafts."""

from __future__ import annotations

from typing import Any

from polyglot_lc.agents.base import Agent
from polyglot_lc.segment.glossary import glossary_to_md


class PlannerAgent(Agent):
    """Chooses translation strategy from doc context + reasoning bank traces."""

    name = "PLANNER"

    def plan(
        self,
        doc_summary: str,
        src_lang: str,
        tgt_lang: str,
        domain: str,
        glossary_md: str = "",
        rb_traces: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Return {strategy, few_shot, pitfalls, notes} for the whole document."""
        trace_lines = [
            f"- {t.strategy} | pitfalls: {'; '.join(t.pitfalls) or '-'}"
            for t in (rb_traces or [])
        ]
        prompt = (
            "You are a senior translation planner.\n"
            f"Language pair: {src_lang} -> {tgt_lang}. Domain: {domain}.\n"
            f"Document summary:\n{doc_summary[:1500]}\n\n"
            f"Glossary candidates:\n{glossary_md[:800]}\n\n"
            "Similar past traces from the reasoning bank:\n"
            + ("\n".join(trace_lines) if trace_lines else "(none)")
            + "\n\nDecide ONE strategy for this document. Reply ONLY with JSON:\n"
            '{"strategy": "...", "few_shot": ["example src -> example tgt", ...],'
            ' "pitfalls": ["...", ...], "notes": "..."}'
        )
        data = self.chat_json([{"role": "user", "content": prompt}])
        return {
            "strategy": str(data.get("strategy", "literal, terminology-consistent")),
            "few_shot": [str(x) for x in data.get("few_shot", [])][:6],
            "pitfalls": [str(x) for x in data.get("pitfalls", [])][:6],
            "notes": str(data.get("notes", "")),
        }

    def summarize(self, text: str, instruction: str) -> str:
        """Generic summarization used by the summary tree."""
        prompt = f"{instruction}\n\n{text[:4000]}"
        return self.chat([{"role": "user", "content": prompt}]).text.strip()
