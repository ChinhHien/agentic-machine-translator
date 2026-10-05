"""Refiner agent: rewrites a draft using the critic's feedback."""

from __future__ import annotations

from polyglot_lc.agents.base import Agent
from polyglot_lc.agents.context import source_context_messages


class RefinerAgent(Agent):
    """Produces an improved draft from (draft, critique)."""

    name = "REFINER"

    def refine(
        self,
        source: str,
        draft: str,
        critique_text: str,
        src_lang: str,
        tgt_lang: str,
        strategy: str = "",
        pitfalls: list[str] | None = None,
        glossary_md: str = "",
    ) -> str:
        """Return the refined translation for *draft*.

        Shares the cached read-source prefix with the Translator/Critic; the
        draft and critique are the only dynamic parts.
        """
        messages = [
            source_context_messages(
                src_lang, tgt_lang, strategy, pitfalls, glossary_md, source
            ),
            {
                "role": "user",
                "content": (
                    f"You are a translation refiner for {src_lang}->{tgt_lang}.\n"
                    f"Current draft:\n{draft[:2000]}\n\n"
                    f"Critic feedback:\n{critique_text[:1500]}\n\n"
                    "Rewrite the draft fixing every issue, referring to the "
                    "source text above. Keep formatting and placeholders "
                    "intact. Output ONLY the improved translation."
                ),
            },
        ]
        resp = self.chat(messages, cache_prefix=True)
        return resp.text.strip()
