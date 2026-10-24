"""Consistency editor: fixes terminology drift between chunks of one document.

Per-chunk translation cannot see its neighbours, so the same source term can
come out as two different target terms. This agent applies the smallest possible
edit — the planned glossary mapping — to a chunk that drifted, and leaves the
rest of the sentence alone.

It reuses the read-source group's provider/model (``REFINER`` configuration) so
its request shares the same cacheable prefix as the Translator, the Critic and
the Refiner, and so no new environment variables are required.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from polyglot_lc.agents.base import Agent
from polyglot_lc.agents.context import source_context_messages


class ConsistencyAgent(Agent):
    """Rewrites one chunk so it honours the document glossary exactly."""

    name = "CONSISTENCY"

    def patch(
        self,
        source: str,
        translation: str,
        fixes: list[tuple[str, str]],
        src_lang: str,
        tgt_lang: str,
        *,
        strategy: str = "",
        glossary_md: str = "",
        section_note: str = "",
    ) -> str:
        """Return *translation* with *fixes* (``source_term, target_term``) applied.

        Returns the original text unchanged when the model answers with nothing,
        so a failed patch can never delete a chunk.
        """
        if not fixes:
            return translation
        wanted = "\n".join(f"- {term} must be translated as: {target}" for term, target in fixes)
        messages: list[dict[str, Any]] = [
            source_context_messages(
                src_lang, tgt_lang, strategy, None, glossary_md, source, section_note
            ),
            {
                "role": "user",
                "content": (
                    "You are a terminology consistency editor.\n"
                    # The WHOLE translation, never a slice: the model answers
                    # with what it is shown, so a truncated view became a
                    # truncated rewrite — one that still carried the wanted
                    # term and replaced a complete translation with a
                    # fragment.
                    f"Current translation:\n{translation}\n\n"
                    "The translators of the other sections used these exact terms:\n"
                    f"{wanted}\n\n"
                    "Fix ONLY the terminology above so it matches. Change nothing "
                    "else: keep the same meaning, wording, formatting and "
                    "placeholders (e.g. ⟦CODE_1⟧). Output ONLY the corrected "
                    "translation."
                ),
            },
        ]
        try:
            out = self.chat(messages, cache_prefix=True).text.strip()
        except Exception as exc:  # noqa: BLE001 - patching is best-effort
            logger.warning("Consistency patch failed ({}); keeping the draft", exc)
            return translation
        return out or translation
