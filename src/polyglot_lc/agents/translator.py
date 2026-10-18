"""Translator agent: chunk translation with strategy, glossary and TM context."""

from __future__ import annotations

from loguru import logger

from polyglot_lc.agents.base import Agent
from polyglot_lc.agents.context import (
    extract_source,
    source_context_messages,
    stable_prefix_fingerprint,
)
from polyglot_lc.llm.resilience import TruncatedError
from polyglot_lc.memory.tm import TranslationMemory

#: Output-token cap for the one retry that recovers a truncated reply. High
#: enough that a 700-token chunk (the default ``chunk_max_tokens``) fits with
#: room to spare; most providers accept up to 4096 on a free tier.
_RAISED_OUTPUT_CAP = 4096


class TranslatorAgent(Agent):
    """Translates one chunk under the planner's strategy and constraints."""

    name = "TRANSLATOR"

    def __init__(self, config=None, tm: TranslationMemory | None = None) -> None:
        super().__init__(config)
        # NB: not `tm or ...` — TranslationMemory defines __len__, so an empty
        # TM is falsy and would silently get replaced by a fresh instance.
        self.tm = tm if tm is not None else TranslationMemory()

    def build_prompt(
        self,
        chunk_text: str,
        src_lang: str,
        tgt_lang: str,
        strategy: str,
        few_shot: list[str] | None = None,
        pitfalls: list[str] | None = None,
        glossary_md: str = "",
        extra_critique: str = "",
        section_note: str = "",
    ) -> list[dict[str, str]]:
        """Assemble the translation prompt (exposed for SkillOpt testing).

        ``messages[0]`` is the shared read-source prefix (identical to the
        Critic's and Refiner's for the same chunk) so the provider can cache
        it; the dynamic instruction and examples follow it.
        """
        messages = [
            source_context_messages(
                src_lang,
                tgt_lang,
                strategy,
                pitfalls,
                glossary_md,
                chunk_text,
                section_note,
            )
        ]
        user_parts: list[str] = []
        if few_shot:
            user_parts.append("Examples:\n" + "\n".join(few_shot))
        tm_hits = self.tm.lookup(chunk_text, src_lang, tgt_lang)
        if tm_hits:
            user_parts.append(
                "Translation memory matches:\n" + "\n".join(f"- {s} => {t}" for s, t in tm_hits)
            )
        if extra_critique:
            # Carries the verifier's rejection reasons on the single bounded
            # retranslate, so the wording stays neutral about who rejected it.
            user_parts.append(
                "An earlier draft was rejected for these reasons; the new "
                "translation must fix every one of them:\n" + extra_critique
            )
        user_parts.append(
            "Rules:\n"
            "- Translate the source text above faithfully; keep formatting, "
            "headings and placeholders (e.g. ⟦CODE_1⟧) exactly as-is.\n"
            "- Keep the emphasis pairs ⟦BOLD⟧...⟦/BOLD⟧ and "
            "⟦ITALIC⟧...⟦/ITALIC⟧ around the words they emphasise; translate the "
            "text inside them, never translate or rename the tokens, and never "
            "add a pair of your own.\n"
            "- Preserve the document structure exactly: keep every paragraph "
            "separated by a blank line, keep headings (#, ##...) and bullet "
            "lists as-is, and never merge two paragraphs into one.\n"
            "- Keep Markdown tables as tables: translate the text inside a "
            "cell, but keep the rows, the `|` separators and the `| --- |` "
            "header line, and never rewrite a table as prose or merge its "
            "cells into a sentence.\n"
            "- Use the glossary consistently.\n"
            "- Output ONLY the translation, no commentary."
        )
        messages.append({"role": "user", "content": "\n\n".join(user_parts)})
        return messages

    def translate(self, prompt: list[dict[str, str]], chunk_id: str = "") -> str:
        """Run translation and store the result in TM (source->target)."""
        # Debug-only fingerprint of the shared cacheable prefix (never logs text).
        logger.debug(
            "chunk={} cache_fingerprint={} model={}",
            chunk_id or "-",
            stable_prefix_fingerprint(prompt, self.config.model),
            self.config.model,
        )
        # Mark the shared read-source prefix as a cache breakpoint.
        #
        # A provider that cuts the reply at its own output-token limit raises
        # TruncatedError: the model was mid-sentence, so the tail of the chunk
        # was never translated. Retrying with a raised cap is the only fix that
        # actually recovers those words — a text heuristic cannot tell a cut
        # sentence from a legitimately short one, so the provider's own
        # ``finish_reason`` is the signal we act on.
        try:
            resp = self.chat(prompt, cache_prefix=True)
        except TruncatedError:
            logger.warning(
                "chunk={} reply was cut at the provider's output-token limit; "
                "re-asking with a raised cap",
                chunk_id or "-",
            )
            resp = self.chat(
                prompt, cache_prefix=True, max_tokens=_RAISED_OUTPUT_CAP
            )
        out = resp.text.strip()
        # Source text is recovered from the shared context markers.
        source = extract_source(prompt)
        if source:
            self.tm.store(source, out, "auto", "auto")
        return out
