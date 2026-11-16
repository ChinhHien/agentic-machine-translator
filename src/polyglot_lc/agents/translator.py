"""Translator agent: chunk translation with strategy, glossary and TM context."""

from __future__ import annotations

import os

from loguru import logger

from polyglot_lc.agents.base import Agent
from polyglot_lc.agents.context import (
    extract_source,
    source_context_messages,
    stable_prefix_fingerprint,
)
from polyglot_lc.llm.resilience import TruncatedError
from polyglot_lc.memory.tm import TranslationMemory

#: Output budget for the recovery ask after a reply is cut at the provider's
#: limit. A reasoning model charges its thinking to the same completion budget
#: as the answer: on a real run the Translator's successful calls used a median
#: of ~9000 completion tokens (thinking ~6100 + answer ~2600) and a maximum of
#: ~17000, because the first ask sets no cap and the endpoint's own default
#: allows them. The recovery used to ask for 4096 — under half the median — so
#: it cut again and reported a failure the chunk had not earned. 16384 stays
#: below the largest budget that endpoint was observed to serve, so asking for
#: it explicitly does not exceed what the gateway already allows by default.
DEFAULT_TRANSLATOR_OUTPUT_CAP = 16384

#: Hard ceiling for the recovery ladder. Doubling is unbounded otherwise, and
#: gateways reject a ``max_tokens`` above their own limit outright.
TRANSLATOR_MAX_TOKENS_CEILING = 32768


def _translator_output_cap() -> int:
    """The configured recovery budget, for an endpoint with a different ceiling."""
    raw = os.getenv("TRANSLATOR_MAX_TOKENS", "").strip()
    if not raw:
        return DEFAULT_TRANSLATOR_OUTPUT_CAP
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "TRANSLATOR_MAX_TOKENS={!r} is not a whole number; using {}",
            raw,
            DEFAULT_TRANSLATOR_OUTPUT_CAP,
        )
        return DEFAULT_TRANSLATOR_OUTPUT_CAP
    return value if value > 0 else DEFAULT_TRANSLATOR_OUTPUT_CAP


def _recovery_cap(cap: int, cut_at: int | None) -> int:
    """The budget a recovery ask uses, given the budget that was just cut.

    At least twice what the provider consumed before stopping: asking for less
    than that cannot produce a longer reply, so a tuned cap below the observed
    cut yields to the evidence. Never above the ceiling.
    """
    return min(max(cap, (cut_at or 0) * 2), TRANSLATOR_MAX_TOKENS_CEILING)


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
        skill_instruction: str = "",
        profile: str = "body",
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
        if profile == "references":
            user_parts.append(
                "This chunk is a reference list. Translate citation prose, but "
                "keep article/book titles, journal names, author names, "
                "publisher names and chemical names as written."
            )
        if skill_instruction:
            # Constant for the whole run, first in the dynamic part: the skill
            # is read once per run, so this line does not churn between chunks.
            user_parts.append("Style learned over previous runs: " + skill_instruction)
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
        # The first ask sets no cap: the endpoint's own default is what let a
        # reasoning model think ~6000 tokens and still answer ~2600. A budget is
        # only imposed on a recovery, where the provider has already said the
        # one it used was not enough — and a text heuristic cannot tell a cut
        # sentence from a legitimately short one, so ``finish_reason`` is the
        # signal acted on.
        try:
            resp = self.chat(prompt, cache_prefix=True)
        except TruncatedError as err:
            cap = _recovery_cap(_translator_output_cap(), err.completion_tokens)
            logger.warning(
                "chunk={} reply was cut at {} completion tokens (thinking={} answer={}); "
                "re-asking with a budget of {}",
                chunk_id or "-",
                err.completion_tokens,
                err.reasoning_tokens,
                err.answer_tokens,
                cap,
            )
            try:
                resp = self.chat(prompt, cache_prefix=True, max_tokens=cap)
            except TruncatedError as again:
                raised = _recovery_cap(cap, again.completion_tokens)
                if raised <= cap:
                    # At the ceiling: another ask is the same question. The
                    # chunk layer salvages by splitting instead of spinning.
                    raise
                logger.warning(
                    "chunk={} reply was cut again at {} (thinking={} answer={}); "
                    "re-asking with a budget of {}",
                    chunk_id or "-",
                    again.completion_tokens,
                    again.reasoning_tokens,
                    again.answer_tokens,
                    raised,
                )
                resp = self.chat(prompt, cache_prefix=True, max_tokens=raised)
        out = resp.text.strip()
        # Source text is recovered from the shared context markers.
        source = extract_source(prompt)
        if source:
            self.tm.store(source, out, "auto", "auto")
        return out
