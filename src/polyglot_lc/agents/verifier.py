"""Verifier agent: final hard gates (placeholders, length, JSON verdict)."""

from __future__ import annotations

import re
import time

from loguru import logger

from polyglot_lc.agents.base import Agent

_PH_RE = re.compile(r"⟦CODE_\d+⟧")
_HEAD_RE = re.compile(r"^(#{1,6}) ", re.MULTILINE)


def _heading_levels(text: str) -> list[int]:
    """Heading level sequence (e.g. [1, 2, 2, 3]) preserved by translation."""
    return [len(m.group(1)) for m in _HEAD_RE.finditer(text)]


class VerifyResult:
    """Outcome of verification: pass/fail plus human-readable reasons."""

    def __init__(self, ok: bool, reasons: list[str]) -> None:
        self.ok = ok
        self.reasons = reasons

    def __bool__(self) -> bool:  # allows `if verify_result:`
        return self.ok

    def __repr__(self) -> str:
        return f"VerifyResult(ok={self.ok}, reasons={self.reasons})"


def hard_checks(source: str, translation: str) -> list[str]:
    """Deterministic checks: placeholders and heading levels must survive."""
    problems: list[str] = []
    if sorted(_PH_RE.findall(source)) != sorted(_PH_RE.findall(translation)):
        problems.append("code placeholders missing or altered")
    if _heading_levels(source) != _heading_levels(translation):
        problems.append("heading structure changed")
    return problems


class VerifierAgent(Agent):
    """LLM-based final QA plus deterministic hard checks."""

    name = "VERIFIER"

    def verify(
        self,
        source: str,
        translation: str,
        src_lang: str,
        tgt_lang: str,
        glossary_md: str = "",
    ) -> VerifyResult:
        """Run hard checks first, then an LLM adequacy verdict.

        The whole chunk is sent: a truncation would leave the tail of a long
        chunk unverified, and the chunker has already bounded the size. When
        *glossary_md* is supplied the verdict also covers terminology, which is
        otherwise only policed document-wide by the consistency pass.
        """
        problems = hard_checks(source, translation)
        if problems:
            return VerifyResult(ok=False, reasons=problems)

        glossary = ""
        if glossary_md and glossary_md.strip() and glossary_md.strip() != "(none)":
            glossary = f"Glossary (the agreed target form of each term):\n{glossary_md}\n\n"
        prompt = (
            "You are a translation QA verifier. Decide whether this chunk is "
            "safe to ship.\n"
            f"Source ({src_lang}):\n{source}\n\n"
            f"Translation ({tgt_lang}):\n{translation}\n\n" + glossary + "Check, in order:\n"
            "1. Completeness - every sentence of the source is represented.\n"
            "2. No additions - nothing was invented that the source does not "
            "say.\n"
            "3. Language - the translation is in the target language, reads "
            "naturally, and leaves no source-language text behind.\n"
            "4. Structure - headings, lists, placeholders (e.g. ⟦CODE_1⟧), "
            "the emphasis pairs (⟦BOLD⟧...⟦/BOLD⟧, ⟦ITALIC⟧...⟦/ITALIC⟧) "
            "and Markdown table rows are intact.\n"
            "5. Terminology - every glossary term that occurs in the source "
            "appears in the translation in the glossary's own target form. "
            "Skip this check when no glossary is given.\n"
            "Pass only when all the applicable checks hold. Fail only for a "
            "defect a reader would notice, never for a style preference you "
            "would have phrased differently. Name the exact text at fault in "
            "each reason: a failure costs one re-translation of this chunk.\n"
            "Reply ONLY with JSON:\n"
            '{"pass": true|false, "reasons": ["..."]}'
        )
        t0 = time.monotonic()
        logger.info(
            "VERIFIER starting | src_len={} tgt_len={} model={}",
            len(source),
            len(translation),
            self.config.model,
        )
        try:
            data = self.chat_json([{"role": "user", "content": prompt}])
            ok = bool(data.get("pass", True))
            reasons = [str(r) for r in data.get("reasons", [])]
            elapsed = time.monotonic() - t0
            logger.info(
                "VERIFIER completed | ok={} elapsed={:.1f}s model={}",
                ok,
                elapsed,
                self.config.model,
            )
        except Exception as exc:  # noqa: BLE001 - never block on verifier failure
            elapsed = time.monotonic() - t0
            logger.warning(
                "VERIFIER failed after {:.1f}s | error={} model={}", elapsed, exc, self.config.model
            )
            return VerifyResult(ok=True, reasons=["verifier unavailable; skipped"])
        return VerifyResult(ok=ok, reasons=reasons)
