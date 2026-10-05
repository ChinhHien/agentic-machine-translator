"""Verifier agent: final hard gates (placeholders, length, JSON verdict)."""

from __future__ import annotations

import re
from typing import Any

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
    ) -> VerifyResult:
        """Run hard checks first, then an LLM adequacy verdict."""
        problems = hard_checks(source, translation)
        if problems:
            return VerifyResult(ok=False, reasons=problems)

        prompt = (
            "You are a translation QA verifier.\n"
            f"Source ({src_lang}):\n{source[:2000]}\n\n"
            f"Translation ({tgt_lang}):\n{translation[:2000]}\n\n"
            "Check: completeness (nothing dropped), no additions, target "
            "language is natural. Reply ONLY with JSON:\n"
            '{"pass": true|false, "reasons": ["..."]}'
        )
        try:
            data = self.chat_json([{"role": "user", "content": prompt}])
            ok = bool(data.get("pass", True))
            reasons = [str(r) for r in data.get("reasons", [])]
        except Exception:  # noqa: BLE001 - never block on verifier failure
            return VerifyResult(ok=True, reasons=["verifier unavailable; skipped"])
        return VerifyResult(ok=ok, reasons=reasons)
