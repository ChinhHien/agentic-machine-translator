"""Deterministic complexity + draft-quality signals (zero LLM calls).

The adaptive quality pipeline needs two numbers before it can route a chunk:

* how hard the chunk looks *before* translation (:func:`score_chunk`), and
* how good the draft looks *after* translation (:func:`draft_signals`).

Both are computed locally — from document structure and from the same
deterministic hard checks the verifier enforces — so routing costs no extra
tokens and behaves identically in mock mode, in tests and in production.
Nothing here calls a model, which keeps the fast path free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from polyglot_lc.segment.glossary import extract_glossary

#: Above this token count a chunk contributes its maximum length complexity.
LONG_CHUNK_TOKENS = 400

#: Character length below which the source/target length ratio is meaningless
#: (very short chunks are legitimately translated to a much shorter or longer
#: string), so the ratio check is skipped entirely.
MIN_LENGTH_CHECK_CHARS = 160

#: Length ratios outside this band mean the draft is truncated or runaway.
#: A *band* rather than a graded penalty: legitimate compression differs a lot
#: by language pair (en->ja is far shorter than en->vi), so only the extremes are
#: treated as evidence of a broken draft.
MIN_LENGTH_RATIO = 0.25
MAX_LENGTH_RATIO = 4.0

#: Content words shorter than this are ignored by the leakage signal.
MIN_CONTENT_WORD = 4

_PLACEHOLDER_RE = re.compile(r"⟦CODE_\d+⟧")
_HEADING_RE = re.compile(r"^(#{1,6}) ", re.MULTILINE)
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)]) ", re.MULTILINE)
_WORD_RE = re.compile(r"[A-Za-zÀ-ỹ][\w'-]*")

#: Weights sum to 1.0 so :func:`score_chunk` is already normalised.
_W_TOKENS = 0.35
_W_PLACEHOLDERS = 0.20
_W_HEADING = 0.10
_W_DEEP_HEADING = 0.05
_W_LIST = 0.10
_W_LONG_SENTENCE = 0.10
_W_TERMS = 0.12
_W_PLANNER_NOTE = 0.05


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class Complexity:
    """How hard one chunk looks, in ``[0, 1]``, with the reasons why."""

    value: float
    reasons: list[str] = field(default_factory=list)

    def __float__(self) -> float:  # convenience for comparisons/logging
        return self.value


def score_chunk(
    text: str,
    *,
    token_count: int = 0,
    section_note: str = "",
    doc_complexity: float = 0.0,
) -> Complexity:
    """Score one chunk's translation difficulty in ``[0, 1]``.

    Signals (document structure only, no model calls): token volume, shielded
    code placeholders, heading depth, list markup, very long sentences, and the
    density of terminology candidates. A Planner-flagged section note and the
    document-level complexity from the unified planner are blended in.
    """
    reasons: list[str] = []
    value = 0.0

    if token_count > 0:
        ratio = _clamp(token_count / LONG_CHUNK_TOKENS)
        value += _W_TOKENS * ratio
        if ratio >= 0.5:
            reasons.append(f"long chunk ({token_count} tokens)")

    placeholders = len(_PLACEHOLDER_RE.findall(text))
    if placeholders:
        value += _W_PLACEHOLDERS * _clamp(placeholders / 4.0)
        reasons.append(f"{placeholders} code placeholder(s) to preserve")

    headings = _HEADING_RE.findall(text)
    if headings:
        value += _W_HEADING
        reasons.append("headings must keep their level")
        if max(len(h) for h in headings) >= 3:
            value += _W_DEEP_HEADING
            reasons.append("deep heading nesting")

    if _LIST_RE.search(text):
        value += _W_LIST
        reasons.append("list markup")

    words = _WORD_RE.findall(text)
    sentences = [s for s in re.split(r"[.!?…]+", text) if s.strip()]
    if sentences:
        mean_words = len(words) / len(sentences)
        if mean_words > 30:
            value += _W_LONG_SENTENCE
            reasons.append(f"long sentences (mean {mean_words:.0f} words)")

    terms = extract_glossary(text, min_count=2, max_terms=40)
    if terms:
        value += _W_TERMS * _clamp(len(terms) / 12.0)
        if len(terms) >= 6:
            reasons.append(f"{len(terms)} terminology candidates")

    if section_note:
        value += _W_PLANNER_NOTE
        reasons.append("planner flagged this section")

    chunk_value = _clamp(value)
    # A hard document weighs on every chunk, but the chunk's own structure wins.
    blended = 0.7 * chunk_value + 0.3 * _clamp(doc_complexity)
    return Complexity(value=round(blended, 4), reasons=reasons)


@dataclass(frozen=True)
class DraftSignals:
    """Cheap opinion of a draft, in the same 0-10 scale the Critic uses."""

    score: float
    issues: list[str] = field(default_factory=list)
    length_ratio: float = 1.0
    leakage: float = 0.0

    @property
    def clean(self) -> bool:
        """True when no structural problem was detected."""
        return not self.issues


def _leakage(source: str, draft: str) -> float:
    """Share of the source's content words left untranslated in *draft*.

    Deliberately blind to short words, so names, acronyms and loan words do not
    inflate the signal; a fully untranslated chunk scores ~1.0.
    """
    source_words = {w.lower() for w in _WORD_RE.findall(source) if len(w) >= MIN_CONTENT_WORD}
    if not source_words:
        return 0.0
    draft_lower = draft.lower()
    hits = sum(1 for w in source_words if w in draft_lower)
    return hits / len(source_words)


def draft_signals(
    source: str,
    draft: str,
    *,
    hard_issues: list[str] | None = None,
) -> DraftSignals:
    """Score a draft 0-10 without asking a model.

    Starts from a perfect score and subtracts for deterministic structural
    problems (via *hard_issues*, normally
    :func:`polyglot_lc.agents.verifier.hard_checks`), for a truncated or runaway
    draft, and for untranslated source text. The result gates the fast path
    only; the Critic still runs whenever the signals are anything but clean.
    """
    if hard_issues is None:
        from polyglot_lc.agents.verifier import hard_checks

        hard_issues = hard_checks(source, draft)

    issues = list(hard_issues)
    score = 10.0

    stripped = draft.strip()
    if not stripped:
        return DraftSignals(score=0.0, issues=["empty draft"])
    if stripped == source.strip() and len(source.strip()) > 20:
        return DraftSignals(score=0.0, issues=["draft identical to source"])

    score -= 3.0 * len(hard_issues)

    source_len = len(source.strip())
    ratio = len(stripped) / max(1, source_len)
    if source_len >= MIN_LENGTH_CHECK_CHARS:
        if ratio < MIN_LENGTH_RATIO:
            score -= 2.5
            issues.append(f"draft is only {ratio:.0%} of the source length")
        elif ratio > MAX_LENGTH_RATIO:
            score -= 2.5
            issues.append(f"draft is {ratio:.1f}x the source length")

    leakage = _leakage(source, draft)
    if leakage > 0.5:
        penalty = min(3.0, (leakage - 0.5) * 6.0)
        score -= penalty
        if leakage >= 0.9:
            issues.append("source text appears untranslated")

    return DraftSignals(
        score=round(_clamp(score, 0.0, 10.0), 2),
        issues=issues,
        length_ratio=round(ratio, 3),
        leakage=round(leakage, 3),
    )


def heuristic_document_complexity(text: str) -> float:
    """Document-level complexity in ``[0, 1]`` when the Planner gave none.

    Length and terminology density are the two signals a translator actually
    feels at document scale; both are cheap to compute on the plain text.
    """
    if not text:
        return 0.0
    length = _clamp(len(text) / 40_000)
    terms = _clamp(len(extract_glossary(text, min_count=2, max_terms=40)) / 30.0)
    placeholders = _clamp(len(_PLACEHOLDER_RE.findall(text)) / 20.0)
    return round(_clamp(0.5 * length + 0.3 * terms + 0.2 * placeholders), 4)
