"""Cross-chunk consistency: one source term, one target term.

Chunks are translated independently, so a term planned as *A* can come out as
*A* in one chunk and *B* in another. This pass looks for exactly that
divergence across the assembled document and rewrites **only the offending
chunks** — unlike a failed verification, which would re-translate a whole chunk
from scratch.

Two deliberate limits keep the pass cheap and safe:

* a *uniform miss* is not an inconsistency. If no chunk used the planned term,
  the translator disagreed with the glossary everywhere; that is reported, not
  patched, because there is nothing to make the chunks agree *on*.
* patches are bounded (``CONSISTENCY_MAX_PATCHES``, and at least a share of the
  chunks that actually drifted — see :data:`_PATCH_SHARE`) and are kept only
  when they actually introduce the expected term *and* pass the same hard
  checks as a translation (dropped code markers repaired, heading structure
  unchanged), so a bad edit can never delete a chunk — or its code.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from polyglot_lc.agents.verifier import hard_checks
from polyglot_lc.pipeline.placeholders import repair_code_placeholders

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.strip()).casefold()


def contains_term(haystack: str, needle: str) -> bool:
    """Case- and whitespace-insensitive containment test for a glossary term."""
    term = _normalize(needle)
    if not term:
        return False
    return term in _normalize(haystack)


#: How much of the text a patch replaces must still be there afterwards. A
#: terminology edit changes a word or two, so its reply measures the same
#: length as the text it replaces; one that comes back much shorter is a
#: truncated answer — a provider cap, or the model repeating back the
#: fragment of the text it was shown — not an edit. The term check alone
#: cannot catch it, because a fragment can still carry the term.
#:
#: Deliberately **not** ``assembler._MIN_COVERAGE_RATIO``, although both are
#: 0.85: that one compares a *translation* with its *source chunk* and answers
#: "did the reply cover the whole chunk?". This one compares a *patch* with
#: the *translation it replaces* and answers "is this still the same text?".
#: A chunk may legitimately translate to 0.9 of its source while a patch of
#: that translation must stay near 1.0 of it; sharing one constant would tie
#: two unrelated judgements together.
_MIN_PATCH_LENGTH_SHARE = 0.85

#: How many patches in a row may be refused before the pass stops asking. Two
#: refusals in a row mean the editor is no longer following the instruction (it
#: drops the term, loses the translation, or breaks the structure), so the
#: remaining requests would spend budget on the same answer. The chunks left
#: unpatched are still counted as skipped — the pass stopped, the drift is
#: still there.
_MAX_CONSECUTIVE_REFUSALS = 2

#: The share of the drifting chunks the pass may rewrite, above the fixed floor
#: in ``CONSISTENCY_MAX_PATCHES``. One number cannot fit both a 3-chunk memo and
#: a 300-chunk manual: at 5 patches the long document leaves most of its drift
#: alone, while a memo rarely drifts in more than a couple of chunks. Each patch
#: is one request over one chunk, so the cost stays proportional to the drift.
_PATCH_SHARE = 0.3


#: A structural unit opens a paragraph and is short: "Preconditions:", "Step 3.",
#: "Figure 2.1 The architecture". A longer first line is a sentence that happens
#: to start the paragraph, not a label, and clipping it to the cap would make two
#: unrelated sentences look like the same repeated phrase.
UNIT_MAX_CHARS = 60

#: Characters of a translated paragraph used as a unit's observed form.
UNIT_FORM_CHARS = 80

#: Words that open a label even when the source line has no trailing colon.
_MARKER_WORDS = frozenset(
    {
        "actor",
        "bài",
        "bảng",
        "bước",
        "caution",
        "example",
        "exercise",
        "fig",
        "figure",
        "ghi",
        "hình",
        "input",
        "lưu",
        "note",
        "output",
        "postcondition",
        "precondition",
        "solution",
        "step",
        "table",
        "trigger",
        "ví",
        "warning",
        "điều",
        "đầu",
    }
)
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_PUNCT_RE = re.compile(r"[^\w\s]+")


@dataclass(frozen=True)
class Inconsistency:
    """One planned term — or structural unit — that some chunks honour and others do not."""

    term: str
    expected: str
    missing_chunks: tuple[str, ...]
    compliant_chunks: tuple[str, ...]
    #: ``glossary`` for a planned term, ``label`` for a unit the document itself
    #: voted on (a label or a repeated paragraph opening).
    kind: str = "glossary"


@dataclass
class ConsistencyReport:
    """What the pass looked at and what it changed."""

    checked_terms: int = 0
    #: True when the pass had no planned term to compare (a fallback plan, no
    #: CSV): "0 inconsistencies" over zero terms and over 175 terms are not the
    #: same result.
    no_pairs: bool = False
    #: Structural units compared, and how many of them drifted: the part of the
    #: pass that runs without any glossary at all.
    labels_checked: int = 0
    drifted_labels: int = 0
    inconsistencies: int = 0
    #: Patch requests this pass was allowed to spend (``min(budget, drifting)``).
    #: Reported before the run so the pass's cost is not a surprise: each patch
    #: sends one chunk's whole translation plus the glossary and gets it back.
    patch_budget: int = 0
    patched_chunks: int = 0
    skipped_chunks: int = 0
    #: True when the pass stopped early because consecutive patches were refused.
    stopped_early: bool = False
    unresolved: list[str] = field(default_factory=list)

    def as_run_fields(self) -> dict[str, Any]:
        """The pass as the counters a run reports, including what it could *not* fix.

        Skipped and rejected chunks are the difference between "the document is
        consistent" and "the pass ran out of budget or refused a rewrite":
        without them a run that left most of its drift in place looked like one
        that found nothing to fix.

        One mapping, and the names are the ``run_end``/``PipelineResult`` ones,
        so the coordinator can copy it wholesale instead of restating each
        field — the two used to be maintained by hand, three times over.
        """
        return {
            "consistency_inconsistencies": self.inconsistencies,
            "consistency_patches": self.patched_chunks,
            "consistency_skipped": self.skipped_chunks,
            "consistency_rejected": len(self.unresolved),
            "consistency_labels": self.labels_checked,
            "consistency_label_drift": self.drifted_labels,
            "consistency_checked_terms": self.checked_terms,
            "consistency_no_pairs": self.no_pairs,
            "consistency_patch_budget": self.patch_budget,
            "consistency_stopped_early": self.stopped_early,
        }


def find_inconsistencies(
    chunks: list[Any],
    translations: dict[str, str],
    pairs: dict[str, str],
    *,
    min_compliant: int = 1,
) -> list[Inconsistency]:
    """Find planned terms whose use diverges between chunks.

    A term is only reported when at least *min_compliant* chunk used the planned
    target form (proof the term is achievable) while other chunks that carry the
    same source term did not.
    """
    if not pairs:
        return []
    # Normalise every chunk and translation once, not once per (term, chunk):
    # the per-call version spent 8.7 s on 200 terms over 400 chunks, and the
    # term budget now grows with the document.
    normalized_source = {c.id: _normalize(c.text) for c in chunks}
    normalized_translation = {cid: _normalize(text) for cid, text in translations.items()}
    found: list[Inconsistency] = []
    for term, expected in pairs.items():
        term, expected = str(term).strip(), str(expected).strip()
        if not term or not expected:
            continue
        needle, wanted = _normalize(term), _normalize(expected)
        if not needle:
            continue
        carrying = [c.id for c in chunks if needle in normalized_source[c.id]]
        if not carrying:
            continue
        compliant = tuple(cid for cid in carrying if wanted in normalized_translation.get(cid, ""))
        if len(compliant) < min_compliant:
            continue
        missing = tuple(cid for cid in carrying if cid not in compliant)
        if missing:
            found.append(
                Inconsistency(
                    term=term,
                    expected=expected,
                    missing_chunks=missing,
                    compliant_chunks=compliant,
                )
            )

    return found


def _paragraphs(text: str) -> list[str]:
    return [p for p in _PARAGRAPH_RE.split(text.strip()) if p.strip()]


def _opening(paragraph: str) -> str | None:
    """The paragraph's first line, when it is a complete short unit.

    A line longer than :data:`UNIT_MAX_CHARS` is a sentence that happens to open
    the paragraph; clipping it would make two unrelated sentences look like one
    repeated phrase, so such a paragraph yields no unit at all.
    """
    line = paragraph.strip().splitlines()[0].strip()
    if not line:
        return None
    # A label ends at its colon: "Preconditions: user is logged in" and
    # "Preconditions: admin rights" are the same unit with different tails.
    colon = line.find(":", 0, UNIT_MAX_CHARS)
    if colon != -1:
        return line[: colon + 1]
    if len(line) > UNIT_MAX_CHARS:
        return None
    return line


def _label_like(unit: str) -> bool:
    if unit.endswith(":"):
        return True
    first = _PUNCT_RE.sub("", unit.split()[0]).casefold()
    return first in _MARKER_WORDS


def _observed(translated_paragraph: str, *, up_to_colon: bool) -> str:
    """The translated form of a unit: up to its colon, else the opening line."""
    text = translated_paragraph.strip()
    if up_to_colon:
        colon = text.find(":", 0, UNIT_FORM_CHARS)
        if colon != -1:
            return text[: colon + 1]
    return text.splitlines()[0].strip()[:UNIT_FORM_CHARS]


def _unit_key(text: str) -> str:
    return _normalize(text).strip(" .:;,–—-")


def find_structural_drift(
    chunks: list[Any],
    translations: dict[str, str],
) -> tuple[list[Inconsistency], int]:
    """Find labels and repeated openings that different chunks rendered differently.

    Planned terms come with a target form; a label such as ``Preconditions:`` or a
    caption repeated across chunks does not. The document decides instead: the
    form most chunks used becomes the expected one, and the chunks that chose
    another are reported.

    Three rules keep that honest:

    * the translation is located by position — paragraph *i* of a translation is
      paragraph *i* of its source. A chunk whose two sides disagree on the number
      of paragraphs is skipped, never guessed at;
    * a unit must occur in at least two chunks to be compared at all, so a label
      used once is somebody else's problem;
    * the majority needs at least two votes. One chunk one way and one chunk the
      other way is a tie, not an inconsistency, so nothing is patched on a guess.

    Returns the issues and how many units were compared.
    """
    occurrences: dict[str, list[tuple[str, str]]] = {}
    spellings: dict[str, str] = {}
    for chunk in chunks:
        translated = translations.get(chunk.id, "")
        if not translated.strip():
            continue
        source_paragraphs = _paragraphs(chunk.text)
        translated_paragraphs = _paragraphs(translated)
        if len(source_paragraphs) != len(translated_paragraphs):
            continue
        for source, target in zip(source_paragraphs, translated_paragraphs):
            unit = _opening(source)
            if unit is None:
                continue
            key = _unit_key(unit)
            if not key:
                continue
            spellings.setdefault(key, unit)
            occurrences.setdefault(key, []).append(
                (chunk.id, _observed(target, up_to_colon=_label_like(unit)))
            )

    issues: list[Inconsistency] = []
    checked = 0
    for key, seen in occurrences.items():
        carriers = {chunk_id for chunk_id, _ in seen}
        if len(carriers) < 2:
            continue
        checked += 1
        forms: dict[str, list[str]] = {}
        for chunk_id, form in seen:
            normalized = _unit_key(form)
            if normalized:
                forms.setdefault(normalized, []).append(chunk_id)
        if len(forms) < 2:
            continue
        ranked = sorted(forms.items(), key=lambda item: (-len(item[1]), item[0]))
        majority_key, majority_chunks = ranked[0]
        if len(majority_chunks) < 2:
            continue
        expected = next(form for _, form in seen if _unit_key(form) == majority_key)
        expected = expected.strip(" .:;,")
        if not expected:
            continue
        missing: list[str] = []
        for chunk_id, form in seen:
            if _unit_key(form) != majority_key and chunk_id not in missing:
                missing.append(chunk_id)
        if not missing:
            continue
        issues.append(
            Inconsistency(
                term=spellings[key],
                expected=expected,
                missing_chunks=tuple(missing),
                compliant_chunks=tuple(dict.fromkeys(majority_chunks)),
                kind="label",
            )
        )
    return issues, checked


class ConsistencyChecker:
    """Applies the planned glossary to the chunks that drifted from it."""

    def __init__(
        self,
        agent: Any,
        *,
        max_patches: int = 5,
        enabled: bool = True,
        ui: Any = None,
        sink: Any = None,
    ) -> None:
        self.agent = agent
        self.max_patches = max(0, max_patches)
        self.enabled = enabled
        self.ui = ui
        self.sink = sink

    def _step(self, chunk_id: str) -> Any:
        if self.ui is not None:
            return self.ui.step("consistency", chunk_id)
        from contextlib import nullcontext

        return nullcontext()

    def run(
        self,
        chunks: list[Any],
        translations: dict[str, str],
        pairs: dict[str, str],
        *,
        context: dict[str, str] | None = None,
    ) -> tuple[dict[str, str], ConsistencyReport]:
        """Return (possibly patched) translations plus a report."""
        report = ConsistencyReport(checked_terms=len(pairs), no_pairs=not pairs)
        if not self.enabled or not chunks:
            return translations, report

        # Glossary terms have a planned target form; labels and repeated
        # paragraph openings do not, so the document votes on those instead.
        structural, report.labels_checked = find_structural_drift(chunks, translations)
        report.drifted_labels = len(structural)
        issues = find_inconsistencies(chunks, translations, pairs) + structural
        report.inconsistencies = len(issues)
        if not issues:
            return translations, report

        # One request per drifting chunk, carrying every term it must fix.
        fixes: dict[str, list[tuple[str, str]]] = {}
        for issue in issues:
            for chunk_id in issue.missing_chunks:
                fixes.setdefault(chunk_id, []).append((issue.term, issue.expected))
        # Worst first: a patch costs one request whatever it fixes, so with a
        # bounded budget the chunk carrying four drifting terms is worth the
        # same request as one carrying a single term. Document order only breaks
        # ties, which keeps a run reproducible.
        position = {c.id: i for i, c in enumerate(chunks)}
        ordered = sorted(
            (c for c in chunks if c.id in fixes),
            key=lambda c: (-len(fixes[c.id]), position[c.id]),
        )
        budget = (
            0
            if self.max_patches == 0
            else max(self.max_patches, math.ceil(len(ordered) * _PATCH_SHARE))
        )
        report.patch_budget = min(budget, len(fixes))
        logger.info(
            "Consistency: {} drifting term(s) across {} chunk(s); patching {}",
            len(issues),
            len(fixes),
            report.patch_budget,
        )
        rendered = dict(translations)
        context = context or {}
        consecutive_refusals = 0

        def refuse(chunk_id: str, reason: str) -> None:
            """Count one refused patch; two in a row end the pass's spending."""
            nonlocal consecutive_refusals
            report.unresolved.append(chunk_id)
            consecutive_refusals += 1
            if self.sink is not None:
                self.sink.info("consistency", chunk_id, f"patch rejected: {reason}")

        for index, chunk in enumerate(ordered):
            wanted = fixes[chunk.id]
            if index >= budget or consecutive_refusals >= _MAX_CONSECUTIVE_REFUSALS:
                report.skipped_chunks += 1
                if consecutive_refusals >= _MAX_CONSECUTIVE_REFUSALS:
                    report.stopped_early = True
                if self.sink is not None:
                    reason = (
                        "patch budget exhausted"
                        if index >= budget
                        else "pass stopped after consecutive refusals"
                    )
                    self.sink.info("consistency", chunk.id, f"{reason} for {chunk.id}")
                continue
            with self._step(chunk.id):
                patched = self.agent.patch(
                    source=chunk.text,
                    translation=rendered.get(chunk.id, ""),
                    fixes=wanted,
                    src_lang=context.get("src_lang", ""),
                    tgt_lang=context.get("tgt_lang", ""),
                    strategy=context.get("strategy", ""),
                    glossary_md=context.get("glossary_md", ""),
                    section_note=context.get("section_note", ""),
                )
            original = rendered.get(chunk.id, "")
            if original and len(patched) < _MIN_PATCH_LENGTH_SHARE * len(original):
                refuse(chunk.id, "rewrite lost most of the translation")
                continue
            # A patch is a rewrite, and the model rewriting a chunk can drop the
            # code markers it was told to keep. Restoration at assembly is a
            # plain ``str.replace``, so an unmentioned marker means the block is
            # deleted without trace — repaired here, exactly as the translator's
            # reply is repaired, and then gated like any other translation.
            patched, repaired = repair_code_placeholders(chunk.text, patched)
            problems = hard_checks(chunk.text, patched)
            if problems:
                refuse(chunk.id, "; ".join(problems[:2]))
                continue
            if not all(contains_term(patched, expected) for _, expected in wanted):
                refuse(chunk.id, "expected terminology still missing")
                continue
            if repaired and self.sink is not None:
                self.sink.info(
                    "consistency",
                    chunk.id,
                    f"patch re-inserted {len(repaired)} dropped code marker(s)",
                )
            rendered[chunk.id] = patched
            report.patched_chunks += 1
            consecutive_refusals = 0
        return rendered, report
