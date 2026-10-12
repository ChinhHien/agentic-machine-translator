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
* patches are bounded (``CONSISTENCY_MAX_PATCHES``) and are kept only when they
  actually introduce the expected term, so a bad edit can never delete a chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.strip()).casefold()


def contains_term(haystack: str, needle: str) -> bool:
    """Case- and whitespace-insensitive containment test for a glossary term."""
    term = _normalize(needle)
    if not term:
        return False
    return term in _normalize(haystack)


@dataclass(frozen=True)
class Inconsistency:
    """One planned term that some chunks honour and others do not."""

    term: str
    expected: str
    missing_chunks: tuple[str, ...]
    compliant_chunks: tuple[str, ...]


@dataclass
class ConsistencyReport:
    """What the pass looked at and what it changed."""

    checked_terms: int = 0
    inconsistencies: int = 0
    patched_chunks: int = 0
    skipped_chunks: int = 0
    unresolved: list[str] = field(default_factory=list)

    def as_stats(self) -> dict[str, Any]:
        return {
            "consistency_terms": self.checked_terms,
            "consistency_inconsistencies": self.inconsistencies,
            "consistency_patches": self.patched_chunks,
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
    found: list[Inconsistency] = []
    for term, expected in pairs.items():
        term, expected = str(term).strip(), str(expected).strip()
        if not term or not expected:
            continue
        carrying = [c.id for c in chunks if contains_term(c.text, term)]
        if not carrying:
            continue
        compliant = tuple(
            cid for cid in carrying if contains_term(translations.get(cid, ""), expected)
        )
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
        report = ConsistencyReport(checked_terms=len(pairs))
        if not self.enabled or not pairs or not chunks:
            return translations, report

        issues = find_inconsistencies(chunks, translations, pairs)
        report.inconsistencies = len(issues)
        if not issues:
            return translations, report

        # One request per drifting chunk, carrying every term it must fix.
        fixes: dict[str, list[tuple[str, str]]] = {}
        for issue in issues:
            for chunk_id in issue.missing_chunks:
                fixes.setdefault(chunk_id, []).append((issue.term, issue.expected))
        logger.info(
            "Consistency: {} drifting term(s) across {} chunk(s)",
            len(issues),
            len(fixes),
        )

        ordered = [c for c in chunks if c.id in fixes]
        rendered = dict(translations)
        context = context or {}
        for index, chunk in enumerate(ordered):
            wanted = fixes[chunk.id]
            if index >= self.max_patches:
                report.skipped_chunks += 1
                if self.sink is not None:
                    self.sink.info(
                        "consistency",
                        chunk.id,
                        f"patch budget exhausted for {chunk.id}",
                    )
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
            if all(contains_term(patched, expected) for _, expected in wanted):
                rendered[chunk.id] = patched
                report.patched_chunks += 1
            else:
                report.unresolved.append(chunk.id)
                if self.sink is not None:
                    self.sink.info(
                        "consistency",
                        chunk.id,
                        "patch rejected: expected terminology still missing",
                    )
        return rendered, report
