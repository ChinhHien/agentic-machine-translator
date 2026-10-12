"""Deterministic repair of code placeholders the translator dropped.

Code blocks never reach the model: they are shielded into ``⟦CODE_n⟧`` markers
and substituted back verbatim at assembly time. That contract only holds if the
model echoes the markers, and it does not always do so — in the run that
motivated this module the Translator dropped 157 of 192 markers, so 82% of the
book's code disappeared without a single error, because restoration is a plain
``str.replace``: a marker absent from the reply is simply never substituted.

A dropped marker is recoverable, because the marker is its own paragraph on both
sides of the call. The chunk text joins source blocks with a blank line, so each
marker is a standalone unit, and the Translator prompt requires the same
paragraph structure back. We therefore re-insert a missing marker at the
paragraph offset it had in the source, which keeps the code next to the text it
belongs to instead of losing it.
"""

from __future__ import annotations

import re

#: Unit separator produced by the chunker's block join and by the Translator.
PARAGRAPH_SEP = "\n\n"

#: A shielded code marker.
_PH_RE = re.compile(r"⟦CODE_\d+⟧")


def _prose_offset(source_text: str) -> dict[str, int]:
    """How many prose units precede each marker in *source_text*."""
    offsets: dict[str, int] = {}
    prose = 0
    for unit in source_text.split(PARAGRAPH_SEP):
        stripped = unit.strip()
        match = _PH_RE.fullmatch(stripped)
        if match:
            offsets.setdefault(match.group(0), prose)
        elif stripped:
            prose += 1
    return offsets


def repair_code_placeholders(source_text: str, translation: str) -> tuple[str, list[str]]:
    """Re-insert code markers missing from *translation*.

    Returns ``(text, repaired)``: the translation with every source marker
    present again, plus the list of markers that had to be re-inserted (empty
    when the model behaved). Markers are placed after the same number of prose
    paragraphs as in the source, so a repair keeps the code where it belongs;
    anything that cannot be placed is appended rather than dropped.
    """
    if not source_text or not translation:
        return translation, []
    present = set(_PH_RE.findall(translation))
    offsets = _prose_offset(source_text)
    missing = {marker: offset for marker, offset in offsets.items() if marker not in present}
    if not missing:
        return translation, []

    by_offset: dict[int, list[str]] = {}
    for marker, offset in missing.items():
        by_offset.setdefault(offset, []).append(marker)

    out: list[str] = []
    repaired: list[str] = []
    prose_seen = 0
    for unit in translation.split(PARAGRAPH_SEP):
        is_marker = _PH_RE.fullmatch(unit.strip()) is not None
        if not is_marker:
            for marker in by_offset.pop(prose_seen, []):
                out.append(marker)
                repaired.append(marker)
            if unit.strip():
                prose_seen += 1
        out.append(unit)

    # Markers that pointed past the end of the reply: keep the code anyway.
    for offset in sorted(by_offset):
        for marker in by_offset[offset]:
            out.append(marker)
            repaired.append(marker)

    return PARAGRAPH_SEP.join(out), repaired
