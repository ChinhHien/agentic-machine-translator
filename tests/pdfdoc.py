"""Render a small Markdown string into a real PDF, for tests that need a file.

Input is PDF-only, so a test that wants a document has to *build* one. This
renders Markdown using the very typographic cues the extractor reads: oversized
runs become headings, monospace runs become code, and a blank line becomes the
vertical gap that separates two paragraphs. It is not a general Markdown→PDF
converter — only what the test suite feeds it.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

_WIDTH, _HEIGHT = 595.0, 842.0
_LEFT = 72.0
#: First baseline. Below the band (12% of the page) that the extractor reads as a
#: running head, so text here is content.
_TOP = 150.0
#: Last baseline that still fits above the page's bottom edge.
_BOTTOM = 815.0

_BODY_FONT, _BODY_SIZE = "helv", 10.0
_CODE_FONT, _CODE_SIZE = "cour", 9.0
_HEADING_FONT = "hebo"
_HEADING_SIZES = {1: 24.0, 2: 18.0, 3: 15.0}

#: Line advance inside a paragraph, and the extra gap a blank line adds. The
#: extractor breaks a paragraph when a gap exceeds 1.35x the page's *typical*
#: line spacing — which it estimates from the page itself. Body text is
#: therefore wrapped, so a paragraph has several tight lines and the blank-line
#: gap stays clearly larger than the estimate; single-line paragraphs would make
#: their own gap the norm and no break would ever be seen.
_ADVANCE = 12.0
_GAP = 28.0

#: Characters per body line: the text column (451 pt) at 10 pt Helvetica.
_WRAP = 48

_INLINE_RE = re.compile(r"(\*\*|\*|`)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _plain(text: str) -> str:
    """Strip the inline Markdown the PDF cannot carry."""
    return _INLINE_RE.sub("", text)


def write_pdf(path: str | Path, markdown: str) -> Path:
    """Write *markdown* as a PDF at *path* and return the path."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=_WIDTH, height=_HEIGHT)
    y = _TOP

    def newline(delta: float) -> None:
        nonlocal page, y
        y += delta
        if y > _BOTTOM:
            page = doc.new_page(width=_WIDTH, height=_HEIGHT)
            y = _TOP

    def put(text: str, font: str, size: float) -> None:
        page.insert_text((_LEFT, y), text, fontname=font, fontsize=size)

    in_code = False
    gap = 0.0
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            gap = _GAP
            continue
        if not line.strip():
            gap = _GAP
            continue
        if in_code:
            newline(gap)
            gap = 0.0
            put(line, _CODE_FONT, _CODE_SIZE)
            newline(_ADVANCE)
            continue
        heading = _HEADING_RE.match(line.strip())
        if heading:
            level = min(len(heading.group(1)), 3)
            size = _HEADING_SIZES[level]
            newline(gap)
            gap = 0.0
            put(_plain(heading.group(2)), _HEADING_FONT, size)
            newline(size * 1.5)
            continue
        newline(gap)
        gap = 0.0
        wrapped = textwrap.wrap(
            _plain(line.strip()),
            width=_WRAP,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        for piece in wrapped:
            put(piece, _BODY_FONT, _BODY_SIZE)
            newline(_ADVANCE)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    doc.close()
    return out
