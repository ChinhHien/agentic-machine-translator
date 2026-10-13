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


def write_table_pdf(
    path: str | Path,
    rows: list[list[str]],
    *,
    col_widths: list[float] | None = None,
    row_height: float = 18.0,
    left: float = _LEFT,
    top: float = _TOP,
    size: float = 9.0,
) -> Path:
    """Write *rows* as a ruled table on one PDF page and return the path.

    A table has to be *drawn* to be one: pdfplumber finds tables from their
    rules, not from how the glyphs happen to line up, so every row boundary gets
    a horizontal line and every column boundary a vertical one. The first row is
    set in bold, the way a header would be.
    """
    import pymupdf

    widths = list(col_widths) if col_widths else [90.0] * max(len(row) for row in rows)
    right = left + sum(widths)
    height = row_height * len(rows)
    doc = pymupdf.open()
    page = doc.new_page(width=_WIDTH, height=_HEIGHT)

    def column_x(column: int) -> float:
        return left + sum(widths[:column])

    for index in range(len(rows) + 1):
        y = top + index * row_height
        page.draw_line((left, y), (right, y))
    for index in range(len(widths) + 1):
        x = column_x(index)
        page.draw_line((x, top), (x, top + height))

    for r, row in enumerate(rows):
        font = _HEADING_FONT if r == 0 else _BODY_FONT
        for c, text in enumerate(row):
            page.insert_text(
                (column_x(c) + 3, top + r * row_height + 12),
                text,
                fontname=font,
                fontsize=size,
            )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    doc.close()
    return out


def write_styled_pdf(
    path: str | Path, before: str, bold: str, after: str, *, size: float = _BODY_SIZE
) -> Path:
    """One body line whose middle words are set in bold, then plain again.

    The three runs are drawn on **one baseline** — pdfplumber groups glyphs into
    a line by baseline, so this is a single source line that mixes styles, which
    is the only thing the emphasis sentinels need from a real PDF. Widths come
    from ``pymupdf.get_text_length`` so the runs abut the way word spacing does.
    """
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=_WIDTH, height=_HEIGHT)
    x = _LEFT
    for text, font in ((before, _BODY_FONT), (bold, _HEADING_FONT), (after, _BODY_FONT)):
        page.insert_text((x, _TOP), text, fontname=font, fontsize=size)
        x += pymupdf.get_text_length(text, fontname=font, fontsize=size)
    # A second line, so the page has a paragraph rather than a lone line whose
    # own gap would become the norm for paragraph-break detection.
    page.insert_text(
        (_LEFT, _TOP + _ADVANCE), "A plain following line.", fontname=_BODY_FONT, fontsize=size
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    doc.close()
    return out
