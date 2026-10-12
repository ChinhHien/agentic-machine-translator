"""Layout-aware PDF extraction: typography -> structured Markdown.

Why this module exists
----------------------
Guessing structure from text layout is not enough. A generic "PDF to Markdown"
converter once produced 3.157 table rows, zero headings and zero code fences on
the book this was written for, and downstream that was doubly fatal — the
markdown parser then had no prose branch for tables, so the text was dropped,
and the translator never saw a single heading to anchor on. That fallback is
gone: this is the only extractor now.

This extractor reads the PDF's own typography instead of guessing:

* **word gaps** are rebuilt from glyph coordinates. A TeX PDF often emits no
  space characters at all, so naive extraction glues words together
  (``bestforsmallproblemsandanotherforbigones``); a horizontal gap wider than
  :data:`_SPACE_GAP_RATIO` of the font size is a space;
* **code** is the monospace font runs, with indentation restored from the
  glyph x-offset;
* **headings** are the oversized runs, level taken from the font size;
* the **tables** pdfplumber can find are emitted as Markdown pipe tables, and
  the lines inside their bounding boxes are not emitted a second time as prose;
* the **top margin band** is the running head, and it is a band *of this page*
  (a fraction of its height) rather than a fixed number of points — 110 pt is a
  seventh of a portrait book page and a fifth of a landscape sheet;
* **headings and running heads are told apart** by the page number: a line is a
  running head when it *is* a page number or ends/starts with one, so a section
  title sitting in the margin band survives as a heading;
* both **font-size thresholds are derived from the document**, since the size
  that means "heading" differs by typesetter (see :func:`_font_thresholds`);
* **paragraph breaks** are the vertical gaps that are clearly larger than the
  page's own line spacing;
* hyphenated line breaks are rejoined;
* a **contents page** is kept as content — one entry per line, with the dot
  leader collapsed to a single separator. A TOC is text the reader asked to
  translate, and on a book whose section titles never became headings it is the
  only place they appear, so discarding it lost them silently.

The output is ordinary Markdown fed to the structural parser in
:mod:`polyglot_lc.ingest.md`, so chunking, code shielding and code restoration
behave exactly as they always did.

Alongside the Markdown, every emitted unit keeps the page rectangles it was read
from (see :func:`bind_page_geometry`). Those rectangles are what let the same
translation be re-rendered as a layout-preserving PDF — there is no second
classifier that could disagree about where a block belongs.
"""

from __future__ import annotations

import io
import itertools
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from polyglot_lc.ingest.md import markdown_table, parse_markdown
from polyglot_lc.ingest.model import Document

#: Font-name fragments that mean "monospace" (TeX's CMTT plus common PDF fonts).
_MONO_MARKERS = ("CMTT", "Courier", "Mono", "Consolas", "Menlo", "Typewriter")

#: A horizontal glyph gap wider than this fraction of the font size is a space.
_SPACE_GAP_RATIO = 0.10

#: Fraction of monospace glyphs that makes a line a line of code.
_CODE_RATIO = 0.75

#: Characters that only occur in code (a sentence never ends in ``;`` or ``}``).
#: Monospace alone is not enough: a prose line can be mostly a URL, which TeX
#: sets in the typewriter face too.
_CODE_CHARS = "{};"
#: Comment markers that make a monospace line code on their own.
_CODE_PREFIXES = ("//", "/*", "*", "#")

#: Fallback font sizes (pt) for heading detection, used only when the document
#: is too small to derive its own (see :func:`_font_thresholds`).
_HEADING_SIZE = 15.0
_H1_SIZE = 22.0

#: How many points larger than the body size a line must be to be a heading.
_HEADING_STEP = 2.0

#: Fewest text lines needed before the document's own font sizes are trusted.
_MIN_SIZE_SAMPLE = 20

#: Fraction of the page height that may hold a running head, and the band used
#: when a page reports no height (tests, exotic PDFs). A fixed band cannot work
#: for both a portrait A4 book and a landscape specification sheet: 110 pt is a
#: seventh of a 842-pt book page but a fifth of a 595-pt landscape page.
_HEADER_BAND_RATIO = 0.12
_HEADER_BAND = 110.0

#: Longest line that can still be a running head. A running head carries the
#: page number at one of its edges; content lines are longer and carry numbers
#: inside them ("Windows Server 2016 / 2019 / 2022").
_MAX_HEAD_CHARS = 60

#: A standalone page-number token: arabic digits or a roman numeral.
_PAGE_NUM_RE = re.compile(r"(?<![\w.])(?:\d{1,4}|[IVXLCDM]{1,7})(?![\w.])", re.IGNORECASE)
_PAGE_NUM_ONLY_RE = re.compile(r"(?:\d{1,4}|[IVXLCDM]{1,7})", re.IGNORECASE)

#: Vertical gap (as a multiple of the page's line spacing) that breaks a
#: paragraph.
_PARAGRAPH_GAP = 1.35

#: Bullet glyphs: unmapped PDF bullets plus real bullet characters.  A dash or
#: asterisk only counts when a space follows it, so a wrapped line is not
#: mistaken for a list item.
_BULLETS = ("(cid:136)", "(cid:127)", "(cid:149)", "•", "●", "◦")
_BULLET_CHARS = "-*"

#: Leading CMTT advance is 0.525 em; code indentation is restored from x-offset.
_MONO_ADVANCE = 0.525

#: Ordered-list markers ("1. ", "12. ").
_NUMBERED_PREFIX = "0123456789"

#: TeX ligature glyphs -> the letters they stand for.
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
#: This PDF doubles the ligature glyph ("eﬀﬀect", "Proﬁﬁling"); collapse it.
_DOUBLED_LIGATURE_RE = re.compile("([ﬀﬁﬂﬃﬄ])\\1+")
#: A page-number token at the end of a contents entry. Front matter is numbered
#: in roman numerals and the body in arabic, and one contents page carries both.
_TOC_PAGE_RE = r"(?:\d{1,4}|[IVXLCDM]{1,7})"

#: A dot-leader table-of-contents line ("Section . . . . . 42", "Preface .... xi").
#: The page number may be missing. Four dots minimum, so prose ending in an
#: ellipsis is not mistaken for an entry.
_TOC_LINE_RE = re.compile(rf"\.(?:\s*\.){{3,}}\s*(?:{_TOC_PAGE_RE})?\s*$", re.IGNORECASE)

#: The dot leader and the whitespace around it.
_TOC_LEADER_RE = re.compile(r"\s*\.(?:\s*\.){3,}\s*")

#: One separator in place of a leader run.
_TOC_ELLIPSIS = " … "


def normalize_toc_entry(text: str) -> str:
    """Collapse a contents entry's dot leader to a single ellipsis.

    The leader is the only part of a contents line that is pure layout: a run of
    thirty dots says nothing one separator does not, and it makes unrelated
    entries look alike to near-duplicate detection downstream.
    """
    return _TOC_LEADER_RE.sub(_TOC_ELLIPSIS, text).strip()


def normalize_toc_lines(markdown: str) -> str:
    """Collapse dot leaders on contents lines, keeping every entry.

    A contents page used to be deleted outright, on the theory that the
    document's own headings make it redundant. They do not: the contents page is
    text the reader asked to translate, and on a PDF whose section titles never
    got classified as headings it is the *only* place those titles appear — so
    dropping it lost them with no error. Only the leader goes; the entry stays.
    """
    kept = [
        normalize_toc_entry(line) if _TOC_LINE_RE.search(line.strip()) else line
        for line in markdown.splitlines()
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept))


def normalize_glyphs(text: str) -> str:
    """Fix ligature artifacts: doubled ligature glyphs and the glyphs themselves.

    The PDF text layer repeats a ligature glyph where the font maps one glyph
    to a two-letter sequence, so ``effect`` arrives as ``e`` + ``ﬀ`` + ``ﬀ`` +
    ``ect``. Collapse the repeat, then expand the ligature to plain letters so
    the translator and the output see ordinary text.
    """
    text = _DOUBLED_LIGATURE_RE.sub(r"\1", text)
    for glyph, letters in _LIGATURES.items():
        text = text.replace(glyph, letters)
    return text


#: Cheap language guess for the code fence (the book this was built for is Java).
_JAVA_HINTS = (
    "public ",
    "private ",
    "class ",
    "void ",
    "int ",
    "import ",
    "static ",
    "String",
    "boolean",
    "return ",
    "new ",
    "package ",
)


def _font_of(ch: dict[str, Any]) -> str:
    """Bare font name of a glyph (``SBIMKU+CMR12`` -> ``CMR12``)."""
    return str(ch.get("fontname") or "").split("+")[-1]


def _is_mono(font: str) -> bool:
    """True when *font* looks like a fixed-width face."""
    lowered = font.lower()
    return any(marker.lower() in lowered for marker in _MONO_MARKERS)


def _font_style(font: str) -> tuple[bool, bool]:
    """``(bold, italic)`` read off a font name (``Verdana-Bold`` -> bold)."""
    lowered = font.lower()
    return "bold" in lowered, ("italic" in lowered or "oblique" in lowered)


def _rgb(value: Any) -> tuple[float, ...] | None:
    """A glyph's non-stroking colour as an RGB triple, or ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return (float(value),) * 3
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        parts = [float(part) for part in value]
    except (TypeError, ValueError):
        return None
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return tuple(parts)
    if len(parts) == 4:  # CMYK: printed documents still arrive in it.
        c, m, y, k = parts
        return (
            round((1 - c) * (1 - k), 4),
            round((1 - m) * (1 - k), 4),
            round((1 - y) * (1 - k), 4),
        )
    return None


def _char_color(chars: Iterable[dict[str, Any]]) -> tuple[float, ...] | None:
    """The colour the line is painted in (its most common one), or ``None``."""
    seen: Counter[tuple[float, ...]] = Counter()
    for ch in chars:
        rgb = _rgb(ch.get("non_stroking_color"))
        if rgb is not None:
            seen[rgb] += 1
    return seen.most_common(1)[0][0] if seen else None


def _face(chars: list[dict[str, Any]]) -> dict[str, Any]:
    """The keywords :class:`_RawLine` takes to describe how a line *looks*.

    The dominant non-space glyph wins: a line's own bullet or a stray space run
    is set in another face, and the text's face is what the translation has to
    be printed in. The rebuild reads these to replace text without changing the
    document's typography.
    """
    named = Counter(_font_of(c) for c in chars if str(c.get("text") or "").strip())
    if not named:
        named = Counter(_font_of(c) for c in chars)
    font = named.most_common(1)[0][0] if named else ""
    bold, italic = _font_style(font)
    return {"font": font, "bold": bold, "italic": italic, "color": _char_color(chars)}


def looks_like_code(text: str) -> bool:
    """True when *text* is source code rather than prose set in a mono face."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(_CODE_PREFIXES):
        return True
    return any(ch in stripped for ch in _CODE_CHARS)


def line_text(chars: Iterable[dict[str, Any]]) -> str:
    """Rebuild a line's text, inserting a space wherever glyphs are far apart.

    TeX output has no space glyphs: inter-word spacing is pure positioning, so
    the gap between adjacent glyphs is the only evidence that a space belongs
    there. Intra-word gaps are ~0.03 em, inter-word gaps ~0.24 em.
    """
    ordered = sorted(chars, key=lambda c: c["x0"])
    out: list[str] = []
    prev: dict[str, Any] | None = None
    for ch in ordered:
        if prev is not None:
            gap = float(ch["x0"]) - float(prev["x1"])
            if gap > _SPACE_GAP_RATIO * max(float(prev["size"]), 1.0):
                out.append(" ")
        out.append(str(ch["text"]))
        prev = ch
    return "".join(out)


class _RawLine:
    """One visual line of a page, classified by role."""

    __slots__ = (
        "bold",
        "bottom",
        "cells",
        "color",
        "font",
        "italic",
        "kind",
        "level",
        "page",
        "size",
        "text",
        "top",
        "x0",
        "x1",
    )

    def __init__(
        self,
        page: int,
        top: float,
        kind: str,
        level: int,
        text: str,
        x0: float,
        size: float,
        x1: float = 0.0,
        bottom: float = 0.0,
        *,
        font: str = "",
        bold: bool = False,
        italic: bool = False,
        color: tuple[float, ...] | None = None,
        cells: list[list[list | None]] | None = None,
    ) -> None:
        self.page = page
        self.top = top
        self.kind = kind  # heading | code | body | bullet | numbered | header
        self.level = level
        self.text = text
        self.x0 = x0
        self.x1 = x1
        self.bottom = bottom
        self.size = size
        self.font = font
        self.bold = bold
        self.italic = italic
        self.color = color
        #: A table's cells, one entry per cell: box, inset, size, alignment,
        #: weight. Empty for every other kind of line.
        self.cells: list[list[list | None]] = cells or []


#: (page, x0, top, x1, bottom) in PDF points, origin at the top-left corner.
Rect = tuple[int, float, float, float, float]


class _Unit:
    """One emitted Markdown unit plus the page geometry it was read from.

    This is the single source of truth for PDF ingestion: the Markdown is
    rendered *from* these units and the rebuild path
    (:mod:`polyglot_lc.pipeline.pdf_output`) reads their rectangles, so the two
    can never disagree about where a translated block belongs.
    """

    __slots__ = (
        "bold",
        "cells",
        "color",
        "font",
        "italic",
        "kind",
        "level",
        "markers",
        "rects",
        "size",
        "text",
    )

    def __init__(self, kind: str, text: str, level: int = 0, size: float = 0.0) -> None:
        self.kind = kind  # heading | table | code | body | list | toc
        self.text = text
        self.level = level
        self.size = size
        #: One entry per source line, in reading order.
        self.rects: list[Rect] = []
        #: How the unit is set, so the rebuilt PDF can print its translation in
        #: the same face: the first line's font, weight, slant and colour.
        self.font = ""
        self.bold = False
        self.italic = False
        self.color: tuple[float, ...] | None = None
        #: The bullet glyphs this unit's items were marked with, item by item.
        self.markers: list[str] = []
        #: A table's cells (box, inset, size, alignment, weight), row by row.
        self.cells: list[list[list | None]] = []

    def take(self, lines: Iterable[_RawLine]) -> None:
        """Record the source lines that produced this unit."""
        for line in lines:
            self.rects.append((line.page, line.x0, line.top, line.x1, line.bottom))
            self.size = max(self.size, line.size)
            if line.cells:
                self.cells = line.cells
            if not self.font and line.font:
                self.font, self.bold, self.italic, self.color = (
                    line.font,
                    line.bold,
                    line.italic,
                    line.color,
                )


def _header_band(page: Any) -> float:
    """The top-of-page band that may hold a running head, in points.

    Derived from the page's own height so a landscape sheet does not get a book
    page's band. Pages that report no height (test stubs) fall back to the
    constant.
    """
    try:
        height = float(getattr(page, "height", 0.0) or 0.0)
    except (TypeError, ValueError):
        height = 0.0
    if height <= 0:
        return _HEADER_BAND
    return _HEADER_BAND_RATIO * height


def _looks_like_running_head(text: str) -> bool:
    """True when a top-of-page line is a running head rather than content.

    The evidence is the page number: either the line *is* one (``4``, ``iv``),
    or it carries one at an edge (``10 Chapter 2 Analysis``). A content line that
    happens to contain a year or a version (``Windows Server 2016 / 2019``,
    ``Eocortex 4.6``) has the number inside it, not at an edge, and is kept.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > _MAX_HEAD_CHARS:
        return False
    if _PAGE_NUM_ONLY_RE.fullmatch(stripped):
        return True
    match = _PAGE_NUM_RE.search(stripped)
    if match is None:
        return False
    return match.start() == 0 or match.end() == len(stripped)


def _classify(
    page_no: int,
    line: dict[str, Any],
    *,
    header_band: float = _HEADER_BAND,
    heading_min: float = _HEADING_SIZE,
    h1_min: float = _H1_SIZE,
) -> _RawLine | None:
    """Classify one pdfplumber line; ``None`` when it carries no content."""
    chars = line["chars"]
    text = normalize_glyphs(line_text(chars)).strip()
    top = float(line["top"])
    if not text:
        return None

    size = max(float(c["size"]) for c in chars)
    x0 = min(float(c["x0"]) for c in chars)
    x1 = max(float(c["x1"]) for c in chars)
    bottom = float(line.get("bottom") or top)
    face = _face(chars)

    def raw(
        kind: str,
        level: int,
        text: str,
        x0: float,
        size: float,
        x1: float = 0.0,
        bottom: float = 0.0,
    ) -> _RawLine:
        return _RawLine(page_no, top, kind, level, text, x0, size, x1, bottom, **face)

    # Oversized first: the first line of a page is often a section title, and
    # dropping it as a running head is how a document loses every heading.
    if size >= heading_min:
        level = 1 if size >= h1_min else 2
        return raw("heading", level, text, x0, size, x1, bottom)

    if top < header_band and _looks_like_running_head(text):
        return raw("header", 0, text, 0.0, 0.0)

    mono = sum(1 for c in chars if _is_mono(_font_of(c)))
    if mono / len(chars) >= _CODE_RATIO:
        # "mono" = monospace but reads like a sentence. TeX sets URLs and inline
        # identifiers in the typewriter face too, and a verbatim *listing* can
        # contain plain sentences (compiler output, numeric data). The run
        # context decides — see _resolve_mono_runs().
        kind = "code" if looks_like_code(text) else "mono"
        return raw(kind, 0, text, x0, size, x1, bottom)

    kind = "body"
    stripped = text.lstrip()
    if _TOC_LINE_RE.search(stripped):
        kind = "toc"
    elif any(stripped.startswith(mark) for mark in _BULLETS) or (
        stripped[:1] in _BULLET_CHARS and stripped[1:2] == " "
    ):
        kind = "bullet"
    elif len(stripped) > 3 and stripped[0] in _NUMBERED_PREFIX and stripped[1:3] in (". ", ") "):
        kind = "numbered"
    return raw(kind, 0, text, x0, size, x1, bottom)


def _resolve_mono_runs(lines: list[_RawLine]) -> None:
    """Turn a monospace line embedded in a monospace run into code.

    A line that only *looks* like prose is still part of a listing when both of
    its neighbours are in the same monospace run (that is what a ``verbatim``
    block with sentences or numbers in it looks like). An isolated monospace
    line is prose that happens to contain an identifier or a URL.
    """
    for i, ln in enumerate(lines):
        if ln.kind != "mono":
            continue
        neighbours = [lines[j] for j in (i - 1, i + 1) if 0 <= j < len(lines)]
        if len(neighbours) == 2 and all(n.kind in {"code", "mono"} for n in neighbours):
            ln.kind = "code"


def _text_lines(page: Any) -> list[dict[str, Any]]:
    """The raw pdfplumber lines of one page."""
    return list(page.extract_text_lines(layout=False))


def _font_thresholds(pages: list[Any]) -> tuple[float, float]:
    """``(heading_min, h1_min)`` derived from the document's own font sizes.

    The body size is the size most lines are set in; a heading is a couple of
    points larger. Deriving it beats a constant: a book may set headings at
    15 pt while this specification uses 14 pt over a 10 pt body, so any single
    threshold either loses one document's headings or invents the other's.

    Returns the fallback constants when the document is too small to have a
    meaningful distribution (short documents, unit-test stubs).
    """
    sizes: Counter[float] = Counter()
    for page in pages:
        for line in _text_lines(page):
            chars = line.get("chars") or []
            if not chars:
                continue
            sizes[round(max(float(c["size"]) for c in chars), 1)] += 1
    if sum(sizes.values()) < _MIN_SIZE_SAMPLE or len(sizes) < 2:
        return _HEADING_SIZE, _H1_SIZE
    body = sizes.most_common(1)[0][0]
    heading_min = body + _HEADING_STEP
    return heading_min, max(body * 1.9, heading_min + 4.0)


#: One cell's own box: ``(x0, top, x1, bottom)``, same frame as :data:`Rect`.
Cell = tuple[float, float, float, float]

#: Cell geometry used where the source says nothing (no line inside the cell).
_CELL_DEFAULT_SIZE = 10.0
_CELL_PAD_DEFAULT = 2.0
#: How far the source's own inset is honoured, and how far it is trusted.
_CELL_PAD_MIN = 0.5
_CELL_PAD_MAX = 2.0
#: How far off a cell's centre a line may sit and still read as centred.
_CELL_ALIGN_TOLERANCE = 2.0


def _cell_text(value: Any) -> str:
    """One table cell as a single line of text."""
    return " ".join(str(value or "").split())


def _cell_box(value: Any) -> Cell | None:
    """A pdfplumber cell box as a 4-tuple, or ``None`` when it has no box."""
    if value is None:
        return None
    try:
        x0, top, x1, bottom = (float(part) for part in tuple(value)[:4])
    except (TypeError, ValueError, IndexError):
        return None
    return (x0, top, x1, bottom)


def _union_box(first: Cell | None, second: Cell | None) -> Cell | None:
    """The box covering both cells, for the columns a merge collapsed."""
    if first is None:
        return second
    if second is None:
        return first
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _table_boxes(table: Any, nrows: int, ncols: int) -> list[list[Cell | None]]:
    """Cell boxes of *table*, one entry per cell of its extracted text matrix.

    A row whose reported cell count disagrees with the width of its text (the
    "phantom" columns pdfplumber sometimes reports) gets ``None`` throughout:
    aligning those boxes by index would print one cell's translation into its
    neighbour, which is worse than leaving that row in the source language.
    """
    rows = getattr(table, "rows", None) or ()
    boxes: list[list[Cell | None]] = []
    for row in rows:
        cells = [_cell_box(cell) for cell in getattr(row, "cells", None) or ()]
        boxes.append(cells if len(cells) == ncols else [None] * ncols)
    if len(boxes) != nrows:
        return [[None] * ncols for _ in range(nrows)]
    return boxes


def _table_grid(table: Any) -> tuple[list[list[str]], list[list[Cell | None]]] | None:
    """Normalised cell matrix of *table* plus the source box of every cell.

    pdfplumber reports a merged cell's text in *every* column it spans, so
    adjacent columns are merged when they never disagree (equal, or one of them
    empty). That turns an 11-column reading of a 6-column license matrix back
    into one column per license. The same merges are applied to the cell boxes,
    so index ``[i][j]`` of the text and of the boxes always mean the same cell.
    """
    try:
        raw = table.extract()
    except Exception as exc:  # noqa: BLE001 - a broken table must not kill ingest
        logger.debug("table extraction failed: {}", exc)
        return None
    rows = [[_cell_text(c) for c in row] for row in (raw or [])]
    if not rows:
        return None
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    boxes = _table_boxes(table, len(rows), width)

    kept: list[int] = [0]
    for col in range(1, width):
        prev = kept[-1]
        if all(r[prev] == r[col] or not r[prev] or not r[col] for r in rows):
            for r in rows:
                if not r[prev]:
                    r[prev] = r[col]
            for index in range(len(boxes)):
                boxes[index][prev] = _union_box(boxes[index][prev], boxes[index][col])
        else:
            kept.append(col)
    rows = [[r[c] for c in kept] for r in rows]
    boxes = [[row[c] for c in kept] for row in boxes]
    # A row with no text at all is not part of the table; its boxes go with it,
    # so the two matrices stay index-aligned.
    kept_rows = [(row, box) for row, box in zip(rows, boxes, strict=True) if any(row)]
    rows = [row for row, _ in kept_rows]
    boxes = [box for _, box in kept_rows]
    # A one-column "table" is a framed paragraph, not a table; a single row has
    # no header to compare against. Let those lines flow as normal text.
    if len(rows) < 2 or len(rows[0]) < 2:
        return None
    return rows, boxes


def _cell_entry(lines: list[_RawLine], box: Cell | None) -> list | None:
    """How one cell was set: box, inset, size, alignment and weight.

    The source's own lines inside the cell are the only evidence of its padding
    and alignment, so they decide both rather than a constant — a centred header
    or a right-aligned number stays where the source put it. The inset is kept
    small: the room belongs to a translation that is usually longer.
    """
    if box is None:
        return None
    x0, top, x1, bottom = box
    inside = [
        line
        for line in lines
        if line.kind not in {"table", "header"}
        and x0 <= (line.x0 + line.x1) / 2 <= x1
        and top <= line.top <= bottom
    ]
    size = max((line.size for line in inside), default=_CELL_DEFAULT_SIZE)
    if not inside:
        return [x0, top, x1, bottom, _CELL_PAD_DEFAULT, size, "left", False, False]
    gaps = [
        min(line.x0 - x0, x1 - line.x1, line.top - top, bottom - max(line.bottom, line.top))
        for line in inside
    ]
    pad = min(max(min(gaps), _CELL_PAD_MIN), _CELL_PAD_MAX)
    first = inside[0]
    box_center = (x0 + x1) / 2
    line_center = (first.x0 + first.x1) / 2
    if abs(line_center - box_center) <= _CELL_ALIGN_TOLERANCE:
        align = "center"
    elif abs(first.x1 - x1) <= _CELL_ALIGN_TOLERANCE and abs(first.x0 - x0) > _CELL_ALIGN_TOLERANCE:
        align = "right"
    else:
        align = "left"
    return [x0, top, x1, bottom, pad, size, align, first.bold, first.italic]


@dataclass
class _Table:
    """One detected table: its box, its Markdown, and a box per cell."""

    bbox: tuple[float, float, float, float]
    markdown: str
    boxes: list[list[Cell | None]]


def _page_tables(page: Any, page_no: int) -> list[_Table]:
    """The real tables of one page, with a source box for every cell."""
    finder = getattr(page, "find_tables", None)
    if not callable(finder):
        return []
    found: list[_Table] = []
    for table in finder():
        grid = _table_grid(table)
        if grid is None:
            continue
        rows, boxes = grid
        bbox = tuple(float(v) for v in table.bbox)
        # A nested table would emit content its outer table already carries.
        if any(
            other.bbox[0] <= bbox[0] and other.bbox[3] >= bbox[3] and other.bbox != bbox
            for other in found
        ):
            continue
        found.append(_Table(bbox=bbox, markdown=markdown_table(rows), boxes=boxes))
    return found


def _page_lines(
    page: Any,
    page_no: int,
    *,
    heading_min: float = _HEADING_SIZE,
    h1_min: float = _H1_SIZE,
) -> list[_RawLine]:
    """Classified lines of one page, in reading order, tables included.

    Lines that fall inside a detected table are replaced by the table itself, so
    a row is emitted once, as a row, instead of being flattened into prose.
    """
    header_band = _header_band(page)
    lines: list[_RawLine] = []
    for line in _text_lines(page):
        classified = _classify(
            page_no,
            line,
            header_band=header_band,
            heading_min=heading_min,
            h1_min=h1_min,
        )
        if classified is not None:
            lines.append(classified)

    tables = _page_tables(page, page_no)
    if tables:
        for table in tables:
            x0, top, x1, bottom = table.bbox
            # Read the cells' own typography off the lines this table replaces,
            # before those lines are dropped below.
            cells = [[_cell_entry(lines, box) for box in row] for row in table.boxes]
            lines.append(
                _RawLine(
                    page_no,
                    top,
                    "table",
                    0,
                    table.markdown,
                    x0,
                    0.0,
                    x1=x1,
                    bottom=bottom,
                    cells=cells,
                )
            )
        lines = [
            ln
            for ln in lines
            if ln.kind == "table" or not any(t.bbox[1] <= ln.top <= t.bbox[3] for t in tables)
        ]
        lines.sort(key=lambda ln: (ln.top, ln.kind != "table"))

    _resolve_mono_runs(lines)
    return lines


def _line_spacing(lines: list[_RawLine]) -> float:
    """Typical vertical gap between consecutive content lines of a page.

    Used as the yardstick for "this gap is a paragraph break": a TeX page has a
    uniform leading (15.2 pt here) and paragraph skips are clearly larger. The
    *lower quartile* is the estimator, not the median — on a page whose lines
    are mostly paragraph starts the median would drift up towards the paragraph
    gap and stop recognising breaks.
    """
    tops = [ln.top for ln in lines if ln.kind not in {"header", "table"}]
    deltas = sorted(b - a for a, b in itertools.pairwise(tops) if 0 < b - a < 60)
    if not deltas:
        return 0.0
    if len(deltas) == 1:
        return deltas[0]
    return statistics.quantiles(deltas, n=4)[0]


def _is_new_block(prev: _RawLine | None, cur: _RawLine, spacing: float) -> bool:
    """True when *cur* starts a fresh paragraph/block rather than continuing."""
    if prev is None:
        return True
    if prev.kind != cur.kind and {prev.kind, cur.kind} & {"heading", "code"}:
        return True
    if prev.kind in {"bullet", "numbered"} and cur.kind in {"bullet", "numbered"}:
        return True
    if prev.kind == "toc" or cur.kind == "toc":
        return True  # a table of contents is one entry per line
    if spacing <= 0:
        return False
    return (cur.top - prev.top) > _PARAGRAPH_GAP * spacing


def _join_wrapped(parts: list[str]) -> str:
    """Join wrapped source lines of one paragraph back into one line.

    Rejoins words hyphenated across a line break ("previ-" + "ous") and
    otherwise separates with a single space.
    """
    out = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if not out:
            out = piece
            continue
        if out.endswith("-") and piece[:1].islower():
            out = out[:-1] + piece
        else:
            out = out + " " + piece
    return out


def _code_block(lines: list[_RawLine]) -> str:
    """Render code lines as a fenced block, restoring indentation from x-offset.

    The zero column is per page: the whole text column shifts between pages
    (a chapter opener is indented), so a block continuing onto the next page
    would otherwise gain a phantom indent.
    """
    page_base: dict[int, float] = {}
    for ln in lines:
        page_base[ln.page] = min(page_base.get(ln.page, ln.x0), ln.x0)
    rendered: list[str] = []
    for ln in lines:
        char_width = max(_MONO_ADVANCE * (ln.size or 12.0), 0.5)
        indent = round((ln.x0 - page_base[ln.page]) / char_width)
        rendered.append(" " * max(indent, 0) + ln.text.rstrip())
    body = "\n".join(rendered)
    joined = body[:2000]
    language = "java" if any(hint in joined for hint in _JAVA_HINTS) else ""
    return f"```{language}\n{body}\n```"


def _bullet_text(text: str) -> str:
    """Strip the bullet glyph(s) from a list-item line.

    The glyph is repeated on some items (this PDF emits eight copies of the same
    unmapped bullet before the item text), so strip until none is left.
    """
    stripped = text.lstrip()
    while stripped:
        for mark in (*_BULLETS, *_BULLET_CHARS):
            if stripped.startswith(mark) and (mark in _BULLETS or stripped[1:2] == " "):
                stripped = stripped[len(mark) :].lstrip()
                break
        else:
            break
    return stripped or text.lstrip()


def _bullet_marker(text: str) -> str:
    """The bullet glyph a list-item line starts with, or ``""``.

    The rebuild prints it again: the source glyph is redacted along with the
    item's text, and a list that lost its marks no longer reads as a list.
    """
    stripped = text.lstrip()
    for mark in (*_BULLETS, *_BULLET_CHARS):
        if stripped.startswith(mark) and (mark in _BULLETS or stripped[1:2] == " "):
            return mark
    return ""


def _number_marker(text: str) -> str:
    """The ``1.``/``2)`` label a numbered list item starts with, or ``""``."""
    match = re.match(r"\s*(\d{1,3}[.)])(?=\s)", text)
    return match.group(1) if match else ""


def _build_units(pages: list[Any]) -> list[_Unit]:
    """Group every page into Markdown units, keeping each unit's geometry.

    The Markdown string is rendered from these units and the rebuild path reads
    their rectangles, so the pages are grouped exactly once — there is no second
    classifier that could disagree about where a translated block belongs.
    """
    heading_min, h1_min = _font_thresholds(pages)
    classified = [
        (page_no, ln)
        for page_no, page in enumerate(pages)
        for ln in _page_lines(page, page_no, heading_min=heading_min, h1_min=h1_min)
    ]
    spacing_by_page = {
        page_no: _line_spacing([ln for pg, ln in classified if pg == page_no])
        for page_no in {pg for pg, _ in classified}
    }

    units: list[_Unit] = []
    # (line, rendered text, bullet glyph) of one unit.
    pending: list[tuple[_RawLine, str, str]] = []
    pending_kind = "body"
    code_run: list[_RawLine] = []
    prev: _RawLine | None = None
    last_heading: tuple[int, str] | None = None

    def flush() -> None:
        nonlocal pending, pending_kind
        if not pending:
            return
        text = _join_wrapped([rendered for _, rendered, _ in pending])
        if text:
            unit = _Unit(pending_kind, text)
            unit.take([line for line, _, _ in pending])
            unit.markers = [marker for _, _, marker in pending if marker]
            units.append(unit)
        pending, pending_kind = [], "body"

    def flush_code() -> None:
        nonlocal code_run
        if code_run:
            unit = _Unit("code", _code_block(code_run))
            unit.take(code_run)
            units.append(unit)
            code_run = []

    def emit(kind: str, text: str, level: int = 0) -> _Unit:
        unit = _Unit(kind, text, level=level)
        units.append(unit)
        return unit

    for page_no, ln in classified:
        if ln.kind == "header":
            continue
        spacing = spacing_by_page.get(page_no, 0.0) or 15.0

        if ln.kind == "table":
            flush()
            flush_code()
            last_heading = None
            emit("table", ln.text).take([ln])
            prev = ln
            continue

        if ln.kind == "heading":
            flush()
            flush_code()
            if (
                last_heading
                and last_heading[0] == ln.level
                and units
                and units[-1].kind == "heading"
            ):
                # "Chapter 3" / "ArrayList" are two oversized lines of one title.
                units[-1].text += " " + ln.text
                units[-1].take([ln])
            else:
                emit("heading", "#" * ln.level + " " + ln.text, level=ln.level).take([ln])
            last_heading = (ln.level, ln.text)
            prev = ln
            continue

        if ln.kind == "code":
            flush()
            code_run.append(ln)
            last_heading = None
            prev = ln
            continue

        flush_code()
        last_heading = None
        new_block = _is_new_block(prev, ln, spacing)
        if ln.kind == "toc":
            flush()
            emit("toc", ln.text).take([ln])
            prev = ln
            continue
        if ln.kind in {"bullet", "numbered"}:
            if new_block:
                flush()  # closes the previous item (or paragraph)
            pending_kind = "list"
            item = _bullet_text(ln.text) if ln.kind == "bullet" else ln.text.strip()
            marker = _bullet_marker(ln.text) if ln.kind == "bullet" else _number_marker(ln.text)
            # Markdown needs the marker back for the parser to see a list.
            pending.append((ln, "- " + item if ln.kind == "bullet" else item, marker))
        else:
            if new_block:
                flush()
            # A body line right under a list item is that item's next line.
            if pending_kind != "list":
                pending_kind = "body"
            pending.append((ln, ln.text, ""))
        prev = ln

    flush()
    flush_code()
    return units


def render_markdown(units: list[_Unit]) -> str:
    """Render units to Markdown exactly as the emitted lines always were."""
    out = ""
    for i, unit in enumerate(units):
        text = unit.text
        if not out:
            out = text
        elif unit.kind == "list" and units[i - 1].kind == "list":
            out += "\n" + text
        else:
            out += "\n\n" + text
    return out


def pages_to_markdown(pages: list[Any]) -> str:
    """Convert pdfplumber pages into Markdown (headings, tables, code, prose)."""
    return render_markdown(_build_units(pages))


#: A shielded code block survives the Markdown parse as this placeholder.
_CODE_PLACEHOLDER_RE = re.compile(r"^⟦CODE_\d+⟧$")


def _units_for_block(units: list[_Unit], index: int, block: Any) -> int | None:
    """How many units at *index* make up *block*, or ``None`` when unsure.

    The blocks come from parsing the Markdown rendered from these very units, so
    they line up in order: a list block owns one unit per item, a heading owns
    one heading unit, and so on. Answering ``None`` instead of guessing keeps one
    surprise from shifting every later block's geometry onto the wrong lines.
    """
    if index >= len(units):
        return None
    first = units[index]
    if block.kind == "list":
        span = max(1, len(block.meta.get("items") or []))
        window = units[index : index + span]
        if len(window) == span and all(unit.kind == "list" for unit in window):
            return span
        return 1 if first.kind == "list" else None
    if block.kind == "heading":
        return 1 if first.kind == "heading" else None
    if _CODE_PLACEHOLDER_RE.match(block.text.strip()):
        return 1 if first.kind == "code" else None
    if block.meta.get("table"):
        return 1 if first.kind == "table" else None
    return 1 if first.kind in {"body", "toc"} else None


def bind_page_geometry(doc: Document, units: list[_Unit]) -> int:
    """Attach each block's source rectangles as ``block.meta["pdf"]``.

    Returns how many blocks were bound. A block left unbound is simply not
    rebuilt (its source text stays), which is the honest outcome: a mis-bound
    rectangle would print the translation over the wrong lines.
    """
    index = 0
    bound = 0
    for block in doc.blocks:
        span = _units_for_block(units, index, block)
        if span is None:
            # A unit the Markdown parser folded into a neighbouring block is
            # skipped, not mis-assigned, so the blocks after it stay aligned.
            for ahead in range(1, 4):
                span = _units_for_block(units, index + ahead, block)
                if span is not None:
                    logger.debug(
                        "PDF geometry: skipped {} unmatched unit(s) before {!r}",
                        ahead,
                        block.text[:40],
                    )
                    index += ahead
                    break
        if span is None:
            logger.debug("PDF geometry: no unit for block {!r}", block.text[:40])
            continue
        owned = units[index : index + span]
        markers = [mark for unit in owned for mark in unit.markers if mark]
        geometry: dict[str, Any] = {
            "rects": [rect for unit in owned for rect in unit.rects],
            "size": max((unit.size for unit in owned), default=0.0),
            "kind": owned[0].kind,
            # How the source set this block: the rebuild prints the translation
            # in the same face and colour instead of a generic substitute.
            "font": owned[0].font,
            "bold": owned[0].bold,
            "italic": owned[0].italic,
            "color": owned[0].color,
            # The markers the list's items carried, item by item.
            "markers": markers,
        }
        # A table also carries where each of its cells is, so the rebuild can
        # print the translation of a cell inside that cell's own box.
        if owned[0].kind == "table" and owned[0].cells:
            geometry["cells"] = owned[0].cells
        block.meta["pdf"] = geometry
        index += span
        bound += 1
    if bound != len(doc.blocks):
        logger.warning(
            "PDF geometry: bound {} of {} blocks; unbound blocks keep their "
            "source text in the rebuilt PDF",
            bound,
            len(doc.blocks),
        )
    return bound


def load_pdf_layout(data: bytes) -> Document:
    """Extract *data* with the layout-aware extractor and parse it to blocks.

    Each block also carries the page rectangles it was read from (see
    :func:`bind_page_geometry`), which is what lets the same translation be
    re-rendered as a layout-preserving PDF without a second extraction pass.
    """
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = list(pdf.pages)
        units = _build_units(pages)
        markdown = render_markdown(units)
    if not markdown.strip():
        raise ValueError("Layout extractor found no text in the PDF")
    markdown = normalize_toc_lines(markdown)
    doc = parse_markdown(
        unicodedata.normalize("NFC", markdown),
        source_format="pdf",
        extra_metadata={"converter": "layout", "pages": len(pages)},
    )
    bind_page_geometry(doc, units)
    logger.debug("Layout extractor: {} pages -> {} blocks", len(pages), len(doc.blocks))
    return doc
