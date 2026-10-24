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
* **emphasis** is read off the same runs: a bold or italic span inside a line is
  marked with a sentinel pair (``⟦BOLD⟧…⟦/BOLD⟧``) around the words it covers,
  so the line can be translated and printed with its emphasis intact, while a
  line set in a single style emits no sentinel at all. See
  :mod:`polyglot_lc.emphasis`;
* the **tables** pdfplumber can find are emitted as Markdown pipe tables, and
  the lines inside their bounding boxes are not emitted a second time as prose;
* the **top margin band** is the running head, and it is a band *of this page*
  (a fraction of its height) rather than a fixed number of points — 110 pt is a
  seventh of a portrait book page and a fifth of a landscape sheet;
* **headings and running heads are told apart** by the page number: a line is a
  running head when it *is* a page number or carries *this page's own* folio at
  an edge — the offset between the printed folio and the PDF's page index is read
  off the document's own bare folios — so a section title sitting in the margin
  band, a table row opening with its row number ("1 IBM Armonk …") and a
  contents entry ending with the page it points at all survive as content. A
  numeral has to be well formed to count ("iv", not "id");
* the **bottom band** is the running foot (the folio), and a line there that is
  *nothing but* a page number is dropped like a running head — it is furniture
  repeated on every page, and left in the flow it glues itself to the first line
  of the next page: the next page's first line is above this page's last one in
  its own coordinates, so the gap between them reads as no gap at all. The block
  that came out of that pair owned only the folio's rectangle, a few points wide;
* a **page break** is therefore a text question, not a geometric one: the page's
  last line and the next page's first are one paragraph only when the sentence
  reads on (no full stop, and the new line starts mid-sentence);
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
from functools import lru_cache
from typing import Any

from loguru import logger

from polyglot_lc import emphasis
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

#: Fraction of the page height that may hold a running foot — the *folio*, the
#: page number set below the text block. The symmetric band to
#: `_HEADER_BAND_RATIO`, and a fraction for the same reason: 52 pt is a twelfth
#: of a book page but a quarter of a landscape sheet, so a fixed band cannot
#: serve both.
_FOOTER_BAND_RATIO = 0.08

#: A standalone page-number token: arabic digits or a roman numeral. The numeral
#: is validated by :func:`_roman_value` — a bare run of the seven letters is not
#: one, and it read words too: a table's header row opening with "id" (ID, 501)
#: or a list item "• C" (C, 100) was taken for a folio and dropped whole.
_PAGE_NUM_ONLY_RE = re.compile(r"(?:\d{1,4}|[IVXLCDM]+)", re.IGNORECASE)
#: A well-formed Roman numeral: subtractive notation, in order, nothing left over.
_ROMAN_NUMERAL_RE = re.compile(
    r"M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", re.IGNORECASE
)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
#: Fewest bare folios the document must show before its own pagination offset is
#: trusted. One stray number in a header is not a pagination.
_MIN_FOLIO_SAMPLE = 3

#: A line that ends here ends a paragraph. Used where the *geometry* of a break
#: says nothing: the page's last line to the next page's first.
_SENTENCE_END = ".!?:;…"
#: Vertical gap (as a multiple of the page's line spacing) that breaks a
#: paragraph.
_PARAGRAPH_GAP = 1.35

#: Two gaps within this many points of each other are the same rhythm: a page's
#: leading is one measurement repeated, so near-equal values are the same run of
#: lines jittered by the PDF's own rounding rather than two different rhythms.
_GAP_TOLERANCE = 1.0

#: A figure's own label is set this many points *below* the page's body size, and
#: is no wider than this share of the page's text block. Both tests are needed: a
#: smaller face alone is also a footnote or a caption, and a narrow line alone is
#: also the last line of a paragraph — which sits right *at* the body size, and is
#: what the size test keeps out.
_LABEL_SIZE_STEP = 0.8
_LABEL_WIDTH_SHARE = 0.45

#: Bullet glyphs: unmapped PDF bullets plus real bullet characters.  A dash or
#: asterisk only counts when a space follows it, so a wrapped line is not
#: mistaken for a list item.
_BULLETS = ("(cid:136)", "(cid:127)", "(cid:149)", "•", "●", "◦")
_BULLET_CHARS = "-*"

#: Leading CMTT advance is 0.525 em; code indentation is restored from x-offset.
_MONO_ADVANCE = 0.525

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


#: Superscript forms of the characters an exponent is written with. A PDF sets
#: an exponent as a *small raised glyph*, not as the character ``²``; reading
#: the glyph as its ordinary digit is how ``n²`` reaches the translator as
#: ``n2`` and leaves the rebuilt page without its exponent.
_SUPERSCRIPT = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")


def _baseline(chars: list[dict[str, Any]]) -> tuple[float, float | None]:
    """The line's main size and the bottom of its full-size glyphs.

    ``bottom`` is in pdfplumber's top-down coordinates, so a raised glyph has a
    *smaller* bottom than the text it is raised above.
    """
    size = max(float(ch["size"]) for ch in chars)
    bottoms = [
        float(ch["bottom"])
        for ch in chars
        if "bottom" in ch and abs(float(ch["size"]) - size) < 0.05
    ]
    return size, (max(bottoms) if bottoms else None)


def _superscript(ch: dict[str, Any], size: float, baseline: float | None) -> str | None:
    """*ch* as a superscript character, or ``None`` when it is ordinary text."""
    if baseline is None or "bottom" not in ch:
        return None
    if float(ch["size"]) > size * 0.85:
        return None
    if float(ch["bottom"]) > baseline - size * 0.08:
        return None
    mapped = str(ch["text"]).translate(_SUPERSCRIPT)
    return mapped if mapped != str(ch["text"]) else None


def line_spans(chars: Iterable[dict[str, Any]]) -> list[tuple[str, bool, bool]]:
    """A line's text split into runs of one style, in reading order.

    Every glyph carries its own face, so a mixed-style line is a lossless source
    of emphasis: a title line that reads "**Eocortex ST** được thiết kế để"
    arrives here as three runs and leaves as three, where the block-level font
    record would keep only the majority face and lose the bold phrase entirely.

    Spaces are inserted on the same rule as :func:`line_text` — a gap wider than
    :data:`_SPACE_GAP_RATIO` of the size is a word gap — so a uniform line yields
    exactly the same characters as before, in one run.
    """
    ordered = sorted(chars, key=lambda c: c["x0"])
    size, baseline = _baseline(ordered)
    runs: list[tuple[str, bool, bool]] = []
    prev: dict[str, Any] | None = None
    for ch in ordered:
        style = _font_style(_font_of(ch))
        raised = _superscript(ch, size, baseline)
        if raised is not None:
            # An exponent follows its base with no word gap, whatever the
            # smaller glyph's advance looks like.
            text = raised
        else:
            gap = ""
            if prev is not None and float(ch["x0"]) - float(prev["x1"]) > _SPACE_GAP_RATIO * max(
                float(prev["size"]), 1.0
            ):
                gap = " "
            text = gap + str(ch["text"])
        if runs and runs[-1][1:] == style:
            runs[-1] = (runs[-1][0] + text, *style)
        else:
            runs.append((text, *style))
        prev = ch
    return [(normalize_glyphs(text), bold, italic) for text, bold, italic in runs]


def _join_wrapped_runs(
    parts: list[list[tuple[str, bool, bool]]],
) -> list[tuple[str, bool, bool]]:
    """Join the style runs of the source lines of one paragraph.

    The same rule as :func:`_join_wrapped` — separate two lines with a space, and
    heal a word hyphenated across the break — applied to runs, so that a bold
    phrase wrapped over two source lines becomes **one** run (and so one pair of
    tokens) instead of a pair closing and reopening at every line end.
    """
    out: list[tuple[str, bool, bool]] = []
    for part in parts:
        runs = _trim_runs(part)
        if not runs:
            continue
        if out:
            tail, bold, italic = out[-1]
            head = runs[0][0]
            if tail.endswith("-") and head[:1].islower():
                out[-1] = (tail[:-1], bold, italic)  # "previ-" + "ous"
            else:
                out[-1] = (tail + " ", bold, italic)
        for text, bold, italic in runs:
            if out and out[-1][1:] == (bold, italic):
                out[-1] = (out[-1][0] + text, bold, italic)
            else:
                out.append((text, bold, italic))
    return [(text, bold, italic) for text, bold, italic in out if text]


def _trim_runs(
    runs: list[tuple[str, bool, bool]],
) -> list[tuple[str, bool, bool]]:
    """Drop a line's edge whitespace without touching the gaps inside it.

    Only the first run's left edge and the last run's right edge are trimmed: a
    run's leading space is the word gap *inside* the line, and stripping it would
    glue two words together across a style change.
    """
    kept = [(text, bold, italic) for text, bold, italic in runs if text]
    if not kept:
        return []
    if len(kept) == 1:
        text = kept[0][0].strip()
        return [(text, kept[0][1], kept[0][2])] if text else []
    head = (kept[0][0].lstrip(), kept[0][1], kept[0][2])
    tail = (kept[-1][0].rstrip(), kept[-1][1], kept[-1][2])
    return [head, *kept[1:-1], tail]


def _wrap_run(text: str, bold: bool, italic: bool) -> str:
    """*text* wrapped in the token pair its style earns.

    A run's leading whitespace stays outside the pair. That space is the word gap
    between two runs, not part of the emphasised phrase, and inside the pair it
    would read as though the emphasis covered the gap too.
    """
    body = text.lstrip()
    space = text[: len(text) - len(body)]
    if not body:
        return text
    if bold and italic:
        return (
            f"{space}{emphasis.ITALIC_OPEN}{emphasis.BOLD_OPEN}{body}"
            f"{emphasis.BOLD_CLOSE}{emphasis.ITALIC_CLOSE}"
        )
    if bold:
        return f"{space}{emphasis.BOLD_OPEN}{body}{emphasis.BOLD_CLOSE}"
    if italic:
        return f"{space}{emphasis.ITALIC_OPEN}{body}{emphasis.ITALIC_CLOSE}"
    return text


def mark_emphasis(runs: list[tuple[str, bool, bool]]) -> str:
    """The paragraph's text, with a token pair around each emphasised run.

    A paragraph set in a single style gets no tokens at all. Marking it would
    put four extra tokens through every prompt for nothing — there is no
    emphasis to carry — so only a genuinely mixed line pays for them.
    """
    kept = [(text, bold, italic) for text, bold, italic in runs if text]
    if len({(bold, italic) for _, bold, italic in kept}) < 2:
        return "".join(text for text, _, _ in kept)
    return "".join(_wrap_run(text, bold, italic) for text, bold, italic in kept)


def line_text(chars: Iterable[dict[str, Any]]) -> str:
    """Rebuild a line's text, inserting a space wherever glyphs are far apart.

    TeX output has no space glyphs: inter-word spacing is pure positioning, so
    the gap between adjacent glyphs is the only evidence that a space belongs
    there. Intra-word gaps are ~0.03 em, inter-word gaps ~0.24 em.
    """
    ordered = sorted(chars, key=lambda c: c["x0"])
    size, baseline = _baseline(ordered)
    out: list[str] = []
    prev: dict[str, Any] | None = None
    for ch in ordered:
        raised = _superscript(ch, size, baseline)
        if raised is not None:
            out.append(raised)
            prev = ch
            continue
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
        "label",
        "level",
        "page",
        "size",
        "spans",
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
        spans: list[tuple[str, bool, bool]] | None = None,
    ) -> None:
        self.page = page
        self.top = top
        self.kind = kind  # heading | code | body | bullet | numbered | header
        #: True when this line is a figure's own label, not prose: set smaller
        #: and narrower than the page's body text (see :func:`_mark_labels`).
        #: The kind is left alone, so a label is still translated and printed as
        #: the body line it is; the flag only keeps it out of a paragraph.
        self.label = False
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
        #: The line's style runs, one per stretch of one ``(bold, italic)`` pair.
        #: ``font``/``bold``/``italic`` above stay the line's *dominant* face,
        #: which is what the rebuilt page prints a translation in when the block
        #: carries no emphasis at all.
        self.spans: list[tuple[str, bool, bool]] = spans or []


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


def _footer_band(page: Any) -> float:
    """The *y* below which a line is in the page's foot, in points.

    A threshold, not a height: the band is measured from the page's bottom edge
    and compared against a line's ``top`` (PDF coordinates run downwards).
    ``inf`` when the page reports no height, so nothing is taken for a running
    foot and every content line of a page we cannot measure is kept.
    """
    try:
        height = float(getattr(page, "height", 0.0) or 0.0)
    except (TypeError, ValueError):
        height = 0.0
    if height <= 0:
        return float("inf")
    return height * (1.0 - _FOOTER_BAND_RATIO)


def _roman_value(token: str) -> int | None:
    """The value of *token* when it is a well-formed Roman numeral, else ``None``.

    Well-formed matters: ``[IVXLCDM]+`` matches ``id`` and ``mix`` too, and a
    word that happens to be spelled in numeral letters is not a folio.
    """
    upper = token.upper()
    if not upper or _ROMAN_NUMERAL_RE.fullmatch(upper) is None:
        return None
    total = 0
    for index, letter in enumerate(upper):
        value = _ROMAN_VALUES[letter]
        following = _ROMAN_VALUES[upper[index + 1]] if index + 1 < len(upper) else 0
        total += -value if following > value else value
    return total


def _folio_value(token: str) -> int | None:
    """The number *token* stands for as a printed folio: ``"42"`` or ``"xiv"``."""
    stripped = token.strip(".,;:()[]")
    if _PAGE_NUM_ONLY_RE.fullmatch(stripped) is None:
        return None
    if stripped.isdigit():
        return int(stripped)
    return _roman_value(stripped)


def _page_num_only(text: str) -> int | None:
    """The folio a line consists of, or ``None`` when it is more than one."""
    stripped = text.strip()
    if not stripped or " " in stripped:
        return None
    return _folio_value(stripped)


def _starts_with_bullet(text: str) -> bool:
    """Whether a line opens with a bullet glyph, which furniture never does."""
    stripped = text.lstrip()
    return any(stripped.startswith(mark) for mark in _BULLETS) or (
        stripped[:1] in _BULLET_CHARS and stripped[1:2] == " "
    )


def _looks_like_running_head(text: str, *, folio: int | None = None) -> bool:
    """True when a top-of-page line is a running head rather than content.

    The evidence is the page number: either the line *is* one (``4``, ``iv``),
    or it carries one at an edge (``10 Chapter 2 Analysis``). A content line that
    happens to contain a year or a version (``Windows Server 2016 / 2019``,
    ``Eocortex 4.6``) has the number inside it, not at an edge, and is kept.

    *folio* is this page's own printed page number, when the document tells us
    (see :func:`_folio_offset`). It is the difference between furniture and a
    line that merely starts or ends in a number: a table row opening with its
    own row number (``1 IBM Armonk New York USA``) or a contents entry ending
    with the page it points at (``… Authors 38``) is content, and used to be
    dropped as a running head. Without that evidence the edge number is taken
    at its word — the old, more forgiving reading.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > _MAX_HEAD_CHARS:
        return False
    if _page_num_only(stripped) is not None:
        return True
    tokens = stripped.split()
    if not tokens:
        return False
    for token in (tokens[0], tokens[-1]):
        value = _folio_value(token)
        if value is None:
            continue
        if folio is None or value == folio:
            return True
    return False


def _classify(
    page_no: int,
    line: dict[str, Any],
    *,
    header_band: float = _HEADER_BAND,
    footer_band: float = float("inf"),
    heading_min: float = _HEADING_SIZE,
    h1_min: float = _H1_SIZE,
    folio: int | None = None,
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
    spans = line_spans(chars)

    def raw(
        kind: str,
        level: int,
        text: str,
        x0: float,
        size: float,
        x1: float = 0.0,
        bottom: float = 0.0,
        spans: list[tuple[str, bool, bool]] | None = None,
    ) -> _RawLine:
        return _RawLine(
            page_no, top, kind, level, text, x0, size, x1, bottom, **face, spans=spans or []
        )

    # Oversized first: the first line of a page is often a section title, and
    # dropping it as a running head is how a document loses every heading.
    if size >= heading_min:
        level = 1 if size >= h1_min else 2
        return raw("heading", level, text, x0, size, x1, bottom, spans=spans)

    if (
        top < header_band
        and not _starts_with_bullet(text)
        and _looks_like_running_head(text, folio=folio)
    ):
        return raw("header", 0, text, 0.0, 0.0)

    # The page's own folio, dropped the way a running head is. Leaving it in the
    # flow is what makes a block that carries half of *this* page and half of the
    # next: the first line of the next page is above the last line of this one in
    # its own coordinates, so the gap test reads the folio and that line as one
    # paragraph — and the pair then owns only the folio's rectangle, a sliver a
    # few points wide, which is a page of one letter per line once it is rebuilt.
    # The rule is deliberately narrower than the head's: a foot that carries
    # words as well as its number is something the reader can see, and dropping
    # it would lose those words. A bare number below the text block is furniture.
    if top >= footer_band and _page_num_only(text) is not None:
        return raw("footer", 0, text, 0.0, 0.0)

    mono = sum(1 for c in chars if _is_mono(_font_of(c)))
    if mono / len(chars) >= _CODE_RATIO:
        # "mono" = monospace but reads like a sentence. TeX sets URLs and inline
        # identifiers in the typewriter face too, and a verbatim *listing* can
        # contain plain sentences (compiler output, numeric data). The run
        # context decides — see _resolve_mono_runs().
        kind = "code" if looks_like_code(text) else "mono"
        # Code is never translated, so its runs would never be read.
        return raw(kind, 0, text, x0, size, x1, bottom, spans=None if kind == "code" else spans)

    kind = "body"
    stripped = text.lstrip()
    if _TOC_LINE_RE.search(stripped):
        kind = "toc"
    elif _starts_with_bullet(stripped):
        kind = "bullet"
    elif _number_marker(stripped):
        # The same test the marker extraction uses, not a single-digit look at
        # the second character: an entry numbered "10. " read as body text is
        # folded as the previous item's continuation, and every entry after it
        # runs on into that one line.
        kind = "numbered"
    return raw(kind, 0, text, x0, size, x1, bottom, spans=spans)


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


def _font_thresholds(
    pages: list[Any],
    lines_by_page: list[list[dict[str, Any]]] | None = None,
) -> tuple[float, float, float]:
    """``(heading_min, h1_min, body)`` from the document's own font sizes.

    The body size is the size most lines are set in; a heading is a couple of
    points larger. Deriving it beats a constant: a book may set headings at
    15 pt while this specification uses 14 pt over a 10 pt body, so any single
    threshold either loses one document's headings or invents the other's. The
    body size is read for the same reason — it is what a figure's labels are
    judged smaller *than* (see :func:`_mark_labels`).

    Returns the fallback constants when the document is too small to have a
    meaningful distribution (short documents, unit-test stubs); ``body`` is then
    ``0.0``, which tells the caller to measure each page on its own.
    """
    sizes: Counter[float] = Counter()
    for index, page in enumerate(pages):
        lines = lines_by_page[index] if lines_by_page is not None else _text_lines(page)
        for line in lines:
            chars = line.get("chars") or []
            if not chars:
                continue
            sizes[round(max(float(c["size"]) for c in chars), 1)] += 1
    if sum(sizes.values()) < _MIN_SIZE_SAMPLE or len(sizes) < 2:
        return _HEADING_SIZE, _H1_SIZE, 0.0
    body = sizes.most_common(1)[0][0]
    heading_min = body + _HEADING_STEP
    return heading_min, max(body * 1.9, heading_min + 4.0), body


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
        if line.kind not in {"table", "header", "footer"}
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


def _folio_offset(
    pages: list[Any],
    lines_by_page: list[list[dict[str, Any]]] | None = None,
) -> int | None:
    """How far the printed folio runs behind the PDF's own page index.

    A running head carries *this page's* folio, so telling furniture from content
    means knowing which number that is. Printed page 78 may be the PDF's 92nd
    page — front matter is unnumbered — and the offset is fixed for the whole
    document: it is read off the pages that print a bare folio in their header or
    foot band, and the value most pages agree on wins.

    ``None`` when too few pages print one (see :data:`_MIN_FOLIO_SAMPLE`): the
    caller then keeps the old, more forgiving reading of an edge number.
    """
    if lines_by_page is None:
        lines_by_page = [_text_lines(page) for page in pages]
    offsets: Counter[int] = Counter()
    for index, (page, lines) in enumerate(zip(pages, lines_by_page, strict=True)):
        head, foot = _header_band(page), _footer_band(page)
        for line in lines:
            value = _page_num_only(normalize_glyphs(line_text(line.get("chars") or [])))
            if value is None:
                continue
            top = float(line["top"])
            if top < head or top >= foot:
                offsets[value - (index + 1)] += 1
    if not offsets:
        return None
    offset, seen = offsets.most_common(1)[0]
    return offset if seen >= _MIN_FOLIO_SAMPLE else None


def _page_lines(
    page: Any,
    page_no: int,
    *,
    raw: list[dict[str, Any]] | None = None,
    folio: int | None = None,
    heading_min: float = _HEADING_SIZE,
    h1_min: float = _H1_SIZE,
) -> list[_RawLine]:
    """Classified lines of one page, in reading order, tables included.

    Lines that fall inside a detected table are replaced by the table itself, so
    a row is emitted once, as a row, instead of being flattened into prose.
    *raw* is the page's pdfplumber lines when the caller already read them (the
    thresholds and the folio offset need the same list); *folio* is this page's
    own printed page number, when the document shows its pagination.
    """
    header_band = _header_band(page)
    footer_band = _footer_band(page)
    lines: list[_RawLine] = []
    for line in _text_lines(page) if raw is None else raw:
        classified = _classify(
            page_no,
            line,
            header_band=header_band,
            footer_band=footer_band,
            heading_min=heading_min,
            h1_min=h1_min,
            folio=folio,
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
    uniform leading (15.2 pt here) and paragraph skips are clearly larger.

    The yardstick is the gap that *repeats* — a page of prose measures its
    leading over and over, so the value the most lines share is that leading.
    The lower quartile was the old answer, and a scatter of gaps that happen to
    be small (3.1, 6.4, 8.2 pt on one page of the book this was written against)
    dragged it *below* the real leading; since a break is "more than 1.35 x the
    yardstick", every line of every real paragraph then read as a fresh one.

    Only the lines set in the page's own body size are asked, because a figure's
    labels are set smaller and tighter, and on a diagram page they can outnumber
    the prose outright: page 11 gives eleven 8.45 pt gaps between 7.2 pt labels
    against twelve 11.96 pt gaps between 10 pt prose lines. Counting every gap
    hands back the labels' rhythm there; the prose's own is what the paragraphs
    are measured against, so the body size decides which gaps are asked.
    """
    content = [ln for ln in lines if ln.kind not in {"header", "table"}]
    if not content:
        return 0.0
    # The size most of the page is set in: its body text, not a figure's labels.
    body_size = Counter(round(ln.size, 1) for ln in content).most_common(1)[0][0]
    deltas = _gap_deltas(content, body_size)
    if len(deltas) < 2:
        # Too few body-sized lines to trust (a title page, a plate of figures):
        # every gap there is, which is what the page has to offer.
        deltas = _gap_deltas(content, None)
    if not deltas:
        return 0.0
    runs = _gap_runs(deltas)
    # The tallest run wins, and the smaller value breaks a tie: a page of
    # paragraphs can repeat its paragraph gap as often as its leading, and the
    # yardstick has to stay on the leading's side of the break.
    repeated = max(runs, key=lambda run: (len(run), -statistics.median(run)))
    return statistics.median(repeated)


def _gap_deltas(content: list[_RawLine], size: float | None) -> list[float]:
    """Sorted gaps between consecutive lines, optionally only *size* ones.

    Both ends of a gap have to be set at *size*, so a gap to a heading or a
    caption is not counted as part of the body rhythm. ``None`` asks for every
    gap: the fallback for a page with too few body-sized lines to measure.
    """
    return sorted(
        b.top - a.top
        for a, b in itertools.pairwise(content)
        if 0 < b.top - a.top < 60
        and (size is None or (round(a.size, 1) == size and round(b.size, 1) == size))
    )


def _gap_runs(deltas: list[float]) -> list[list[float]]:
    """Split sorted *deltas* into runs of near-equal values, smallest first.

    A run is chained from its own first value, so it cannot drift arbitrarily
    wide: 12.0, 12.9 and 13.8 are two rhythms, not one.
    """
    runs: list[list[float]] = [[deltas[0]]]
    for delta in deltas[1:]:
        if delta - runs[-1][0] <= _GAP_TOLERANCE:
            runs[-1].append(delta)
        else:
            runs.append([delta])
    return runs


def _mark_labels(classified: list[tuple[int, _RawLine]], body_size: float) -> None:
    """Flag every line that is a figure's own label rather than prose.

    Labels are not prose. Their gaps are the figure's own spacing, so a run of
    them reads as one paragraph that owns a scatter of rectangles — and the
    rebuilt PDF then prints one translation across the whole scatter (see
    :func:`bind_page_geometry`). Kept as a flag rather than a kind, a label is
    still translated and printed as the body line it is; only the block boundary
    changes, and :func:`_is_new_block` reads the flag.

    The page's text block is the widest line on it, so what counts as *narrow*
    is this page's own measure rather than a number of points. And a line that
    reads on from the one above it is a *continuation*, never a label: this is
    what keeps a note set in the same small type whole — its last line is short
    and narrow like a label, and only the fact that its sentence started on the
    line above tells the two apart. A note whose wrapped line stops at a URL or
    a proper name ends mid-sentence without ending a *sentence*, so the line
    that follows it is a continuation just as plainly, whichever letter it
    happens to open with.
    """
    for _page_no, group in itertools.groupby(classified, key=lambda item: item[0]):
        lines = [ln for _, ln in group]
        column = max((ln.x1 - ln.x0 for ln in lines if ln.kind != "table"), default=0.0)
        page_body = body_size or _dominant_size(lines)
        if column <= 0 or page_body <= 0:
            continue
        previous = ""
        wrapped = False  # the line above is a wide small line left mid-sentence
        for ln in lines:
            small = ln.kind in {"body", "bullet", "numbered"} and (
                page_body - ln.size >= _LABEL_SIZE_STEP
            )
            narrow = ln.x1 - ln.x0 <= column * _LABEL_WIDTH_SHARE
            ln.label = small and narrow and not (wrapped or _reads_on(previous, ln.text))
            previous = ln.text.rstrip()
            wrapped = bool(previous) and small and not narrow and previous[-1] not in _SENTENCE_END


def _dominant_size(lines: list[_RawLine]) -> float:
    """The size most of *lines* are set in: the page's own body text."""
    sizes = Counter(round(ln.size, 1) for ln in lines if ln.kind != "table")
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _is_new_block(prev: _RawLine | None, cur: _RawLine, spacing: float) -> bool:
    """True when *cur* starts a fresh paragraph/block rather than continuing."""
    if prev is None:
        return True
    if cur.label or prev.label:
        # A figure's label is a block of its own, at both ends: it must not be
        # swept into the paragraph above it, and prose must not be swept into it.
        return True
    if prev.kind != cur.kind and {prev.kind, cur.kind} & {"heading", "code"}:
        return True
    if prev.kind in {"bullet", "numbered"} and cur.kind in {"bullet", "numbered"}:
        return True
    if prev.kind == "toc" or cur.kind == "toc":
        return True  # a table of contents is one entry per line
    if _label_chain_break(prev, cur):
        # A run of lines that each open with a bold, colon-terminated label
        # ("Project Title: ...", "Student ID: ...") is one field per line, not
        # a wrapped paragraph: a title page sets its fields at the same even
        # leading as prose wraps, so the spacing cannot tell them apart, and
        # the trailing "short line" signal cannot either (the last line of a
        # wrapped paragraph is short just as often). A leading bold label is
        # what a wrapped continuation never carries.
        return True
    if prev.page != cur.page:
        return not _reads_on(prev.text, cur.text)
    if spacing <= 0:
        return False
    return (cur.top - prev.top) > _PARAGRAPH_GAP * spacing


def _leading_label(line: _RawLine) -> str:
    """The line's leading bold label (its text without the colon), or ``''``.

    A label is a bold run that *ends* on a colon -- "Project Title:" before
    regular-weight text. A title bold from first character to last carries no
    such mark, so a heading-like line never reads as a field, and a line whose
    first run is regular reports nothing either.
    """
    for text, bold, _italic in line.spans:
        stripped = text.strip()
        if not stripped:
            continue
        if not bold:
            return ""
        if stripped == stripped.rstrip(":："):
            return ""  # bold but uncolonned: a title or term, not a field label
        return stripped.rstrip(":：")
    return ""


def _label_chain_break(prev: _RawLine, cur: _RawLine) -> bool:
    """True when *cur* opens a new field line right after another one.

    Both lines must carry a label: a lone labelled line before prose is a
    definition line whose own text may read on into the paragraph below it,
    but two labelled lines in a row are two fields -- a wrapped continuation
    never opens with a bold label of its own.
    """
    return bool(_leading_label(prev)) and bool(_leading_label(cur))


def _reads_on(prev_text: str, cur_text: str) -> bool:
    """True when *cur_text* reads as the rest of *prev_text*'s paragraph.

    Asked only where the geometry cannot answer — across a page break — and on
    the evidence a reader uses: the line before it did not end a sentence, and
    the new line continues that sentence instead of opening a new one (a
    capital), a heading or a list item.
    """
    body = (prev_text or "").rstrip()
    if not body or body[-1] in _SENTENCE_END:
        return False
    head = (cur_text or "").lstrip()[:1]
    return bool(head) and head.islower()


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
    #: The page's own pdfplumber lines, read once: the size thresholds, the
    #: document's folio offset and the classification all ask the same list.
    lines_by_page = [_text_lines(page) for page in pages]
    heading_min, h1_min, body_size = _font_thresholds(pages, lines_by_page)
    #: Front matter makes the printed page number differ from the PDF's own page
    #: index; the running-head test needs the number this page actually prints.
    offset = _folio_offset(pages, lines_by_page)
    classified = [
        (page_no, ln)
        for page_no, page in enumerate(pages)
        for ln in _page_lines(
            page,
            page_no,
            raw=lines_by_page[page_no],
            folio=None if offset is None else offset + page_no + 1,
            heading_min=heading_min,
            h1_min=h1_min,
        )
    ]
    _mark_labels(classified, body_size)
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
    #: (level, page) of the heading emitted last, for the two-line title merge
    #: below. The page is part of it: a part title at the top of one page and a
    #: chapter title at the top of the next are two headings, not one title
    #: wrapped over two lines — and with the page's folio no longer sitting
    #: between them, nothing else tells them apart.
    last_heading: tuple[int, int] | None = None

    def flush() -> None:
        nonlocal pending, pending_kind
        if not pending:
            return
        lines = [line for line, _, _ in pending]
        parts = [list(line.spans) for line in lines]
        # Markdown needs its marker back for the parser to see a list, and the
        # glyph was redacted with the item, so `_marked_text` prints it again.
        # Here the source's own bullet is stripped **once**, as a run of its own,
        # so a bold item does not drag "- " inside its token pair.
        if pending_kind == "list" and lines and lines[0].kind == "bullet" and parts and parts[0]:
            head = parts[0][0]
            stripped = _bullet_text(head[0])
            prefix: list[tuple[str, bool, bool]] = [("- ", False, False)]
            if stripped:
                prefix.append((stripped, head[1], head[2]))
            parts[0] = prefix + parts[0][1:]
        text = mark_emphasis(_join_wrapped_runs(parts))
        if text:
            unit = _Unit(pending_kind, text)
            unit.take(lines)
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
        if ln.kind in {"header", "footer"}:
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
                and last_heading[1] == page_no
                and units
                and units[-1].kind == "heading"
            ):
                # "Chapter 3" / "ArrayList" are two oversized lines of one title.
                units[-1].text += " " + ln.text
                units[-1].take([ln])
            else:
                emit("heading", "#" * ln.level + " " + ln.text, level=ln.level).take([ln])
            last_heading = (ln.level, page_no)
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

#: The key every code block compares equal to. Code is never translated, and a
#: placeholder's number is local to one parse, so two code blocks are told apart
#: by their order, never by their text.
_CODE_KEY = "\x00code"

#: How far ahead of the cursor a block's own units may lie. A unit the Markdown
#: parser folds into a neighbouring block (a numbered line cannot open a list in
#: the middle of a paragraph, so it becomes that paragraph's next line) is
#: skipped, and the block after it still owns the units that follow.
_BIND_WINDOW = 24

#: Most units one block can own. A list block owns one unit per item — the one
#: place where a block is much longer than a unit.
_BIND_MAX_SPAN = 64

#: Whitespace and table pipes carry no identity: the Markdown parse re-wraps
#: lines and rebuilds the matrix, so both sides are compared without them.
_KEY_SPACE_RE = re.compile(r"\s+")


def _plain_key(text: str) -> str:
    """The comparable form of *text*: no whitespace, no pipes, no case."""
    return _KEY_SPACE_RE.sub("", (text or "").replace("|", "")).lower()


def _parsed_key(block: Any) -> str:
    """The key of a *parsed* block, on either side of the comparison.

    Both the units and the document's blocks go through the same Markdown
    parse, so equal keys mean the same source text — not the same raw bytes.
    """
    if block.kind == "code" or _CODE_PLACEHOLDER_RE.match(block.text.strip()):
        return _CODE_KEY
    if block.kind == "list":
        return _plain_key(" ".join(block.meta.get("items") or [block.text]))
    return _plain_key(block.text)


@lru_cache(maxsize=4096)
def _unit_key(text: str, kind: str) -> str:
    """What one unit contributes to the block it is part of.

    The unit's own text is put through the parse the whole document went
    through, so an emphasised run, an escaped character or a line markdown-it
    reads as HTML cannot make the two sides look different: the comparison is
    between two parse results rather than between a parse result and raw text.
    """
    if kind == "code":
        return _CODE_KEY
    parsed = parse_markdown(normalize_toc_lines(text), source_format="pdf")
    return "".join(_parsed_key(block) for block in parsed.blocks)


def _units_for_block(
    units: list[_Unit], keys: list[str], cursor: int, block: Any
) -> tuple[int, int] | None:
    """``(start, span)`` of the units that produced *block*, or ``None``.

    The match is *verified*: the units' own keys must concatenate to the
    block's key exactly, so a block is never bound to geometry that belongs to a
    different one. Finding nothing answers ``None`` — the block then keeps its
    source text — and, because the cursor only ever moves over units that were
    verified, that answer cannot drag the blocks after it onto other lines.
    """
    target = _parsed_key(block)
    if not target:
        return None
    limit = len(target) + 1
    for start in range(cursor, min(cursor + _BIND_WINDOW, len(units))):
        accum = ""
        for span in range(1, _BIND_MAX_SPAN + 1):
            end = start + span
            if end > len(units):
                break
            accum += keys[end - 1]
            if accum == target:
                return start, span
            if len(accum) > limit:
                break
    return None


def bind_page_geometry(doc: Document, units: list[_Unit]) -> int:
    """Attach each block's source rectangles as ``block.meta["pdf"]``.

    Every block is matched to the units it was *parsed from*, by text (see
    :func:`_units_for_block`), instead of walking both lists in step. The units
    come from the layout extractor and the blocks from the Markdown parser, so
    one disagreement about where a unit ends and the next begins used to shift
    every later block's geometry — and a block of prose bound to one line of a
    page six pages on is a page of tiny type printed over the wrong text.

    A block that cannot be verified is not rebuilt at all (its source text
    stays), which is the honest outcome: a mis-bound rectangle would print the
    translation over the wrong lines. Returns how many blocks were bound.
    """
    keys = [_unit_key(unit.text, unit.kind) for unit in units]
    cursor = 0
    bound = 0
    for block in doc.blocks:
        found = _units_for_block(units, keys, cursor, block)
        if found is None:
            logger.debug("PDF geometry: no unit for block {!r}", block.text[:40])
            continue
        index, span = found
        if index > cursor:
            # A unit the Markdown parser folded into a neighbouring block is
            # skipped, not mis-assigned, so the blocks after it stay aligned.
            logger.debug(
                "PDF geometry: skipped {} unmatched unit(s) before {!r}",
                index - cursor,
                block.text[:40],
            )
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
        cursor = index + span
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
