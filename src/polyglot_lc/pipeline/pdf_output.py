"""Rebuild the source PDF with the translated text, keeping its layout.

The source PDF supplies everything that is *not* language: page size, images,
vector graphics, table rules and the position, size and *typography* of every
text block. This module replaces only the blocks the pipeline translated — in
place, in the same rectangle, at the same size, in the same family and weight —
with a Unicode font embedded for Vietnamese. When the source face is installed
locally the translation is printed in that very face; otherwise the nearest
family of the same kind (serif/sans/mono) stands in, so a rebuilt page still
looks like the page it replaces. Two things the source expressed through
*geometry* are rebuilt rather than translated, because they are layout:

* a **list marker** was redacted with its item, so it is printed again;
* a **contents entry**'s dot leader is measured back in between the translated
  title and its page number, which stays against the right margin.

A block is printed at the size the source set it in, and the part of a longer
translation that does not fit its box continues on a new page inserted after it,
at the same size and in the same column. Shrinking the type to squeeze a
Vietnamese sentence into the English line's box is what a rebuilt page cannot
afford: at the 55% floor it printed 10 pt body text at 5.5 pt, over the block
below. A table cell is the one exception, because what surrounds it *is* the
layout.

A **table** is rebuilt cell by cell, inside the grid the source page already
drew: the rules stay exactly where they were and each translated cell is printed
into its own box. Two rules keep that honest:

* a cell's type may be shrunk to 70% of its source size (``_CELL_MIN_SCALE``)
  but no further, and never outside its box — unlike a block, a cell cannot
  continue overleaf, because what surrounds it is the layout;
* when the translation still will not fit, the cell prints its own source text
  at the source size instead of dropping the cell or bleeding into its
  neighbour.

The same reasoning applies to a table whose translation cannot be trusted to
line up with the source grid (wrong shape, a stray ``|``, a bare ``|`` split):
the whole table stays in the source language rather than having its cells
aligned against the wrong boxes.

What is deliberately left alone (and why):

* **code** — never translated by the pipeline (it is shielded as
  ``⟦CODE_n⟧``), so the source text stays.
* **anything without geometry or translation** — a chunk whose retries were
  exhausted leaves its source lines in place rather than blanking the page, so
  a partial run still yields a readable document.

**Emphasis** is the one piece of a block's typography a translation may *add*.
The ingest marks a bold or italic span with a sentinel pair (``⟦BOLD⟧…⟦/BOLD⟧``;
see :mod:`polyglot_lc.emphasis`), and a block whose translation keeps those pairs
is laid out by hand here — greedy word wrap measured run by run, one
``insert_text`` per run in that run's own face — because PyMuPDF's own layout
does not read markup and would print the sentinels as they stand. A pair that
does not match the source is dropped upstream (``settle_emphasis`` in
:mod:`polyglot_lc.orchestrator.graph`), and a face this machine does not have
falls back to the regular one, so no sentinel ever reaches the page: the words
are printed, at worst without their emphasis.

The boxes come from :mod:`polyglot_lc.ingest.pdf_layout`, which records a block's
source rectangles as it extracts it; see ``block.meta["pdf"]``.

One artefact of the rebuild is worth naming. PyMuPDF builds an inserted font's
``ToUnicode`` map by inverting the font's ``cmap``, and where a ``cmap`` gives
one glyph two code points — the space and the no-break space, the hyphen and the
soft hyphen — the higher one wins. Text copied out of a rebuilt page would then
carry U+00A0 where the page shows a space and U+00AD where it shows a hyphen.
The maps of the fonts *this module inserts* are repaired back to the ASCII code
points (:func:`_repair_tounicode`); the source document's own maps are left
alone, so a block kept in the source language copies out exactly as before. The
glyph is shared, so a genuine no-break space in a translation cannot survive
this: it reads as a space, which is the better of the two defaults for prose.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from loguru import logger

from polyglot_lc import emphasis
from polyglot_lc.ingest.model import Document
from polyglot_lc.pipeline.assembler import block_translations

#: Font resource name used for the embedded Unicode font inside every page.
_FONT_NAME = "pgf"

#: Directories searched for a face that matches the source document's.
_FONT_DIRS: tuple[str, ...] = (
    "C:/Windows/Fonts",
    os.path.expandvars("%WINDIR%/Fonts"),
    os.path.expandvars("%LOCALAPPDATA%/Microsoft/Windows/Fonts"),
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.local/share/fonts"),
    os.path.expanduser("~/Library/Fonts"),
    "/Library/Fonts",
    "/System/Library/Fonts",
)

#: Words in a font's name that describe its weight or slant, not its family.
_STYLE_WORDS = ("semibold", "bold", "italic", "oblique", "regular")

#: Families tried, in order, when the source's own family is not installed.
_PREFERRED_FAMILIES: dict[str, tuple[str, ...]] = {
    "sans": (
        "arial",
        "helvetica",
        "liberationsans",
        "dejavusans",
        "notosans",
        "segoeui",
        "tahoma",
        "verdana",
        "calibri",
        "freesans",
    ),
    "serif": (
        "timesnewroman",
        "liberationserif",
        "dejavuserif",
        "notoserif",
        "georgia",
        "freeserif",
        "cambria",
    ),
    "mono": (
        "consolas",
        "couriernew",
        "liberationmono",
        "dejavusansmono",
        "notosansmono",
        "dejavumono",
        "menlo",
    ),
}

_SERIF_MARKERS = (
    "serif",
    "times",
    "georgia",
    "garamond",
    "antiqua",
    "palatino",
    "cambria",
    "century",
)
_MONO_MARKERS = ("mono", "courier", "consol", "menlo", "terminal", "fixed")

#: Glyphs every candidate font must carry: Vietnamese needs Latin Extended
#: Additional, which most PDF base faces and several system fonts lack.
_VIETNAMESE_PROBE = "Đđẹệợứựảữỹỵăâêôơư"

#: The separator :mod:`polyglot_lc.ingest.pdf_layout` leaves in place of a
#: contents entry's dot leader, plus the plain dots a translation may use.
_TOC_ENTRY_RE = re.compile(r"^\s*(?P<title>.*?)\s*(?:…|\.{3,})\s*(?P<number>[\w.\-]{1,8})\s*$")

#: A bullet pdfplumber could not map to text: there is no glyph to reprint.
_CID_RE = re.compile(r"^\(cid:\d+\)$")

#: Printed in place of an unmapped bullet glyph.
_FALLBACK_BULLET = "•"

#: A shielded code block: the pipeline never translates these.
_CODE_PLACEHOLDER_RE = re.compile(r"^⟦CODE_\d+⟧$")

_HEADING_RE = re.compile(r"^#{1,6}\s+")
_LIST_ITEM_RE = re.compile(r"^(?:[-*•]|\d{1,3}[.)])\s+")

#: Fonts that do cover Vietnamese (they carry Latin Extended Additional), in the
#: order we would rather use them. Times/Arial first because a rebuilt book page
#: should look like the page it replaces, not like a DejaVu specimen.
_FONT_CANDIDATES: tuple[str, ...] = (
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
)

#: Room, in points, kept between a rebuilt block and the page's bottom edge.
_BOTTOM_MARGIN = 28.0

#: Room, in points, kept above a continued block on the page it continues on.
#: The source page's top margin is not recorded per block, and every page of a
#: book open with roughly this much white above the first line.
_TOP_MARGIN = 28.0

#: The same limit for a table cell, and a tighter one: a cell is surrounded by
#: the rules of its grid, so shrinking it is the only way it can grow, and text
#: much smaller than the source's would read as a different document's table.
_CELL_MIN_SCALE = 0.70

#: Two code points that share one glyph in most text faces, and whose ToUnicode
#: map PyMuPDF therefore gets wrong: the ASCII code point is what a reader's
#: search box expects, so the no-break space and the soft hyphen are rewritten
#: to it. See the module docstring.
_TOUNICODE_SHARED_GLYPHS: dict[str, str] = {"00a0": "0020", "00ad": "002d"}

#: One standalone ``beginbfchar`` entry whose target is such a code point.
#: Anchored to a whole line and never crossing a line break: a ``\s*`` between
#: the two codes swallows the newline before an entry whose *source* happens to
#: be ``00a0``/``00ad``, fusing two entries into one malformed line. The saved
#: map then parses as garbage — every later entry shifts — and correct input
#: reads back wrong (``JavaScript`` out as ``ÃavaScript``). A bfrange head
#: (``<lo> <hi> <dst>``) is three tokens on one line, so it never matches.
_TOUNICODE_ENTRY_RE = re.compile(
    r"^[ \t]*<([0-9A-Fa-f]{4})>[ \t]+<(00a0|00ad)>[ \t]*$", re.MULTILINE
)


class PdfOutputError(RuntimeError):
    """The PDF could not be rebuilt (usually: no Unicode font is available)."""


@dataclass
class RebuildStats:
    """What the rebuild changed, for the run report."""

    pages: int = 0
    placed: int = 0
    kept: int = 0
    #: Blocks whose words went into the translation printed for the block above
    #: them (a title the source wrapped over two lines, translated as one
    #: paragraph): their own lines are cleared and nothing is printed in their
    #: place, because that paragraph already carries their words.
    merged: int = 0
    #: Pages added for blocks whose translation did not fit the box the source
    #: set it in. Their text is printed there at the source size, never shrunk
    #: into a neighbour's lines.
    continued: int = 0
    #: Table cells printed with their translation, and cells that fell back to
    #: their own source text because the translation would not fit the box.
    cells: int = 0
    cells_kept: int = 0
    #: Inserted fonts whose copied-text map was repaired (see
    #: :func:`_repair_tounicode`).
    repaired_maps: int = 0
    font: str = ""
    failures: list[str] = field(default_factory=list)


def resolve_font(explicit: str | Path | None = None) -> Path:
    """The Unicode font file used to print the translation.

    Vietnamese needs Latin Extended Additional glyphs, which the PDF base-14
    fonts do not have, so a real TTF has to be embedded. *explicit* (or
    ``PDF_FONT_PATH``) always wins; otherwise the first readable candidate is
    used.
    """
    wanted = str(explicit or os.getenv("PDF_FONT_PATH", "")).strip()
    candidates = ([wanted] if wanted else []) + list(_FONT_CANDIDATES)
    for candidate in candidates:
        path = Path(candidate)
        if candidate and path.is_file():
            return path
    raise PdfOutputError(
        "no Unicode font found for the translated PDF; set PDF_FONT_PATH to a "
        ".ttf with Vietnamese coverage (e.g. DejaVuSans.ttf)"
    )


def _font_key(name: str) -> str:
    """A family key: a face's name without its weight, slant or subset tag.

    ``Verdana-Bold``, ``Verdana`` and ``verdana regular`` are one family, so a
    document that uses any of them finds the local ``verdana.ttf``.
    """
    lowered = re.sub(r"(?:psmt|mt)$", "", str(name or "").strip().lower())
    for word in _STYLE_WORDS:
        lowered = lowered.replace(word, "")
    return re.sub(r"[^a-z0-9]", "", lowered)


def _style_of(name: str) -> tuple[bool, bool]:
    """``(bold, italic)`` read off a font's name."""
    lowered = str(name or "").lower()
    return "bold" in lowered, ("italic" in lowered or "oblique" in lowered)


def _kind_of(key: str) -> str:
    """``"mono"``, ``"serif"`` or ``"sans"`` for a family key."""
    if any(marker in key for marker in _MONO_MARKERS):
        return "mono"
    if any(marker in key for marker in _SERIF_MARKERS):
        return "serif"
    return "sans"


@lru_cache(maxsize=512)
def _covers_vietnamese(font_file: str) -> bool:
    """True when *font_file* can print Vietnamese."""
    import pymupdf

    try:
        font = pymupdf.Font(fontfile=font_file)
    except Exception:  # noqa: BLE001 - an unreadable font is simply not a match
        return False
    return all(font.has_glyph(ord(ch)) for ch in _VIETNAMESE_PROBE)


@lru_cache(maxsize=1)
def installed_fonts() -> dict[str, dict[tuple[bool, bool], str]]:
    """The font files on this machine, by family key and weight/slant.

    Read once per process: naming a face (``C:/Windows/Fonts/verdana.ttf``) is
    the only reliable way to match the source document's typography across
    platforms, and the family key is what survives the PDF's subset prefixes
    and style suffixes.
    """
    import pymupdf

    found: dict[str, dict[tuple[bool, bool], str]] = {}
    seen: set[Path] = set()
    for directory in _FONT_DIRS:
        try:
            root = Path(directory).resolve()
        except OSError:  # pragma: no cover - a path we cannot resolve
            continue
        # The same directory is named twice on Windows ("C:/Windows/Fonts" and
        # %WINDIR%): reading it twice would double the cost for nothing.
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in (".ttf", ".otf"):
                continue
            try:
                name = pymupdf.Font(fontfile=str(path)).name
            except Exception as exc:  # noqa: BLE001 - a font we cannot read is
                # simply not a candidate; the fallback face is still embedded.
                logger.debug("PDF rebuild: skipping font {} ({})", path, exc)
                continue
            key = _font_key(name)
            if key:
                found.setdefault(key, {}).setdefault(_style_of(name), str(path))
    logger.debug("PDF rebuild: {} font families installed", len(found))
    return found


def _styled_face(faces: dict[tuple[bool, bool], str], style: tuple[bool, bool]) -> str | None:
    """The closest usable face of one family: exact style, then plainer ones."""
    bold, italic = style
    for wanted in ((bold, italic), (bold, False), (False, italic), (False, False)):
        path = faces.get(wanted)
        if path and _covers_vietnamese(path):
            return path
    return None


@lru_cache(maxsize=512)
def match_font(name: str, bold: bool, italic: bool) -> str | None:
    """A local file for the source face, or ``None`` when nothing matches.

    The document's own family first, then any family of the same kind: printing
    a translated page in the face the page was set in is the difference between
    a rebuilt document and a rough approximation of one.
    """
    key = _font_key(name)
    if not key:
        return None
    families = installed_fonts()
    own = families.get(key)
    if own:
        found = _styled_face(own, (bold, italic))
        if found:
            return found
    for family in _PREFERRED_FAMILIES.get(_kind_of(key), ()):
        faces = families.get(family)
        if not faces:
            continue
        found = _styled_face(faces, (bold, italic))
        if found:
            return found
    return None


def plain_text(markdown_text: str, kind: str) -> str:
    """The text to print on the page: the translation without Markdown syntax.

    Emphasis tokens are Markdown syntax too, so they are dropped here as well:
    a token that reaches this function has no run to be printed with (a list, a
    table cell, a block the model invented tokens for) and must not show up.
    """
    lines = markdown_text.splitlines() or [markdown_text]
    out: list[str] = []
    for line in lines:
        stripped = _HEADING_RE.sub("", line) if kind == "heading" else line
        if kind == "list":
            stripped = _LIST_ITEM_RE.sub("", stripped)
        out.append(emphasis.strip(stripped).strip())
    return "\n".join(line for line in out if line).strip()


#: Kinds whose translation may be printed run by run. A list item still needs
#: its source bullet re-inserted by :func:`_marked_text` and a contents entry
#: its rebuilt dot leader, and neither happens on the run path — so both keep
#: one face, as they did before.
_RUN_KINDS: frozenset[str] = frozenset({"body", "heading"})


def _has_tokens(text: str) -> bool:
    """True when *text* carries at least one emphasis token."""
    return any(token in text for token in emphasis.TOKENS)


def _runs_without_heading_marker(
    runs: list[tuple[str, bool, bool]],
) -> list[tuple[str, bool, bool]]:
    """Drop a heading's leading ``#``/``## `` marker from its first run.

    ``plain_text`` strips that marker; the run path prints the runs themselves,
    so it has to be stripped here or a translated title would print its hashes.
    """
    if not runs:
        return runs
    text, bold, italic = runs[0]
    stripped = _HEADING_RE.sub("", text)
    rest = runs[1:]
    if not stripped:
        return rest
    return [(stripped, bold, italic), *rest]


@dataclass
class _Placement:
    """One translated block: what to remove and what to print in its place."""

    page: int
    redact: list[tuple[float, float, float, float]]
    box: tuple[float, float, float, float]
    size: float
    text: str
    kind: str
    #: The box the source line actually occupied, kept for deciding alignment:
    #: a widened box no longer says where the source put its own text.
    source_box: tuple[float, float, float, float] | None = None
    #: The source face to print in; ``None`` leaves the choice to the caller.
    font_file: str | None = None
    color: tuple[float, ...] | None = None
    #: A contents entry: title and page number, to be printed as one line with
    #: a dot leader between them, the way the source page had it.
    toc: tuple[str, str] | None = None
    #: How the source aligned this text, for a table cell; ``None`` keeps the
    #: automatic choice (left, and centred for a centred heading).
    align: str | None = None
    #: A cell's own source text, printed in its place when the translation does
    #: not fit the box the source fitted in.
    source_text: str | None = None
    #: The source family name, which is what picks the bold/italic face of the
    #: same family (``match_font`` needs a name, not a font file path).
    face: str = ""
    #: The translation split into runs of one style, or ``None`` to print the
    #: whole block in one face with ``insert_textbox`` (the usual case).
    runs: list[tuple[str, bool, bool]] | None = None
    #: Whether the source's family may be matched to a local face. A pinned font
    #: opts out, and the runs have to respect that too: otherwise a font the
    #: caller chose on purpose would come back for the bold names only.
    match_style: bool = True


#: How far a one-line block may be stretched, in points, from the text column.
_CENTER_TOLERANCE = 18.0


def _page_geometry(
    doc: Document,
) -> tuple[dict[int, tuple[float, float]], dict[int, list[tuple[float, float, float, float]]]]:
    """Each page's text column, and everything already printed on it.

    A one-line source block is set in a box that hugs the *English* line, and a
    translation of it is longer far more often than it is shorter. Stretching a
    line into the column it already sits in is what keeps the source's type size
    instead of shrinking the text to fit the old line — but only where the room
    is genuinely empty.
    """
    columns: dict[int, tuple[float, float]] = {}
    occupied: dict[int, list[tuple[float, float, float, float]]] = {}
    for block in doc.blocks:
        geo = block.meta.get("pdf")
        if not isinstance(geo, dict):
            continue
        for page, x0, top, x1, bottom in geo.get("rects") or []:
            lo, hi = columns.get(int(page), (x0, x1))
            columns[int(page)] = (min(lo, x0), max(hi, x1))
            occupied.setdefault(int(page), []).append((x0, top, x1, bottom))
    return columns, occupied


#: How much white a grown box leaves above whatever stands below it.
_BLOCK_CLEARANCE = 2.0

#: How many characters of the source size a box must be able to hold before the
#: translation is printed in it — a backstop, two characters wide. A block whose
#: rectangle is narrower than this cannot be set a sentence on (PyMuPDF wraps it
#: one or two letters per line, into pages of rubble), so it keeps its source
#: text instead. The extractor no longer produces such boxes for a running foot
#: (see ``pdf_layout._footer_band``); this is what keeps the *next* one from
#: reaching the page.
_MIN_BOX_WIDTH_FACTOR = 2.0


def _grown_down(
    box: tuple[float, float, float, float],
    occupied: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """A box allowed to reach down through the empty room under its own lines.

    A translation is a little longer than the English the box was cut for more
    often than not, and the source page has white between its blocks. Growing
    into that white keeps the paragraph where the source put it, at the source's
    type size: the alternative is a last line pushed onto a page of its own, or
    — what this module used to do — the whole block shrunk to fit the English
    line's box. The box stops at whatever is printed below it, so nothing is
    printed over: the caller clamps the bottom to the page's own margin.
    """
    x0, top, x1, bottom = box
    below = [
        other
        for other in occupied
        if other[1] >= bottom - 1 and other[0] < x1 - 1 and other[2] > x0 + 1
    ]
    if not below:
        return box
    limit = min(other[1] for other in below) - _BLOCK_CLEARANCE
    return (x0, top, x1, max(bottom, limit))


def _stretched_box(
    box: tuple[float, float, float, float],
    occupied: list[tuple[float, float, float, float]],
    own: list[tuple[float, float, float, float]],
    column: tuple[float, float] | None,
) -> tuple[float, float, float, float]:
    """A one-line box widened to its column, when the room is actually free."""
    x0, top, x1, bottom = box
    if column is None or len(own) != 1:
        return box
    lo, hi = column
    if hi <= x1 + 1:
        return box
    for px0, ptop, px1, pbottom in occupied:
        if (px0, ptop, px1, pbottom) in own:
            continue
        # Something else already sits on this line, to the right of the block.
        if ptop < bottom - 1 and pbottom > top + 1 and px0 < hi - 1 and px1 > x1 + 1:
            return box
    center, column_center = (x0 + x1) / 2, (lo + hi) / 2
    if abs(center - column_center) < _CENTER_TOLERANCE:
        return (lo, top, hi, bottom)  # a centred line stays centred
    return (x0, top, hi, bottom)


def _placement(
    geo: dict[str, Any],
    text: str,
    kind: str,
    toc: Any = None,
    column: tuple[float, float] | None = None,
    occupied: list[tuple[float, float, float, float]] | None = None,
) -> _Placement | None:
    """Turn a block's geometry + translation into a placement, or ``None``."""
    rects = geo.get("rects") or []
    if not rects:
        return None
    page = int(rects[0][0])
    on_page = [r for r in rects if int(r[0]) == page]
    x0 = min(r[1] for r in on_page)
    x1 = max(r[3] for r in on_page)
    top = min(r[2] for r in on_page)
    bottom = max(r[4] for r in on_page)
    size = max(float(geo.get("size") or 0.0), 6.0)
    # A little room under the last line: the translation is rarely the same
    # length, and a box that ends exactly on the baseline clips the last row.
    box = (x0, top, x1, bottom + max(size * 0.6, 4.0))
    box = _stretched_box(
        box,
        occupied or [],
        [(r[1], r[2], r[3], r[4]) for r in on_page],
        column,
    )
    box = _grown_down(box, occupied or [])
    color = tuple(float(part) for part in geo.get("color") or ()) or None
    return _Placement(
        page=page,
        redact=[(r[1], r[2], r[3], r[4]) for r in rects],
        box=box,
        source_box=(x0, top, x1, bottom + max(size * 0.6, 4.0)),
        size=size,
        text=text,
        kind=kind,
        color=color,
        toc=toc,
    )


#: A Markdown table row, and a separator row's cell (``---``, ``:---:``).
_MD_ROW_START = "|"
_MD_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")


def _markdown_cells(text: str) -> list[list[str]] | None:
    """A Markdown pipe table as a cell matrix, or ``None`` when it is not one.

    The translation of a table block is the *source* table rewritten cell by
    cell, so its shape is the only thing the rebuild has to trust: one row per
    row, one cell per cell. Anything else — prose, a dropped column, a stray
    ``|`` inside a cell — is refused, and the whole source table stays on the
    page rather than having its cells lined up against the wrong boxes.
    """
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(_MD_ROW_START):
            continue
        cells = [" ".join(cell.split()) for cell in stripped.strip("|").split("|")]
        if cells and all(_MD_SEPARATOR_RE.match(cell) for cell in cells):
            continue
        rows.append(cells)
    if len(rows) < 2 or len(rows[0]) < 2:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        return None
    return rows


def _sliver_box(geo: dict[str, Any]) -> bool:
    """True when a block's box is too narrow to set its translation on.

    A running foot glued to the next page's running head has a box the width of
    the page number. Printing there is not a layout the reader asked for; keeping
    the source's own characters is (see :data:`_MIN_BOX_WIDTH_FACTOR`).
    """
    rects = geo.get("rects") or []
    if not rects:
        return True
    page = int(rects[0][0])
    on_page = [r for r in rects if int(r[0]) == page]
    if not on_page:
        return True
    width = max(r[3] for r in on_page) - min(r[1] for r in on_page)
    size = max(float(geo.get("size") or 0.0), 6.0)
    return width < size * _MIN_BOX_WIDTH_FACTOR


def _inside_any(
    box: tuple[float, float, float, float],
    others: list[tuple[float, float, float, float]],
) -> bool:
    """True when *box* is contained in, or equal to, one of *others*."""
    x0, top, x1, bottom = box
    return any(
        x0 >= other[0] - 0.01
        and x1 <= other[2] + 0.01
        and top >= other[1] - 0.01
        and bottom <= other[3] + 0.01
        for other in others
    )


#: Most room, in points, between a merged block and the box that covers it.
#: Its translation is printed *there*, so a block further down the page is not
#: part of that sentence and must not be cleared with it.
_MERGE_GAP = 1.0


def _absorb(item: _Placement | None, geo: dict[str, Any]) -> bool:
    """Give *item* the lines of a block whose words it already translates.

    A block the translator folded into the paragraph above it leaves its own
    source lines behind: they are part of a sentence that is printed in the
    item's box, so they are added to the redaction and the box grows over them —
    a wrapped title owns both of its lines, not just the first. Returns whether
    the block could be covered; when it cannot, the caller leaves it alone
    rather than clearing text that nothing was printed over.
    """
    rects = geo.get("rects") or []
    if item is None or item.kind == "cell" or not rects:
        return False
    if int(rects[0][0]) != item.page:
        return False
    boxes = [(float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rects]
    # The *source* box, not the printed one: a box grown down through the page's
    # white reaches whatever stands below it, and a sentence that was printed
    # high up the page is not the sentence of a block 120 pt further down.
    own = item.source_box or item.box
    if min(box[1] for box in boxes) > own[3] + item.size * _MERGE_GAP:
        return False
    item.redact.extend(boxes)
    x0, top, x1, bottom = item.box
    for box in boxes:
        x0, top = min(x0, box[0]), min(top, box[1])
        x1, bottom = max(x1, box[2]), max(bottom, box[3])
    item.box = (x0, top, x1, bottom)
    return True


def _source_cell(source: list[list[str]] | None, row: int, column: int) -> str | None:
    """The source text of one cell, for the fallback when a cell will not fit."""
    if source is None or row >= len(source) or column >= len(source[row]):
        return None
    return source[row][column]


def _table_placements(
    geo: dict[str, Any],
    block: Any,
    raw: str,
    *,
    match_style: bool,
    default_font: Path | str | None,
    printed: dict[int, list[tuple[float, float, float, float]]],
) -> list[_Placement] | None:
    """One placement per cell of a table block, or ``None`` to keep the source.

    ``None`` is the honest answer whenever the translation cannot be trusted to
    line up with the source grid: no cell boxes, a different shape, or nothing
    printable at all. Each cell prints inside its own box — never stretched past
    the rules — and keeps the alignment and weight the source used there.
    """
    cells = geo.get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    rows = _markdown_cells(raw)
    if rows is None or len(rows) != len(cells):
        return None
    if any(len(row) != len(boxes) for row, boxes in zip(rows, cells, strict=True)):
        return None

    page = int((geo.get("rects") or [(0,)])[0][0])
    already = printed.get(page, [])
    claimed: list[tuple[float, float, float, float]] = []
    source = _markdown_cells(block.text)
    items: list[_Placement] = []
    for row_index, (row, boxes) in enumerate(zip(rows, cells, strict=True)):
        for column, cell in enumerate(boxes):
            # v1 prints one face per cell, so a token inside a cell has no run to
            # be printed with and is dropped rather than printed literally.
            text = emphasis.strip(row[column])
            if cell is None or not text:
                continue
            x0, top, x1, bottom, pad, size, align, bold, italic = cell
            box = (float(x0), float(top), float(x1), float(bottom))
            if _inside_any(box, claimed) or _inside_any(box, already):
                continue  # a merged cell reported twice: print it once
            claimed.append(box)
            item = _Placement(
                page=page,
                redact=[box],
                box=(x0 + pad, top + pad, x1 - pad, bottom - pad),
                source_box=box,
                size=max(float(size or 0.0), 6.0),
                text=text,
                kind="cell",
                align=str(align or "left"),
                color=tuple(float(part) for part in geo.get("color") or ()) or None,
                source_text=_source_cell(source, row_index, column),
            )
            if match_style:
                item.font_file = match_font(str(geo.get("font") or ""), bool(bold), bool(italic))
            if item.font_file is None and default_font is not None:
                item.font_file = str(default_font)
            items.append(item)
    if not items:
        return None
    printed.setdefault(page, []).extend(claimed)
    return items


def _bullet_for(marker: str) -> str:
    """The glyph to reprint for a list marker (unmapped bullets included)."""
    return _FALLBACK_BULLET if _CID_RE.match(marker) else marker


def _marked_text(text: str, markers: list[str]) -> str:
    """Put the source's own list markers back on the translated items.

    The markers were redacted with the item text, and the Markdown markers
    :func:`plain_text` strips are not what the page showed. When the translation
    has one line per source item the markers line up one-to-one; otherwise the
    first marker stands for the block, which is what a uniform list looked like
    anyway.
    """
    glyphs = [_bullet_for(marker) for marker in markers if marker]
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not glyphs:
        return text
    if len(glyphs) != len(lines):
        glyphs = [glyphs[0]] * len(lines)
    return "\n".join(f"{glyph} {line}" for glyph, line in zip(glyphs, lines, strict=True))


def _toc_parts(text: str) -> tuple[str, str] | None:
    """Split a translated contents entry into its title and page number."""
    match = _TOC_ENTRY_RE.match(" ".join(text.split()))
    if match is None or not match.group("title").strip():
        return None
    return match.group("title").strip(), match.group("number")


def placements(
    doc: Document,
    texts: dict[int, str | None],
    *,
    default_font: Path | str | None = None,
    match_style: bool = True,
) -> tuple[list[_Placement], int, int]:
    """Placements for every translatable block, plus how many were kept as-is
    and how many were merged into the block above them.

    *match_style* looks for a local face that reproduces the source's typography
    (family, weight, slant); it is switched off when the caller pinned the font.
    """
    built: list[_Placement] = []
    kept = 0
    merged = 0
    columns, occupied = _page_geometry(doc)
    #: Cell boxes already claimed on each page, so a merged cell reported twice
    #: is not printed twice.
    printed: dict[int, list[tuple[float, float, float, float]]] = {}
    for index, block in enumerate(doc.blocks):
        geo = block.meta.get("pdf")
        if not isinstance(geo, dict) or not geo.get("rects"):
            kept += 1
            continue
        if index in texts and texts[index] is None:
            # The translator's paragraph for this block's sentence is printed
            # for the block above it: this block's own lines go with it.
            if _absorb(built[-1] if built else None, geo):
                merged += 1
            else:
                kept += 1
            continue
        # Code keeps its source text: see the module docstring. A table is
        # rebuilt cell by cell, when its translation matches the source grid.
        if block.meta.get("table"):
            raw_table = texts.get(index)
            table_items = (
                _table_placements(
                    geo,
                    block,
                    raw_table,
                    match_style=match_style,
                    default_font=default_font,
                    printed=printed,
                )
                if raw_table and raw_table.strip()
                else None
            )
            if table_items is None:
                kept += 1
            else:
                built.extend(table_items)
            continue
        if _CODE_PLACEHOLDER_RE.match(block.text.strip()):
            kept += 1
            continue
        raw = texts.get(index)
        if not raw or not raw.strip():
            kept += 1
            continue
        if _sliver_box(geo):
            kept += 1
            continue
        kind = str(geo.get("kind") or block.kind)
        text = plain_text(raw, kind)
        if kind == "list":
            text = _marked_text(text, list(geo.get("markers") or []))
        toc = _toc_parts(text) if kind == "toc" else None
        page = int((geo.get("rects") or [(0,)])[0][0])
        item = (
            _placement(
                geo,
                text,
                kind,
                toc=toc,
                column=columns.get(page),
                occupied=occupied.get(page),
            )
            if text
            else None
        )
        if item is None:
            kept += 1
            continue
        item.face = str(geo.get("font") or "")
        item.match_style = match_style
        if match_style:
            item.font_file = match_font(
                item.face,
                bool(geo.get("bold")),
                bool(geo.get("italic")),
            )
        if item.font_file is None and default_font is not None:
            item.font_file = str(default_font)
        if kind in _RUN_KINDS and _has_tokens(block.text) and emphasis.agrees_with(block.text, raw):
            # The tokens are the model's own placement, and they are only trusted
            # because the source's counts match; two distinct styles are what a
            # run path is for, and a single one prints identically either way.
            runs = emphasis.to_runs(raw)
            if kind == "heading":
                runs = _runs_without_heading_marker(runs)
            if len({(bold, italic) for text, bold, italic in runs if text}) >= 2:
                item.runs = runs
        built.append(item)
    return built, kept, merged


def _resource_name(font_file: str) -> str:
    """The page-resource name for a font file.

    One page may carry blocks set in several faces, and a resource name is what
    binds inserted text to a font, so two faces must not share one.
    """
    digest = hashlib.sha1(str(font_file).encode("utf-8")).hexdigest()[:5]
    return f"{_FONT_NAME}_{digest}"


def _toc_line(item: _Placement, font_file: str, pymupdf: Any) -> str:
    """A contents entry as the source page had it: title, leader, page number.

    The leader is rebuilt rather than translated: the ingest collapses the run
    of dots to one separator, and what the reader recognised as a contents line
    is the dotted gap with the page number against the right margin.
    """
    title, number = item.toc or (item.text, "")
    if not number:
        return title
    try:
        font = pymupdf.Font(fontfile=font_file)
        dot = font.text_length(".", fontsize=item.size)
        used = font.text_length(f"{title}  {number}", fontsize=item.size)
    except Exception:  # noqa: BLE001 - an unmountable font is not worth a crash
        return f"{title} {number}"
    width = item.box[2] - item.box[0]
    count = int((width - used) / dot) if dot > 0 else 0
    if count < 3:
        return f"{title} {number}"
    return f"{title} " + "." * count + f" {number}"


def _heading_is_centred(page: Any, item: _Placement) -> bool:
    """Whether a heading sits clear of both margins and near the page's centre.

    Left-aligning a centred title is the one change a reader notices, so the
    decision lives in one place: the textbox path and the run path both ask here.
    Measured on the *source* box — a box stretched to the text column is centred
    on the page whatever the source did.
    """
    if item.kind != "heading":
        return False
    sx0, _stop, sx1, _sbottom = item.source_box or item.box
    return (
        sx0 > 40 and sx1 < page.rect.width - 40 and abs((sx0 + sx1) / 2 - page.rect.width / 2) < 18
    )


def _align_of(item: _Placement, pymupdf: Any) -> int:
    """The alignment to print *item* with.

    A table cell carries the alignment the source used there (the ingest reads
    it off the cell's own lines), so it is honoured as it is. Everything else
    starts left-aligned and may be re-decided by the caller.
    """
    if item.align:
        return {
            "left": pymupdf.TEXT_ALIGN_LEFT,
            "center": pymupdf.TEXT_ALIGN_CENTER,
            "right": pymupdf.TEXT_ALIGN_RIGHT,
        }.get(str(item.align), pymupdf.TEXT_ALIGN_LEFT)
    return pymupdf.TEXT_ALIGN_LEFT


def _insert_cell(page: Any, item: _Placement, pymupdf: Any, default_font: Path | str) -> str | None:
    """Print one table cell inside its own box.

    Returns ``"translated"``, ``"source"`` or ``None`` (neither would fit).
    Unlike :func:`_insert` there is no branch that borrows the room down to the
    bottom margin: a cell that outgrows its row would print over the rules and
    the rows below it, so its only way down is to shrink the type — and only to
    the floor.    When the translation still will not fit, the cell's own source
    text is printed in its place, at the source size when that fits and at the
    floor otherwise. A cell whose source held nothing is not a failure even
    then: pdfplumber reports sliver boxes a fraction of a line wide where a
    merged column collapsed, and an empty cell that stays empty has lost
    nothing — ``None`` is kept for cells whose *text* could not be placed.
    """
    font_file = str(item.font_file or default_font)
    fontname = _resource_name(font_file)
    align = _align_of(item, pymupdf)
    box = pymupdf.Rect(*item.box)
    floor = max(item.size * _CELL_MIN_SCALE, 4.0)

    def fits(text: str) -> bool:
        size = item.size
        while size >= floor:
            unused = page.insert_textbox(
                box,
                text,
                fontname=fontname,
                fontfile=font_file,
                fontsize=size,
                align=align,
                lineheight=1.15,
                color=item.color,
            )
            if unused >= 0:
                return True
            size -= max(item.size * 0.05, 0.25)
        return False

    if item.text and fits(item.text):
        return "translated"
    source = (item.source_text or "").strip()
    if source and fits(source):
        return "source"
    if not source:
        return "source"
    return None


@dataclass
class _Overflow:
    """The part of a block that did not fit its own box, waiting for a page.

    *text* is the textbox path's tail, and *words* the run path's — each word
    with the face it is printed in, because a tail that loses its emphasis reads
    as a different sentence from the one it continues.
    """

    item: _Placement
    text: str = ""
    words: list[tuple[str, str]] | None = None


def _words_of(text: str) -> list[tuple[str, bool]]:
    """The words of *text*, each with whether a hard line break precedes it."""
    tokens: list[tuple[str, bool]] = []
    for line_index, line in enumerate(text.splitlines()):
        for word_index, word in enumerate(line.split()):
            tokens.append((word, line_index > 0 and word_index == 0))
    return tokens


def _rejoin(tokens: list[tuple[str, bool]]) -> str:
    """Put tokens back together, restoring the line breaks they came with."""
    out = ""
    for index, (word, brk) in enumerate(tokens):
        if not index:
            out = word
        elif brk:
            out = out + "\n" + word
        else:
            out = out + " " + word
    return out


def _head_that_fits(fits: Any, tokens: list[tuple[str, bool]]) -> int:
    """How many leading *tokens* fit, by bisection on the number of words."""
    if not tokens:
        return 0
    if fits(_rejoin(tokens)):
        return len(tokens)
    lo, hi = 0, len(tokens)  # ``lo`` fits, ``hi`` does not
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(_rejoin(tokens[:mid])):
            lo = mid
        else:
            hi = mid
    return lo


class _Measurer:
    """Trial layouts, on a scratch page, to find how much text fits a box.

    ``insert_textbox`` reports only whether a run of text fitted, and it draws
    as it measures, so the real page cannot be used to probe. A blank page of the
    same size can: the only question asked of it is how many words fit.
    """

    def __init__(self, pymupdf: Any, page: Any) -> None:
        self._pymupdf = pymupdf
        self._document = pymupdf.open()
        self._page = self._document.new_page(
            width=float(page.rect.width), height=float(page.rect.height)
        )

    def fits(self, box: Any, text: str, **kwargs: Any) -> bool:
        return self._page.insert_textbox(self._pymupdf.Rect(*box), text, **kwargs) >= 0

    def close(self) -> None:
        """Drop the scratch page; it holds only trial layouts."""
        self._document.close()


class _RunText:
    """A block that mixes styles, laid out and printed one word at a time.

    PyMuPDF's layout does not read markup, so ``insert_textbox`` would print the
    emphasis sentinels as they stand. Wrapping is done here instead — greedy, with
    the metrics of the face each word is printed in — which also means the words
    that did not fit a box are known exactly, so a tail that has to continue
    overleaf keeps its faces.
    """

    def __init__(self, item: _Placement, default_font: Path | str, pymupdf: Any) -> None:
        self._pymupdf = pymupdf
        self.item = item
        self._metrics: dict[str, Any] = {}
        # Which family to look for the other faces in. The source's own family is
        # the usual answer, but a caller who pinned a font gets that font's
        # family instead: otherwise a chosen font loses *every* emphasis, because
        # no local face matches the source's family and the block prints plain.
        base = str(item.font_file or default_font)
        styled = item.face if item.match_style and item.face else ""
        if not styled:
            base_font = self._font_of(base)
            styled = str(base_font.name) if base_font is not None else ""
        # One face per style. Each style is matched separately, so bold text is
        # not printed in the regular face of the same family — a failure the page
        # shows no error for.
        faces: dict[tuple[bool, bool], str] = {}
        for _text, bold, italic in item.runs or []:
            style = (bold, italic)
            if style in faces:
                continue
            found = match_font(styled, bold, italic) if styled else None
            if found is None and style != (False, False):
                logger.debug(
                    "PDF rebuild: no local {} face; printing that run plain",
                    "bold" if bold else "italic",
                )
            faces[style] = str(found or base)
        self.words: list[tuple[str, str]] = [
            (word, faces[(bold, italic)])
            for text, bold, italic in item.runs or []
            for word in text.split()
        ]

    def _font_of(self, font_file: str) -> Any:
        """The measuring font, or ``None`` when it cannot be mounted."""
        if font_file not in self._metrics:
            try:
                self._metrics[font_file] = self._pymupdf.Font(fontfile=font_file)
            except Exception:  # noqa: BLE001 - an unmountable font is not worth a crash
                self._metrics[font_file] = None
        return self._metrics[font_file]

    def advance(self, text: str, font_file: str, size: float) -> float:
        """The width of *text* in *font_file* at *size*."""
        font = self._font_of(font_file)
        if font is None:
            return len(text) * size * 0.5
        return float(font.text_length(text, fontsize=size))

    def layout(
        self, words: list[tuple[str, str]], size: float, width: float
    ) -> list[list[tuple[str, str, float, float]]]:
        """Greedy word wrap at *size*: per line, ``(word, face, width, gap)``.

        The gap is the space *before* the word, measured in that word's own
        face, so printing a line is a matter of adding the two up.
        """
        lines: list[list[tuple[str, str, float, float]]] = []
        line: list[tuple[str, str, float, float]] = []
        used = 0.0
        for word, face in words:
            word_width = self.advance(word, face, size)
            gap = self.advance(" ", face, size) if line else 0.0
            if line and used + gap + word_width > width:
                lines.append(line)
                line, used, gap = [], 0.0, 0.0
            line.append((word, face, word_width, gap))
            used += gap + word_width
        if line:
            lines.append(line)
        return lines

    def print(
        self,
        page: Any,
        box: tuple[float, float, float, float],
        words: list[tuple[str, str]],
        size: float,
    ) -> tuple[list[tuple[str, str]], float]:
        """Print as many of *words* as *box* holds at *size*.

        Returns what is left and the height the printed lines used, so a caller
        that lays several blocks down one column can move its cursor past them.
        """
        item = self.item
        x0, top, x1, bottom = box
        width = max(x1 - x0, 1.0)
        height = bottom - top
        lines = self.layout(words, size, width)
        placed = 0
        used = 0.0
        for line in lines:
            if placed and used + size * _RUN_LEADING > height + size * 0.5:
                break
            placed += len(line)
            used += size * _RUN_LEADING
        if not placed and words:
            # Not even one wrapped line has room: print it rather than leave the
            # box empty, so the words are on the page and the tail still moves on.
            placed = len(lines[0])
        if not placed:
            return [], 0.0
        align = _align_of(item, self._pymupdf)
        centered = align == self._pymupdf.TEXT_ALIGN_CENTER or _heading_is_centred(page, item)
        printed = 0
        used_lines = 0
        for line_index, line in enumerate(lines):
            if printed >= placed:
                break
            # A line that starts printing owns its whole band, even the partial
            # word tail of the last one.
            used_lines = line_index + 1
            natural = sum(word_width + gap for _word, _face, word_width, gap in line)
            x = x0 + max((width - natural) / 2.0, 0.0) if centered else x0
            baseline = top + size + line_index * size * _RUN_LEADING
            for word, face, _word_width, gap in line:
                if printed >= placed:
                    break
                x += gap
                page.insert_text(
                    (x, baseline),
                    word,
                    fontname=_resource_name(face),
                    fontfile=face,
                    fontsize=size,
                    color=item.color,
                )
                x += self.advance(word, face, size)
                printed += 1

        return words[printed:], used_lines * size * _RUN_LEADING


def _insert(
    page: Any, item: _Placement, pymupdf: Any, default_font: Path | str, measurer: _Measurer
) -> tuple[str | None, _Overflow | None]:
    """Print *item* inside its box, at the size the source set it in.

    Returns ``(outcome, overflow)``: ``"block"`` once the text is on the page
    (``None`` when nothing could be placed, which a cell can still answer), and
    the part of a block that did not fit its box, for the caller to continue on a
    new page. A table cell is handed to :func:`_insert_cell` and never continues
    overleaf: the rules around it are the layout.

    The TTF is passed on every call: ``Page.insert_font`` only registers the font
    in the resource dictionary of the page it is called on, so a page that did
    not call it would fall back to a base-14 font and raise on the Vietnamese
    glyphs it cannot encode.
    """
    if item.kind == "cell":
        return _insert_cell(page, item, pymupdf, default_font), None
    x0, top, x1, bottom = item.box
    limit = max(page.rect.height - _BOTTOM_MARGIN, bottom)
    box = (x0, top, x1, min(bottom, limit))
    if item.runs is not None:
        runs = _RunText(item, default_font, pymupdf)
        rest, _used = runs.print(page, box, runs.words, item.size)
        return "block", (_Overflow(item, words=rest) if rest else None)
    font_file = str(item.font_file or default_font)
    fontname = _resource_name(font_file)
    text = _toc_line(item, font_file, pymupdf) if item.toc else item.text
    align = _align_of(item, pymupdf)
    if _heading_is_centred(page, item):
        align = pymupdf.TEXT_ALIGN_CENTER
    tokens = _words_of(text)

    def fits(candidate: str, target: tuple[float, float, float, float] = box) -> bool:
        return measurer.fits(
            target,
            candidate,
            fontname=fontname,
            fontfile=font_file,
            fontsize=item.size,
            align=align,
            lineheight=1.15,
            color=item.color,
        )

    count = _head_that_fits(fits, tokens)
    if count == 0 and tokens:
        # The box does not even hold one line at the source size. One line is put
        # down anyway — a block gone blank is a hole in the page, and redaction
        # has already taken the source's words — and the rest moves on.
        single = (x0, top, x1, max(top + item.size * 1.4, bottom))
        count = 1 if fits(_rejoin(tokens[:1]), single) else 0
        page.insert_text(
            (x0, top + item.size),
            _rejoin(tokens[:count]),
            fontname=fontname,
            fontfile=font_file,
            fontsize=item.size,
            color=item.color,
        )
        return "block", _Overflow(item, text=_rejoin(tokens[count:]))
    unused = page.insert_textbox(
        pymupdf.Rect(*box),
        _rejoin(tokens[:count]),
        fontname=fontname,
        fontfile=font_file,
        fontsize=item.size,
        align=align,
        lineheight=1.15,
        color=item.color,
    )
    if unused < 0:  # pragma: no cover - the probe and the page disagree
        page.insert_text(
            (x0, top + item.size),
            _rejoin(tokens[:count]),
            fontname=fontname,
            fontfile=font_file,
            fontsize=item.size,
            color=item.color,
        )
    rest = tokens[count:]
    return "block", (_Overflow(item, text=_rejoin(rest)) if rest else None)


class _Continuation:
    """The overleaf column shared by one source page's spilling blocks.

    A block that did not fit its box used to insert a whole page to itself, so a
    source page whose blocks each spilled a line or two left a run of nearly
    empty pages behind it — one per block; on a 69-page document that ran to 48
    extra pages. The tails instead share one column: each prints below the one
    before it, in block order, and a fresh page is inserted only when the column
    cannot hold one more line. They already followed the source page in that
    order, so only the pagination changes, not the reading order.
    """

    def __init__(
        self,
        document: Any,
        pymupdf: Any,
        default_font: Path | str,
        *,
        after: int,
        width: float,
        height: float,
    ) -> None:
        self._document = document
        self._pymupdf = pymupdf
        self._default_font = default_font
        self._after = after
        self._width = width
        self._height = height
        self._page: Any = None
        self._y = _TOP_MARGIN

    def _turn(self) -> None:
        """Insert the page continuing the column below the last one."""
        index = self._after + 1 if self._page is None else self._page.number + 1
        self._page = self._document.new_page(pno=index, width=self._width, height=self._height)
        self._y = _TOP_MARGIN

    def carry(self, over: _Overflow) -> int:
        """Print *over* where the column stands; returns how many pages it added.

        The cursor only moves down. A remaining band smaller than one line is
        traded for a fresh page first, and every pass prints at least one line —
        together those make progress the contract, so the loop cannot spin.
        """
        added = 0
        size = over.item.size
        x0, x1 = over.item.box[0], over.item.box[2]
        bottom = self._height - _BOTTOM_MARGIN
        while True:
            if self._page is None or bottom - self._y < size * 1.4:
                self._turn()
                added += 1
            box = (x0, self._y, x1, bottom)
            if over.words is not None:
                rest, used = _RunText(over.item, self._default_font, self._pymupdf).print(
                    self._page, box, over.words, size
                )
                self._y += used
                if not rest:
                    return added
                over = _Overflow(over.item, words=rest)
                continue
            font_file = str(over.item.font_file or self._default_font)
            fontname = _resource_name(font_file)
            align = _align_of(over.item, self._pymupdf)
            tokens = _words_of(over.text)
            kwargs = {
                "fontname": fontname,
                "fontfile": font_file,
                "fontsize": size,
                "align": align,
                "lineheight": 1.15,
                "color": over.item.color,
            }
            measurer = _Measurer(self._pymupdf, self._page)
            try:
                count = _head_that_fits(partial(measurer.fits, box, **kwargs), tokens)
                if count == 0 and tokens:
                    # Not one word fits the band: it goes down anyway, so the
                    # loop makes progress rather than printing nothing again.
                    count = 1
                unused = self._page.insert_textbox(
                    self._pymupdf.Rect(*box), _rejoin(tokens[:count]), **kwargs
                )
            finally:
                measurer.close()
            if unused < 0:  # pragma: no cover - the probe and the page disagree
                self._page.insert_text(
                    (box[0], box[1] + size),
                    _rejoin(tokens[:count]),
                    fontname=fontname,
                    fontfile=font_file,
                    fontsize=size,
                    color=over.item.color,
                )
                # What was drawn up there cannot be measured, so the cursor
                # skips the whole page rather than risk printing over it.
                self._y = bottom
            else:
                self._y += (bottom - box[1]) - unused
            rest = tokens[count:]
            if not rest:
                return added
            over = _Overflow(over.item, text=_rejoin(rest))


#: Line advance of the hand-laid-out run path, in multiples of the type size.
_RUN_LEADING = 1.15


def _font_xrefs(page: Any) -> set[int]:
    """The object ids of a page's fonts; an empty set when it has none."""
    try:
        return {int(entry[0]) for entry in page.get_fonts(full=True)}
    except Exception:  # noqa: BLE001 - a page without resources simply has none
        return set()


def _fix_tounicode_targets(raw: str) -> str:
    """Rewrite the nbsp/soft-hyphen targets of one inserted font's map.

    Split out so the entry regex can be exercised without a PDF at hand.
    """
    return _TOUNICODE_ENTRY_RE.sub(
        lambda m: f"<{m.group(1)}> <{_TOUNICODE_SHARED_GLYPHS[m.group(2).lower()]}>", raw
    )


def _repair_tounicode(document: Any, page: Any, before: set[int]) -> int:
    """Point the inserted fonts' copied-text map back at the ASCII code points.

    Only the fonts this page *gained* are rewritten (the ids in *before* are the
    source page's own), so a block left in the source language still copies out
    exactly as it did. Returns the number of maps repaired.
    """
    repaired = 0
    for entry in page.get_fonts(full=True):
        xref = int(entry[0])
        if xref in before:
            continue
        kind, value = document.xref_get_key(xref, "ToUnicode")
        if kind != "xref":
            continue
        stream_xref = int(value.split()[0])
        raw = document.xref_stream(stream_xref).decode("latin-1")
        fixed = _fix_tounicode_targets(raw)
        if fixed != raw:
            document.update_stream(stream_xref, fixed.encode("latin-1"), compress=True)
            repaired += 1
    if repaired:
        logger.debug("PDF rebuild: repaired {} ToUnicode map(s) on this page", repaired)
    return repaired


def rebuild_pdf(
    source: str | Path,
    doc: Document,
    texts: dict[int, str | None],
    out_path: str | Path,
    *,
    font_path: str | Path | None = None,
) -> RebuildStats:
    """Write the translated PDF to *out_path*; returns what was changed.

    *texts* maps a source-block index to its translated Markdown paragraph (see
    :func:`polyglot_lc.pipeline.assembler.block_translations`); ``None`` means
    the block's words are in the paragraph printed for the block above it.
    """
    import pymupdf

    font_file = resolve_font(font_path)
    # An explicitly chosen font is exactly that: do not second-guess it with a
    # local face that happens to match the source's family.
    pinned = bool(font_path) or bool(os.getenv("PDF_FONT_PATH", "").strip())
    items, kept, merged = placements(doc, texts, default_font=font_file, match_style=not pinned)
    stats = RebuildStats(font=str(font_file), kept=kept, merged=merged)
    if not items:
        logger.warning("PDF rebuild: nothing to replace ({} block(s) kept)", kept)
        return stats

    by_page: dict[int, list[_Placement]] = {}
    for item in items:
        by_page.setdefault(item.page, []).append(item)

    document = pymupdf.open(str(source))
    stats.pages = document.page_count
    #: Pages inserted so far. They sit after the source page they continue, so
    #: every page that follows is that many places further down the document.
    shift = 0
    for page_no, page_items in sorted(by_page.items()):
        #: Where the source page stands now. Pages added to continue it go
        #: after, so this index never moves while the document grows.
        src_index = page_no + shift
        if page_no < 0 or src_index >= document.page_count:
            stats.failures.append(f"page {page_no} is outside the document")
            continue
        page = document[src_index]
        measurer = _Measurer(pymupdf, page)
        # The page's own fonts, so the ToUnicode repair below can tell them from
        # the ones our insertion adds.
        fonts_before = _font_xrefs(page)
        #: The overleaf column this page's spilling blocks share.
        cont: _Continuation | None = None
        for item in page_items:
            for x0, top, x1, bottom in item.redact:
                page.add_redact_annot(pymupdf.Rect(x0, top, x1, bottom), fill=None, text=None)
        # Remove the source text only: images and vector art (table rules,
        # figures, the page's own line work) are part of the layout.
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
        for item in page_items:
            outcome, over = _insert(page, item, pymupdf, font_file, measurer)
            if outcome is None:
                stats.failures.append(f"page {page_no}: text did not fit")
                continue
            stats.placed += 1
            if item.kind == "cell":
                if outcome == "translated":
                    stats.cells += 1
                else:
                    stats.cells_kept += 1
            if over is None:
                continue
            # The block was longer than the box the source set it in: its tail
            # carries on overleaf, at the same size, below the tails already
            # there — one shared column, so spilling blocks stack instead of
            # each claiming its own nearly-empty page.
            if cont is None:
                cont = _Continuation(
                    document,
                    pymupdf,
                    font_file,
                    after=src_index,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                )
            added = cont.carry(over)
            shift += added
            stats.continued += added
            stats.pages += added
            # Inserting pages invalidates the page object; ask for it again —
            # the source page, which the insertions did not move.
            page = document[src_index]
        measurer.close()
        # Copied text is part of the output: repair the map of the fonts we
        # just inserted before the document is written.
        stats.repaired_maps += _repair_tounicode(document, page, fonts_before)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(out), garbage=4, deflate=True)
    document.close()
    logger.info(
        "PDF rebuild: {} block(s) placed ({} cell(s) translated, {} kept in source), "
        "{} merged into the block above, {} kept as source, {} page(s) "
        "({} added for text that continued overleaf), {} font map(s) repaired -> {}",
        stats.placed,
        stats.cells,
        stats.cells_kept,
        stats.merged,
        stats.kept,
        stats.pages,
        stats.continued,
        stats.repaired_maps,
        out,
    )
    return stats


__all__ = [
    "PdfOutputError",
    "RebuildStats",
    "block_translations",
    "placements",
    "plain_text",
    "rebuild_pdf",
    "resolve_font",
]
