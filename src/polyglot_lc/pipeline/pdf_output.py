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

A **table** is rebuilt as well, but cell by cell, inside the grid the source
page already drew: the rules stay exactly where they were and each translated
cell is printed into its own box. Two rules keep that honest:

* a cell's type may be shrunk to 70% of its source size (``_CELL_MIN_SCALE``)
  but no further, and never outside its box — unlike a block, a cell does not
  get to borrow the room down to the bottom margin, because what surrounds it is
  the layout;
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

The boxes come from :mod:`polyglot_lc.ingest.pdf_layout`, which records a block's
source rectangles as it extracts it; see ``block.meta["pdf"]``.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger

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

#: How far the block may be shrunk before we accept a less exact fit (fraction
#: of the source size). Below this, source fidelity is worth more than the fit.
_MIN_SCALE = 0.55

#: The same limit for a table cell, and a tighter one: a cell is surrounded by
#: the rules of its grid, so shrinking it is the only way it can grow, and text
#: much smaller than the source's would read as a different document's table.
_CELL_MIN_SCALE = 0.70


class PdfOutputError(RuntimeError):
    """The PDF could not be rebuilt (usually: no Unicode font is available)."""


@dataclass
class RebuildStats:
    """What the rebuild changed, for the run report."""

    pages: int = 0
    placed: int = 0
    kept: int = 0
    #: Table cells printed with their translation, and cells that fell back to
    #: their own source text because the translation would not fit the box.
    cells: int = 0
    cells_kept: int = 0
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
    """The text to print on the page: the translation without Markdown syntax."""
    lines = markdown_text.splitlines() or [markdown_text]
    out: list[str] = []
    for line in lines:
        stripped = _HEADING_RE.sub("", line) if kind == "heading" else line
        if kind == "list":
            stripped = _LIST_ITEM_RE.sub("", stripped)
        out.append(stripped.strip())
    return "\n".join(line for line in out if line).strip()


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
            text = row[column]
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
    texts: dict[int, str],
    *,
    default_font: Path | str | None = None,
    match_style: bool = True,
) -> tuple[list[_Placement], int]:
    """Placements for every translatable block, plus how many were kept as-is.

    *match_style* looks for a local face that reproduces the source's typography
    (family, weight, slant); it is switched off when the caller pinned the font.
    """
    built: list[_Placement] = []
    kept = 0
    columns, occupied = _page_geometry(doc)
    #: Cell boxes already claimed on each page, so a merged cell reported twice
    #: is not printed twice.
    printed: dict[int, list[tuple[float, float, float, float]]] = {}
    for index, block in enumerate(doc.blocks):
        geo = block.meta.get("pdf")
        if not isinstance(geo, dict) or not geo.get("rects"):
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
        if match_style:
            item.font_file = match_font(
                str(geo.get("font") or ""),
                bool(geo.get("bold")),
                bool(geo.get("italic")),
            )
        if item.font_file is None and default_font is not None:
            item.font_file = str(default_font)
        built.append(item)
    return built, kept


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


def _insert(page: Any, item: _Placement, pymupdf: Any, default_font: Path | str) -> str | None:
    """Print *item* inside its box, shrinking the font until it fits.

    Returns ``"block"`` once the text is on the page, and ``None`` when even the
    fallback could not place it. A table cell is handed to :func:`_insert_cell`.

    The TTF is passed on every call: ``Page.insert_font`` only registers the font
    in the resource dictionary of the page it is called on, so a page that did
    not call it would fall back to a base-14 font and raise on the Vietnamese
    glyphs it cannot encode.
    """
    if item.kind == "cell":
        return _insert_cell(page, item, pymupdf, default_font)
    font_file = str(item.font_file or default_font)
    fontname = _resource_name(font_file)
    text = item.text
    if item.toc:
        text = _toc_line(item, font_file, pymupdf)
    x0, top, x1, bottom = item.box
    limit = max(page.rect.height - _BOTTOM_MARGIN, bottom)
    box = pymupdf.Rect(x0, top, x1, min(bottom, limit))
    align = _align_of(item, pymupdf)
    if item.kind == "heading":
        # A title that sits clear of both margins and near the page's centre is
        # a centred title; left-aligning it is the one change a reader notices.
        # Measured on the *source* box: a box stretched to the text column is
        # centred on the page whatever the source did.
        sx0, _stop, sx1, _sbottom = item.source_box or item.box
        page_center = page.rect.width / 2
        if sx0 > 40 and sx1 < page.rect.width - 40 and abs((sx0 + sx1) / 2 - page_center) < 18:
            align = pymupdf.TEXT_ALIGN_CENTER
    floor = max(item.size * _MIN_SCALE, 4.0)
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
            return "block"
        size -= max(item.size * 0.05, 0.25)
    # Still too long for the original space: use the room down to the bottom
    # margin at the floor size rather than dropping the lines.
    unused = page.insert_textbox(
        pymupdf.Rect(x0, top, x1, limit),
        text,
        fontname=fontname,
        fontfile=font_file,
        fontsize=floor,
        align=align,
        lineheight=1.15,
        color=item.color,
    )
    return "block" if unused >= 0 else None


def rebuild_pdf(
    source: str | Path,
    doc: Document,
    texts: dict[int, str],
    out_path: str | Path,
    *,
    font_path: str | Path | None = None,
) -> RebuildStats:
    """Write the translated PDF to *out_path*; returns what was changed.

    *texts* maps a source-block index to its translated Markdown paragraph (see
    :func:`polyglot_lc.pipeline.assembler.block_translations`).
    """
    import pymupdf

    font_file = resolve_font(font_path)
    # An explicitly chosen font is exactly that: do not second-guess it with a
    # local face that happens to match the source's family.
    pinned = bool(font_path) or bool(os.getenv("PDF_FONT_PATH", "").strip())
    items, kept = placements(doc, texts, default_font=font_file, match_style=not pinned)
    stats = RebuildStats(font=str(font_file), kept=kept)
    if not items:
        logger.warning("PDF rebuild: nothing to replace ({} block(s) kept)", kept)
        return stats

    by_page: dict[int, list[_Placement]] = {}
    for item in items:
        by_page.setdefault(item.page, []).append(item)

    document = pymupdf.open(str(source))
    stats.pages = document.page_count
    for page_no, page_items in sorted(by_page.items()):
        if page_no < 0 or page_no >= document.page_count:
            stats.failures.append(f"page {page_no} is outside the document")
            continue
        page = document[page_no]
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
            outcome = _insert(page, item, pymupdf, font_file)
            if outcome is None:
                stats.failures.append(f"page {page_no}: text did not fit")
                continue
            stats.placed += 1
            if item.kind != "cell":
                continue
            if outcome == "translated":
                stats.cells += 1
            else:
                stats.cells_kept += 1

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(out), garbage=4, deflate=True)
    document.close()
    logger.info(
        "PDF rebuild: {} block(s) placed ({} cell(s) translated, {} kept in source), "
        "{} kept as source, {} page(s) -> {}",
        stats.placed,
        stats.cells,
        stats.cells_kept,
        stats.kept,
        stats.pages,
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
