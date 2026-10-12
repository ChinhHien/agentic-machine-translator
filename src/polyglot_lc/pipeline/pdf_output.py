"""Rebuild the source PDF with the translated text, keeping its layout.

The source PDF supplies everything that is *not* language: page size, images,
vector graphics, table rules and the position and size of every text block.
This module replaces only the blocks the pipeline translated — in place, in the
same rectangle, at the same size — with a Unicode font embedded for Vietnamese.

What is deliberately left alone (and why):

* **tables** — a table's meaning lives in its grid, and re-flowing its text into
  the same box without the columns would be worse than leaving it readable in
  the source language. Tables *are* translated in the Markdown output.
* **code** — never translated by the pipeline (it is shielded as
  ``⟦CODE_n⟧``), so the source text stays.
* **anything without geometry or translation** — a chunk whose retries were
  exhausted leaves its source lines in place rather than blanking the page, so
  a partial run still yields a readable document.

The boxes come from :mod:`polyglot_lc.ingest.pdf_layout`, which records a block's
source rectangles as it extracts it; see ``block.meta["pdf"]``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from polyglot_lc.ingest.model import Document
from polyglot_lc.pipeline.assembler import block_translations

#: Font resource name used for the embedded Unicode font inside every page.
_FONT_NAME = "pgf"

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


class PdfOutputError(RuntimeError):
    """The PDF could not be rebuilt (usually: no Unicode font is available)."""


@dataclass
class RebuildStats:
    """What the rebuild changed, for the run report."""

    pages: int = 0
    placed: int = 0
    kept: int = 0
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


def _placement(geo: dict[str, Any], text: str, kind: str) -> _Placement | None:
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
    return _Placement(
        page=page,
        redact=[(r[1], r[2], r[3], r[4]) for r in rects],
        box=box,
        size=size,
        text=text,
        kind=kind,
    )


def placements(doc: Document, texts: dict[int, str]) -> tuple[list[_Placement], int]:
    """Placements for every translatable block, plus how many were kept as-is."""
    built: list[_Placement] = []
    kept = 0
    for index, block in enumerate(doc.blocks):
        geo = block.meta.get("pdf")
        if not isinstance(geo, dict) or not geo.get("rects"):
            kept += 1
            continue
        # Tables and code keep their source text: see the module docstring.
        if block.meta.get("table") or _CODE_PLACEHOLDER_RE.match(block.text.strip()):
            kept += 1
            continue
        raw = texts.get(index)
        if not raw or not raw.strip():
            kept += 1
            continue
        kind = str(geo.get("kind") or block.kind)
        text = plain_text(raw, kind)
        item = _placement(geo, text, kind) if text else None
        if item is None:
            kept += 1
            continue
        built.append(item)
    return built, kept


def _insert(page: Any, item: _Placement, pymupdf: Any, font_file: Path) -> bool:
    """Print *item* inside its box, shrinking the font until it fits.

    The TTF is passed on every call: ``Page.insert_font`` only registers the font
    in the resource dictionary of the page it is called on, so a page that did
    not call it would fall back to a base-14 font and raise on the Vietnamese
    glyphs it cannot encode.
    """
    x0, top, x1, bottom = item.box
    limit = max(page.rect.height - _BOTTOM_MARGIN, bottom)
    box = pymupdf.Rect(x0, top, x1, min(bottom, limit))
    align = pymupdf.TEXT_ALIGN_LEFT
    if item.kind == "heading":
        # A title that sits clear of both margins and near the page's centre is
        # a centred title; left-aligning it is the one change a reader notices.
        page_center = page.rect.width / 2
        if x0 > 40 and x1 < page.rect.width - 40 and abs((x0 + x1) / 2 - page_center) < 18:
            align = pymupdf.TEXT_ALIGN_CENTER
    floor = max(item.size * _MIN_SCALE, 4.0)
    size = item.size
    while size >= floor:
        unused = page.insert_textbox(
            box,
            item.text,
            fontname=_FONT_NAME,
            fontfile=str(font_file),
            fontsize=size,
            align=align,
            lineheight=1.15,
        )
        if unused >= 0:
            return True
        size -= max(item.size * 0.05, 0.25)
    # Still too long for the original space: use the room down to the bottom
    # margin at the floor size rather than dropping the lines.
    unused = page.insert_textbox(
        pymupdf.Rect(x0, top, x1, limit),
        item.text,
        fontname=_FONT_NAME,
        fontfile=str(font_file),
        fontsize=floor,
        align=align,
        lineheight=1.15,
    )
    return unused >= 0


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
    items, kept = placements(doc, texts)
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
            if _insert(page, item, pymupdf, font_file):
                stats.placed += 1
            else:
                stats.failures.append(f"page {page_no}: text did not fit")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(out), garbage=4, deflate=True)
    document.close()
    logger.info(
        "PDF rebuild: {} block(s) placed, {} kept as source, {} page(s) -> {}",
        stats.placed,
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
