"""PDF loader: the source document, with per-block page geometry.

A PDF is the only input the project accepts, and there is exactly one extractor:
:mod:`polyglot_lc.ingest.pdf_layout` reads the document's own typography
(monospace runs are code, oversized runs are headings, glyph gaps are word
breaks) into Markdown with real structure — and records the page rectangle(s)
each block was read from. Those rectangles are what let the same translation be
re-emitted as a layout-preserving PDF (see
:mod:`polyglot_lc.pipeline.pdf_output`) instead of only as Markdown.

A scanned document is out of scope on purpose: the text layer is required and
nothing here runs OCR, so a page without one fails loudly in the extractor
rather than yielding a silent, empty document.
"""

from __future__ import annotations

from loguru import logger

from polyglot_lc.ingest.model import Document
from polyglot_lc.ingest.pdf_layout import load_pdf_layout


def load_pdf(data: bytes) -> Document:
    """Extract *data*: the document's text *and* the geometry of every block."""
    doc = load_pdf_layout(data)
    logger.debug(
        "PDF: {} block(s), {} with page geometry",
        len(doc.blocks),
        sum(1 for block in doc.blocks if block.meta.get("pdf")),
    )
    return doc
