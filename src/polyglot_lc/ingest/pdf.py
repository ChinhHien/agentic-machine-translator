"""PDF loader: convert PDF bytes to Markdown with MarkItDown, then parse.

MarkItDown turns the PDF into Markdown (headings, lists, tables where the
source allows it), and the resulting markdown is fed through the *same*
structural parser used for ``.md`` files — so downstream chunking, code
shielding and ``restore_code`` behave identically regardless of the origin
container. A legacy pypdf→pdfplumber extractor is kept as a fallback so a
missing/broken MarkItDown install never blocks a run.
"""

from __future__ import annotations

import io
import unicodedata

from loguru import logger

from polyglot_lc.ingest.md import parse_markdown
from polyglot_lc.ingest.model import Block, Document

MIN_CHARS_PER_PAGE = 200

#: Lazily-constructed MarkItDown instance (construction is not free).
_MARKITDOWN = None
_MARKITDOWN_UNAVAILABLE = False


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _get_markitdown():
    """Return a cached MarkItDown instance, or None when it is unavailable."""
    global _MARKITDOWN, _MARKITDOWN_UNAVAILABLE
    if _MARKITDOWN is not None or _MARKITDOWN_UNAVAILABLE:
        return _MARKITDOWN
    try:
        from markitdown import MarkItDown

        _MARKITDOWN = MarkItDown(enable_plugins=False)
    except Exception as exc:  # noqa: BLE001 - optional dependency
        _MARKITDOWN_UNAVAILABLE = True
        logger.warning("MarkItDown unavailable ({}); using legacy PDF extractor", exc)
    return _MARKITDOWN


def convert_pdf_to_markdown(data: bytes) -> str:
    """Convert PDF bytes to Markdown text via MarkItDown.

    Raises whatever MarkItDown raises; callers handle the fallback.
    """
    converter = _get_markitdown()
    if converter is None:
        raise RuntimeError("MarkItDown is not installed")
    result = converter.convert_stream(io.BytesIO(data), file_extension=".pdf")
    return (getattr(result, "text_content", "") or "").strip()


def load_pdf(data: bytes) -> Document:
    """Load a PDF: MarkItDown → Markdown → structural parser.

    Falls back to the legacy pypdf/pdfplumber text extractor when MarkItDown is
    missing, fails, or yields no text (e.g. scanned/image-only PDFs).
    """
    markdown_text = ""
    try:
        markdown_text = convert_pdf_to_markdown(data)
    except Exception as exc:  # noqa: BLE001 - graceful degradation
        logger.warning("MarkItDown PDF conversion failed ({}); using fallback", exc)

    if markdown_text.strip():
        doc = parse_markdown(
            markdown_text,
            source_format="pdf",
            extra_metadata={"converter": "markitdown"},
        )
        if doc.blocks:
            return doc
        logger.warning("MarkItDown produced no blocks; using legacy PDF extractor")

    return _load_pdf_legacy(data)


# --------------------------------------------------------------------------
# Legacy fallback (pypdf first, pdfplumber for sparse/complex layouts)
# --------------------------------------------------------------------------
def _split_pages(text: str) -> list[str]:
    """Split extracted page text into paragraph blocks on blank lines."""
    paras = [p.strip() for p in re_split_paragraphs(text)]
    return [p for p in paras if p]


def re_split_paragraphs(text: str):
    import re

    return re.split(r"\n\s*\n", text)


def _load_pdf_legacy(data: bytes) -> Document:
    """Extract plain text with pypdf, falling back to pdfplumber.

    Fallback triggers when pypdf yields sparse text (< MIN_CHARS_PER_PAGE on
    average) — a sign of complex layouts or scanned content.
    """
    from pypdf import PdfReader

    pages: list[str] = []
    reader = PdfReader(io.BytesIO(data))
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - tolerate broken pages
            pages.append("")

    avg_chars = (sum(len(p) for p in pages) / len(pages)) if pages else 0
    if avg_chars < MIN_CHARS_PER_PAGE:
        pages = _pdfplumber_pages(data)

    blocks: list[Block] = []
    for page_text in pages:
        for para in _split_pages(page_text):
            blocks.append(Block(kind="paragraph", text=_nfc(para)))
    if not blocks:
        raise ValueError("No extractable text found in PDF (may be scanned images)")

    return Document(
        blocks=blocks,
        lang_hint="auto",
        metadata={"format": "pdf", "pages": len(pages), "converter": "legacy"},
    )


def _pdfplumber_pages(data: bytes) -> list[str]:
    """Second-chance extraction with pdfplumber."""
    import pdfplumber

    texts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                texts.append("")
    return texts
