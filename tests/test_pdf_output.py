"""Tests for the rebuilt, layout-preserving PDF output.

Two halves: the pure mapping from chunk translations to source blocks, and the
PyMuPDF rebuild itself. The rebuild needs a Unicode TTF with Vietnamese
coverage; when the machine has none the end-to-end part is skipped rather than
failed, because the *absence of a font* is a configuration problem the code
already reports clearly.
"""

from __future__ import annotations

import pytest

from polyglot_lc.ingest import load_document
from polyglot_lc.ingest.model import Block, Document
from polyglot_lc.pipeline import pdf_output
from polyglot_lc.pipeline.assembler import block_translations
from polyglot_lc.pipeline.pdf_output import (
    PdfOutputError,
    placements,
    plain_text,
    rebuild_pdf,
    resolve_font,
)


class _FakeChunk:
    """Just what the assembler reads off a chunk: its id and its blocks."""

    def __init__(self, cid: str, block_refs: list[int]) -> None:
        self.id = cid
        self.block_refs = block_refs


@pytest.fixture(scope="module")
def unicode_font():
    """A Vietnamese-capable TTF, or a skip when this machine has none."""
    try:
        return resolve_font()
    except PdfOutputError as exc:  # pragma: no cover - depends on the machine
        pytest.skip(str(exc))


# ------------------------- chunk -> block mapping ---------------------------


def test_paragraphs_map_to_their_own_blocks():
    c0 = _FakeChunk("c0", [0, 1])
    assert block_translations([c0], {"c0": "P0\n\nP1"}) == {0: "P0", 1: "P1"}


def test_the_overlapped_head_is_not_emitted_twice():
    """The earlier chunk owns the shared block, exactly as `assemble` decides."""
    c0 = _FakeChunk("c0", [0, 1])
    c1 = _FakeChunk("c1", [1, 2])  # re-covers block 1 as its head
    out = block_translations([c0, c1], {"c0": "P0\n\nP1", "c1": "P1\n\nP2"})
    assert out == {0: "P0", 1: "P1", 2: "P2"}


def test_blocks_the_translation_has_no_text_for_are_left_out():
    """A merged translation must not print the same paragraph over two blocks."""
    c0 = _FakeChunk("c0", [0, 1, 2])
    assert block_translations([c0], {"c0": "Only one paragraph"}) == {0: "Only one paragraph"}


def test_extra_paragraphs_stay_with_the_last_block():
    c0 = _FakeChunk("c0", [0, 1])
    out = block_translations([c0], {"c0": "P0\n\nP1a\n\nP1b"})
    assert out[0] == "P0"
    assert out[1] == "P1a\n\nP1b"


def test_a_chunk_without_a_translation_contributes_nothing():
    assert block_translations([_FakeChunk("c0", [0])], {}) == {}


# ------------------------------ plain text ----------------------------------


def test_plain_text_drops_the_markdown_the_page_cannot_show():
    assert plain_text("## Tiêu đề", "heading") == "Tiêu đề"
    assert plain_text("- một\n- hai", "list") == "một\nhai"
    assert plain_text("1. một\n2. hai", "list") == "một\nhai"
    assert plain_text("Câu bình thường.", "body") == "Câu bình thường."


# ------------------------------- placements ---------------------------------


def _doc_with_geometry() -> Document:
    return Document(
        blocks=[
            Block(
                kind="paragraph",
                text="Hello",
                meta={
                    "pdf": {"rects": [(0, 72.0, 100.0, 300.0, 112.0)], "size": 10.0, "kind": "body"}
                },
            ),
            Block(
                kind="paragraph",
                text="| a | b |",
                meta={
                    "table": True,
                    "pdf": {
                        "rects": [(0, 72.0, 200.0, 300.0, 260.0)],
                        "size": 10.0,
                        "kind": "table",
                    },
                },
            ),
            Block(
                kind="paragraph",
                text="⟦CODE_1⟧",
                meta={
                    "pdf": {"rects": [(0, 72.0, 300.0, 300.0, 340.0)], "size": 10.0, "kind": "code"}
                },
            ),
            Block(kind="paragraph", text="No geometry at all"),
        ]
    )


def test_tables_code_and_geometryless_blocks_are_kept_as_source():
    items, kept = placements(_doc_with_geometry(), {0: "Xin chào", 1: "x", 2: "x", 3: "x"})
    assert [item.text for item in items] == ["Xin chào"]
    assert kept == 3


def test_a_block_without_a_translation_keeps_its_source_text():
    items, kept = placements(_doc_with_geometry(), {})
    assert items == []
    assert kept == 4


# --------------------------------- fonts ------------------------------------


def test_an_explicit_font_path_wins(tmp_path):
    font = tmp_path / "mine.ttf"
    font.write_bytes(b"\x00\x01")
    assert resolve_font(font) == font


def test_no_available_font_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(pdf_output, "_FONT_CANDIDATES", ())
    monkeypatch.delenv("PDF_FONT_PATH", raising=False)
    with pytest.raises(PdfOutputError, match="PDF_FONT_PATH"):
        resolve_font()


# --------------------------------- rebuild ----------------------------------


def test_the_rebuilt_pdf_replaces_the_text_and_keeps_every_page(tmp_path, unicode_font):
    from pdfdoc import write_pdf

    pymupdf = pytest.importorskip("pymupdf")
    src = write_pdf(
        tmp_path / "in.pdf",
        "# Doc Title\n\nHello world from the source page.\n",
    )
    doc = load_document(src)
    texts = {
        index: ("Tiêu đề tài liệu" if block.kind == "heading" else "Nội dung đã dịch")
        for index, block in enumerate(doc.blocks)
    }
    out = tmp_path / "out.pdf"
    stats = rebuild_pdf(src, doc, texts, out, font_path=unicode_font)

    assert stats.placed == len(doc.blocks)
    assert stats.failures == []
    with pymupdf.open(out) as rebuilt, pymupdf.open(src) as original:
        assert rebuilt.page_count == original.page_count
        text = rebuilt[0].get_text()
    # The embedded font's space extracts as a non-breaking space; normalise it
    # before comparing words rather than asserting on the encoder's choice.
    text = text.replace("\xa0", " ")
    assert "Nội dung đã dịch" in text
    assert "Hello world" not in text


def test_an_untranslated_document_rebuilds_nothing(tmp_path, unicode_font):
    from pdfdoc import write_pdf

    src = write_pdf(tmp_path / "in.pdf", "Nothing was translated here.\n")
    doc = load_document(src)
    stats = rebuild_pdf(src, doc, {}, tmp_path / "out.pdf", font_path=unicode_font)
    assert stats.placed == 0
    assert stats.kept >= 1
