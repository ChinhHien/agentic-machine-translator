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


def test_cells_and_code_and_geometryless_blocks_are_kept_as_source():
    items, kept = placements(_doc_with_geometry(), {0: "Xin chào", 1: "x", 2: "x", 3: "x"})
    assert [item.text for item in items] == ["Xin chào"]
    assert kept == 3  # the table has no cell boxes here, and code is never translated


def _table_block() -> Block:
    """A 2x2 table with the geometry and cell boxes the ingest records."""
    return Block(
        kind="paragraph",
        text="| Feature | State |\n| --- | --- |\n| Servers | 5 |",
        meta={
            "table": True,
            "pdf": {
                "rects": [(0, 40.0, 120.0, 500.0, 172.0)],
                "size": 0.0,
                "kind": "table",
                "font": "CMR12",
                "bold": False,
                "italic": False,
                "color": (0.0, 0.0, 0.0),
                "markers": [],
                "cells": [
                    [
                        [40.0, 120.0, 270.0, 146.0, 2.0, 10.0, "left", True, False],
                        [270.0, 120.0, 500.0, 146.0, 2.0, 10.0, "center", True, False],
                    ],
                    [
                        [40.0, 146.0, 270.0, 172.0, 2.0, 9.0, "left", False, False],
                        [270.0, 146.0, 500.0, 172.0, 2.0, 9.0, "right", False, False],
                    ],
                ],
            },
        },
    )


def _table_doc() -> Document:
    return Document(blocks=[_table_block()])


TRANSLATED_TABLE = "| Tính năng | Trạng thái |\n| --- | --- |\n| Máy chủ | 5 |"


def test_table_cells_are_placed_cell_by_cell():
    items, kept = placements(_table_doc(), {0: TRANSLATED_TABLE})

    assert kept == 0
    assert [item.text for item in items] == ["Tính năng", "Trạng thái", "Máy chủ", "5"]
    assert [item.box for item in items] == [
        (42.0, 122.0, 268.0, 144.0),
        (272.0, 122.0, 498.0, 144.0),
        (42.0, 148.0, 268.0, 170.0),
        (272.0, 148.0, 498.0, 170.0),
    ]
    assert [item.size for item in items] == [10.0, 10.0, 9.0, 9.0]
    assert [item.align for item in items] == ["left", "center", "left", "right"]
    assert [item.source_text for item in items] == ["Feature", "State", "Servers", "5"]
    assert items[0].redact == [(40.0, 120.0, 270.0, 146.0)]
    assert all(item.kind == "cell" for item in items)


def test_a_translated_table_with_a_different_shape_keeps_the_source_table():
    items, kept = placements(_table_doc(), {0: "| Tính năng |\n| --- |\n| Máy chủ |"})
    assert items == [] and kept == 1


def test_a_translation_that_splits_a_cell_on_a_pipe_keeps_the_table():
    # A `|` inside a translated cell makes the Markdown parse report an extra
    # column; lining cells up by position would then print them into the wrong
    # boxes, so the whole table stays in the source language instead.
    items, kept = placements(_table_doc(), {0: "| a | b |\n| --- | --- |\n| c | d | e |"})
    assert items == [] and kept == 1


def test_a_table_without_a_translation_is_kept():
    items, kept = placements(_table_doc(), {})
    assert items == [] and kept == 1


def test_a_cell_without_translation_is_not_printed():
    items, _kept = placements(_table_doc(), {0: "| a |  |\n| --- | --- |\n| c | d |"})
    assert [item.text for item in items] == ["a", "c", "d"]
    # The cell we skip is not redacted either: its source text has to survive.
    assert [item.redact for item in items][1] == [(40.0, 146.0, 270.0, 172.0)]


def test_a_cell_holding_a_code_placeholder_prints_it_as_text():
    items, kept = placements(_table_doc(), {0: "| ⟦CODE_1⟧ | b |\n| --- | --- |\n| c | d |"})
    assert kept == 0 and items[0].text == "⟦CODE_1⟧"


def test_a_cell_without_a_box_is_left_alone():
    doc = _table_doc()
    doc.blocks[0].meta["pdf"]["cells"][1][1] = None
    items, _kept = placements(doc, {0: "| a | b |\n| --- | --- |\n| c | d |"})
    assert [item.text for item in items] == ["a", "b", "c"]


def test_an_overlapping_cell_is_printed_once():
    doc = _table_doc()
    cells = doc.blocks[0].meta["pdf"]["cells"]
    cells[1][0] = list(cells[0][0])  # a row merge pdfplumber did not collapse
    items, _kept = placements(doc, {0: "| a | b |\n| --- | --- |\n| c | d |"})
    assert [item.text for item in items] == ["a", "b", "d"]


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


# ----------------------- markers, contents entries, boxes -------------------


def _styled_doc() -> Document:
    """Three blocks whose geometry and source face are already recorded."""

    def block(kind: str, text: str, rect: tuple, **geo: object) -> Block:
        return Block(
            kind=kind,
            text=text,
            meta={"pdf": {"rects": [rect], "size": 10.0, "kind": kind, **geo}},
        )

    return Document(
        blocks=[
            block("toc", "About Eocortex .......... 4", (0, 72.0, 100.0, 500.0, 112.0)),
            block(
                "list",
                "- một\n- hai",
                (0, 72.0, 200.0, 260.0, 240.0),
                markers=["•", "•"],
            ),
            block(
                "body",
                "Một dòng ngắn.",
                (0, 72.0, 300.0, 200.0, 340.0),
                font="Helvetica",
                bold=False,
                color=(0.0, 0.0, 0.0),
            ),
            block("body", "Một dòng khác.", (0, 72.0, 400.0, 520.0, 440.0)),
        ]
    )


def test_a_translated_list_gets_the_source_markers_back():
    items, _ = placements(
        _styled_doc(),
        {0: "Giới thiệu về Eocortex … 4", 1: "một\nhai", 2: "Dòng ngắn.", 3: "Dòng khác."},
        match_style=False,
    )
    assert [item.text for item in items][1] == "• một\n• hai"


def test_a_contents_entry_is_split_for_its_leader_and_number():
    items, _ = placements(_styled_doc(), {0: "Giới thiệu về Eocortex … 4"}, match_style=False)
    assert items[0].toc == ("Giới thiệu về Eocortex", "4")
    assert items[0].text == "Giới thiệu về Eocortex … 4"


def test_a_short_line_may_use_the_column_next_to_it():
    """A translation of a short line is longer than the English one far more
    often than not, and shrinking the type is what that otherwise costs."""
    items, _ = placements(
        _styled_doc(),
        {2: "Câu dịch dài hơn hẳn câu gốc.", 3: "x"},
        match_style=False,
    )
    short = next(item for item in items if item.text.startswith("Câu dịch"))
    assert short.box[2] == 520.0  # stretched to the column's right edge
    assert short.source_box[2] == 200.0  # the source line itself is remembered


def test_a_line_with_a_neighbour_beside_it_is_left_alone():
    doc = _styled_doc()
    doc.blocks.append(
        Block(
            kind="paragraph",
            text="Bên cạnh.",
            meta={
                "pdf": {"rects": [(0, 300.0, 300.0, 520.0, 340.0)], "size": 10.0, "kind": "body"}
            },
        )
    )
    items, _ = placements(
        doc, {2: "Câu dịch dài hơn hẳn câu gốc.", 4: "Bên cạnh."}, match_style=False
    )
    short = next(item for item in items if item.text.startswith("Câu dịch"))
    assert short.box[2] == 200.0


def test_a_centred_title_widens_both_ways_so_it_stays_centred():
    doc = Document(
        blocks=[
            Block(
                kind="heading",
                text="Tài liệu",
                meta={
                    "pdf": {
                        "rects": [(0, 240.0, 100.0, 360.0, 140.0)],
                        "size": 28.0,
                        "kind": "heading",
                    }
                },
            ),
            Block(
                kind="paragraph",
                text="Thân bài.",
                meta={
                    "pdf": {"rects": [(0, 60.0, 300.0, 540.0, 340.0)], "size": 10.0, "kind": "body"}
                },
            ),
        ]
    )
    items, _ = placements(doc, {0: "Tiêu đề tài liệu", 1: "Thân bài."}, match_style=False)
    item = next(item for item in items if item.kind == "heading")
    assert item.box[0] == 60.0 and item.box[2] == 540.0


# --------------------------------- fonts ------------------------------------


def test_a_family_key_ignores_weight_slant_and_subset_tags():
    assert pdf_output._font_key("Verdana-Bold") == "verdana"
    assert pdf_output._font_key("verdana regular") == "verdana"
    assert pdf_output._font_key("TimesNewRomanPSMT") == "timesnewroman"
    assert pdf_output._font_key("Segoe UI Semibold") == "segoeui"
    assert pdf_output._kind_of("timesnewroman") == "serif"
    assert pdf_output._kind_of("dejavusansmono") == "mono"
    assert pdf_output._kind_of("verdana") == "sans"


def test_the_source_family_is_used_when_it_is_installed(monkeypatch):
    monkeypatch.setattr(pdf_output, "_covers_vietnamese", lambda path: True)
    monkeypatch.setattr(
        pdf_output,
        "installed_fonts",
        lambda: {"verdana": {(False, False): "/f/verdana.ttf"}},
    )
    pdf_output.match_font.cache_clear()
    assert pdf_output.match_font("Verdana-Bold", True, False) == "/f/verdana.ttf"


def test_an_unknown_family_falls_back_to_one_of_the_same_kind(monkeypatch):
    monkeypatch.setattr(pdf_output, "_covers_vietnamese", lambda path: True)
    monkeypatch.setattr(
        pdf_output,
        "installed_fonts",
        lambda: {
            "arial": {(False, False): "/f/arial.ttf"},
            "timesnewroman": {(False, False): "/f/times.ttf"},
        },
    )
    pdf_output.match_font.cache_clear()
    assert pdf_output.match_font("Klingon", False, False) == "/f/arial.ttf"
    assert pdf_output.match_font("Georgia", False, False) == "/f/times.ttf", (
        "a serif source must not be replaced by a sans face"
    )


def test_a_pinned_font_is_not_second_guessed(tmp_path):
    """``--font-path``/``PDF_FONT_PATH`` means \"this face\", not \"a face like it\"."""
    pinned = tmp_path / "pinned.ttf"
    pinned.write_bytes(b"\x00\x01")
    items, _ = placements(_styled_doc(), {2: "Dòng ngắn."}, default_font=pinned, match_style=False)
    assert items[0].font_file == str(pinned)


# --------------------------- markers on a real page -------------------------


def test_the_rebuilt_pdf_prints_markers_and_contents_leaders(tmp_path, unicode_font):
    """The two things the source expressed by layout alone, end to end.

    The document is ASCII: the test renderer writes base-14 fonts, which cannot
    carry Vietnamese at all, so a translated sample would arrive here already
    mangled and prove nothing about the rebuild.
    """
    from pdfdoc import write_pdf

    pymupdf = pytest.importorskip("pymupdf")
    src = write_pdf(
        tmp_path / "in.pdf",
        "# Doc Title\n\n- item one\n- item two\n\nAbout Doc .......... 4\n\nBody text.\n",
    )
    doc = load_document(src)
    texts = {index: block.text for index, block in enumerate(doc.blocks)}
    out = tmp_path / "out.pdf"
    stats = rebuild_pdf(src, doc, texts, out, font_path=unicode_font)
    assert stats.failures == []

    with pymupdf.open(out) as rebuilt:
        blocks = [b for b in rebuilt[0].get_text("dict")["blocks"] if b.get("type") == 0]
    # The embedded font's encoder writes a non-breaking space for a space and a
    # soft hyphen for a hyphen; normalise the encoder's choices rather than the
    # behaviour under test.
    lines = {
        "".join(span["text"] for span in line["spans"])
        .replace("\xa0", " ")
        .replace("\xad", "-"): line
        for block in blocks
        for line in block["lines"]
    }
    assert "- item one" in lines and "- item two" in lines

    entry = next(text for text in lines if text.startswith("About Doc"))
    assert "....." in entry  # the leader the ingest collapsed is printed again
    assert entry.rstrip().endswith("4")
    toc = next(block for block in doc.blocks if (block.meta.get("pdf") or {}).get("kind") == "toc")
    right = toc.meta["pdf"]["rects"][0][3]
    assert lines[entry]["bbox"][2] >= right - 12  # the number sits at the margin
