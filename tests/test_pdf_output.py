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
    """Just what the assembler reads off a chunk: its id, its blocks, and the
    source text it was built from — one piece per block, which is what the
    translator was given and what the paragraphs are aligned against."""

    def __init__(self, cid: str, block_refs: list[int], text: str = "") -> None:
        self.id = cid
        self.block_refs = block_refs
        self.text = text


@pytest.fixture(scope="module")
def unicode_font():
    """A Vietnamese-capable TTF, or a skip when this machine has none."""
    try:
        return resolve_font()
    except PdfOutputError as exc:  # pragma: no cover - depends on the machine
        pytest.skip(str(exc))


# ------------------------- chunk -> block mapping ---------------------------


def test_paragraphs_map_to_their_own_blocks():
    c0 = _FakeChunk("c0", [0, 1], "S0\n\nS1")
    assert block_translations([c0], {"c0": "P0\n\nP1"}) == {0: "P0", 1: "P1"}


def test_the_overlapped_head_is_not_emitted_twice():
    """The earlier chunk owns the shared block, exactly as `assemble` decides."""
    c0 = _FakeChunk("c0", [0, 1], "S0\n\nS1")
    c1 = _FakeChunk("c1", [1, 2], "S1\n\nS2")  # re-covers block 1 as its head
    out = block_translations([c0, c1], {"c0": "P0\n\nP1", "c1": "P1\n\nP2"})
    assert out == {0: "P0", 1: "P1", 2: "P2"}


def test_blocks_the_translation_has_no_text_for_are_left_out():
    """Without the source text there is nothing to align on, so the paragraphs
    stay where their position puts them and the blocks beyond them are left
    out."""
    c0 = _FakeChunk("c0", [0, 1, 2])
    assert block_translations([c0], {"c0": "Only one paragraph"}) == {0: "Only one paragraph"}


def test_extra_paragraphs_stay_with_the_last_block():
    c0 = _FakeChunk("c0", [0, 1])
    out = block_translations([c0], {"c0": "P0\n\nP1a\n\nP1b"})
    assert out[0] == "P0"
    assert out[1] == "P1a\n\nP1b"


def test_a_chunk_without_a_translation_contributes_nothing():
    assert block_translations([_FakeChunk("c0", [0])], {}) == {}


def test_a_title_the_source_wrapped_over_two_blocks_keeps_them_together():
    """The translator returns one paragraph for a block that is half a title.

    Blocks 1 and 2 are one sentence — the second has no full stop of its own —
    and the paragraph carries both of their words. Block 1 owns it and block 2
    is left empty, because those words are already printed where block 1 is;
    what must *not* happen is the paragraphs after it moving up one block, which
    is what printed every later paragraph of the page in its neighbour's place.
    """
    c0 = _FakeChunk(
        "c0",
        [0, 1, 2, 3],
        "Introduction.\n\n1.7. Better Graph Data for Rapidly Changing\n\n"
        "Data Model\n\nThe rest of the chapter follows.",
    )
    out = block_translations(
        [c0],
        {
            "c0": "Giới thiệu.\n\n1.7. Dữ liệu đồ thị tốt hơn cho mô hình dữ liệu thay đổi "
            "nhanh chóng\n\nPhần còn lại của chương tiếp theo sau."
        },
    )
    assert out[0] == "Giới thiệu."
    assert out[1].startswith("1.7.")
    assert out[2] is None, "the half-title is covered by block 1's paragraph"
    assert out[3] == "Phần còn lại của chương tiếp theo sau."


def test_a_block_that_came_back_as_two_paragraphs_keeps_both():
    """The translator wraps one long entry over two paragraphs, and the second
    of them still belongs to that entry — its page number says so — rather than
    being pushed onto the entry below it."""
    c0 = _FakeChunk(
        "c0",
        [0, 1],
        "1.1. A … 3\n\n1.2. A very long entry title that goes on and on … 5",
    )
    out = block_translations(
        [c0],
        {"c0": "1.1. A … 3\n\n1.2. Một tiêu đề rất dài kéo dài mãi\n\nvà vẫn tiếp tục … 5"},
    )
    assert out[0] == "1.1. A … 3"
    assert out[1] == "1.2. Một tiêu đề rất dài kéo dài mãi\n\nvà vẫn tiếp tục … 5"


def test_contents_entries_keep_their_own_page_numbers():
    """A translator that drops one entry must not renumber the others.

    Every entry ends in its page number, and that number survives translation:
    it is the evidence that says which paragraph belongs to which entry. Ran on
    the book this was written for, reading them position by position left 37 of
    128 entries — the whole contents page — showing their neighbour's number.
    """
    c0 = _FakeChunk(
        "c0",
        [0, 1, 2, 3],
        "1.1. Why use RDF? … 3\n\n1.2. Who is this Book Written for? … 5\n\n"
        "1.3. Why is the PDF Free? … 5\n\n1.4. Book Software … 6",
    )
    out = block_translations(
        [c0],
        {
            "c0": "1.1. Tại sao dùng RDF? … 3\n\n1.2. Cuốn sách này dành cho ai? … 5\n\n"
            "1.4. Phần mềm đi kèm sách … 6"
        },
    )
    assert out[0].endswith("3")
    assert out[1].endswith("5")
    assert out[3].endswith("6")
    # The entry the translator left out keeps its English line rather than
    # borrowing the paragraph of the entry below it.
    assert 2 not in out


def test_a_chunk_missing_its_own_text_keeps_the_old_pairing():
    """Without the source text there is no evidence to align on, so the
    paragraphs stay where their position puts them."""
    c0 = _FakeChunk("c0", [0, 1])
    assert block_translations([c0], {"c0": "P0\n\nP1"}) == {0: "P0", 1: "P1"}


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
    items, kept, _ = placements(_doc_with_geometry(), {0: "Xin chào", 1: "x", 2: "x", 3: "x"})
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
    items, kept, _ = placements(_table_doc(), {0: TRANSLATED_TABLE})

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
    items, kept, _ = placements(_table_doc(), {0: "| Tính năng |\n| --- |\n| Máy chủ |"})
    assert items == [] and kept == 1


def test_a_translation_that_splits_a_cell_on_a_pipe_keeps_the_table():
    # A `|` inside a translated cell makes the Markdown parse report an extra
    # column; lining cells up by position would then print them into the wrong
    # boxes, so the whole table stays in the source language instead.
    items, kept, _ = placements(_table_doc(), {0: "| a | b |\n| --- | --- |\n| c | d | e |"})
    assert items == [] and kept == 1


def test_a_table_without_a_translation_is_kept():
    items, kept, _ = placements(_table_doc(), {})
    assert items == [] and kept == 1


def test_a_cell_without_translation_is_not_printed():
    items, _kept, _ = placements(_table_doc(), {0: "| a |  |\n| --- | --- |\n| c | d |"})
    assert [item.text for item in items] == ["a", "c", "d"]
    # The cell we skip is not redacted either: its source text has to survive.
    assert [item.redact for item in items][1] == [(40.0, 146.0, 270.0, 172.0)]


def test_a_cell_holding_a_code_placeholder_prints_it_as_text():
    items, kept, _ = placements(_table_doc(), {0: "| ⟦CODE_1⟧ | b |\n| --- | --- |\n| c | d |"})
    assert kept == 0 and items[0].text == "⟦CODE_1⟧"


def test_a_cell_without_a_box_is_left_alone():
    doc = _table_doc()
    doc.blocks[0].meta["pdf"]["cells"][1][1] = None
    items, _kept, _ = placements(doc, {0: "| a | b |\n| --- | --- |\n| c | d |"})
    assert [item.text for item in items] == ["a", "b", "c"]


def test_an_overlapping_cell_is_printed_once():
    doc = _table_doc()
    cells = doc.blocks[0].meta["pdf"]["cells"]
    cells[1][0] = list(cells[0][0])  # a row merge pdfplumber did not collapse
    items, _kept, _ = placements(doc, {0: "| a | b |\n| --- | --- |\n| c | d |"})
    assert [item.text for item in items] == ["a", "b", "d"]


def test_a_block_without_a_translation_keeps_its_source_text():
    items, kept, _ = placements(_doc_with_geometry(), {})
    assert items == []
    assert kept == 4


def _wrapped_title(top: float = 200.0, gap: float = 12.0) -> Document:
    """Two blocks the source wrapped one title over: a plain line and its rest."""

    def line(text: str, at: float) -> Block:
        return Block(
            kind="paragraph",
            text=text,
            meta={
                "pdf": {
                    "rects": [(0, 72.0, at, 300.0, at + 12.0)],
                    "size": 10.0,
                    "kind": "body",
                }
            },
        )

    return Document(blocks=[line("Rapidly Changing Data", top), line("Model", top + gap)])


def test_a_merged_block_gives_its_lines_to_the_paragraph_printed_above_it():
    """One block, one paragraph: the second line's English words are covered by
    that paragraph, so they are cleared and its box grows over them — otherwise
    the translation prints over the English line it replaces."""
    items, kept, merged = placements(
        _wrapped_title(), {0: "Mô hình dữ liệu thay đổi nhanh", 1: None}
    )
    assert merged == 1
    assert kept == 0
    assert len(items) == 1
    assert items[0].box == (72.0, 200.0, 300.0, 224.0)
    assert items[0].redact == [(72.0, 200.0, 300.0, 212.0), (72.0, 212.0, 300.0, 224.0)]


def test_a_merged_block_the_paragraph_does_not_reach_keeps_its_source_text():
    """The paragraph was printed high up the page, so this block is not part of
    that sentence: clearing it would delete text nothing replaced."""
    items, kept, merged = placements(_wrapped_title(gap=120.0), {0: "Mô hình", 1: None})
    assert merged == 0
    assert kept == 1
    assert len(items) == 1
    assert items[0].redact == [(72.0, 200.0, 300.0, 212.0)]


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
    assert "Nội dung đã dịch" in text
    assert "Hello world" not in text


def test_copied_text_keeps_real_spaces_and_hyphens(tmp_path, unicode_font):
    """What is copied out of a rebuilt page is part of the output.

    PyMuPDF builds an inserted font's ToUnicode map by inverting the font's
    cmap; the space glyph carries both U+0020 and U+00A0 there, and the hyphen
    both U+002D and U+00AD, so the higher code point used to win and a copied
    page came back full of non-breaking spaces and soft hyphens. The rebuild
    repairs the map of the fonts it inserts.
    """
    from pdfdoc import write_pdf

    pymupdf = pytest.importorskip("pymupdf")
    src = write_pdf(tmp_path / "in.pdf", "Hello world from the source page.\n")
    doc = load_document(src)
    translated = "Bản dịch Eocortex ST - một cột, có gạch nối."
    texts = {index: translated for index in range(len(doc.blocks))}

    out = tmp_path / "out.pdf"
    stats = rebuild_pdf(src, doc, texts, out, font_path=unicode_font)

    assert stats.repaired_maps >= 1
    with pymupdf.open(out) as rebuilt:
        copied = rebuilt[0].get_text()
    assert "\xa0" not in copied, "dấu cách bị mã hoá thành NO-BREAK SPACE"
    assert "\xad" not in copied, "gạch nối bị mã hoá thành SOFT HYPHEN"
    assert "Eocortex ST - một cột" in " ".join(copied.split())


def test_the_map_repair_only_rewrites_standalone_entries():
    """The rewrite must never cross a line break.

    The entry regex used to allow any whitespace between the two codes, so a
    line ending in a hex code followed by a line starting with ``<00a0>``
    fused into one malformed entry. The saved map then parsed as garbage and
    extraction returned U+00C3 for glyphs whose input was correct — every
    ``JavaScript`` came back as ``ÃavaScript``. A bfrange head whose ``hi``
    is ``00ad`` was rewritten too, leaving ``hi < lo``.
    """
    stream = "\n".join(
        [
            "<0003> <00a0>",  # standalone nbsp target: rewritten
            "<0010> <00ad>",  # standalone soft-hyphen target: rewritten
            "<009f> <2126>",  # next line starts with <00a0>: must not fuse
            "<00a0> <00e6>",  # 00a0 as source, other target: untouched
            "<00ac> <00ad> <00c3>",  # bfrange head with hi=00ad: untouched
            "<0011> <00c3>",  # the U+00C3 mapping itself: untouched
        ]
    )

    fixed = pdf_output._fix_tounicode_targets(stream)

    assert fixed.splitlines() == [
        "<0003> <0020>",
        "<0010> <002d>",
        "<009f> <2126>",
        "<00a0> <00e6>",
        "<00ac> <00ad> <00c3>",
        "<0011> <00c3>",
    ]


def test_the_source_pages_own_font_map_is_not_repaired(tmp_path, unicode_font):
    """A block kept in the source language must copy out as it always did.

    The repair is applied to the fonts the rebuild *inserts*; rewriting the
    source document's own maps would change text it never touched.
    """
    from pdfdoc import write_pdf

    pymupdf = pytest.importorskip("pymupdf")
    first = (
        "First paragraph gets translated here. It holds enough words to wrap "
        "over several lines, which is what tells the extractor where one "
        "paragraph ends."
    )
    kept = (
        "Second paragraph stays in the source language. It wraps over several "
        "lines for exactly the same reason as the first one does."
    )
    src = write_pdf(tmp_path / "in.pdf", f"{first}\n\n{kept}\n")
    doc = load_document(src)
    assert len(doc.blocks) == 2
    texts = {0: "Đoạn đầu đã dịch."}  # block 1 has no translation: it stays put

    out = tmp_path / "out.pdf"
    rebuild_pdf(src, doc, texts, out, font_path=unicode_font)

    with pymupdf.open(src) as original, pymupdf.open(out) as rebuilt:
        # The page wraps its own lines, so compare the words, not the breaks.
        before = " ".join(original[0].get_text().split())
        after = " ".join(rebuilt[0].get_text().split())
    assert kept in before
    assert kept in after, "chữ nguồn phải copy ra y như trước khi dựng lại"
    assert "Đoạn đầu đã dịch" in after
    assert "First paragraph" not in after


def test_an_untranslated_document_rebuilds_nothing(tmp_path, unicode_font):
    from pdfdoc import write_pdf

    src = write_pdf(tmp_path / "in.pdf", "Nothing was translated here.\n")
    doc = load_document(src)
    stats = rebuild_pdf(src, doc, {}, tmp_path / "out.pdf", font_path=unicode_font)
    assert stats.placed == 0
    assert stats.kept >= 1


# ---------------------------------- cells -----------------------------------


def _table_pdf(tmp_path):
    """One page holding nothing but a ruled 2x2 table.

    The table is two columns wide because a single-column region is not a table
    at all as far as the ingest is concerned, and two rows because a lone row has
    no header to be read against.
    """
    from pdfdoc import write_table_pdf

    return write_table_pdf(
        tmp_path / "table.pdf",
        [["Feature", "State"], ["Servers", "5"]],
        col_widths=[240.0, 70.0],
    )


def test_a_cell_that_cannot_fit_keeps_its_source_text(tmp_path, unicode_font):
    pymupdf = pytest.importorskip("pymupdf")
    src = _table_pdf(tmp_path)
    doc = load_document(src)
    # Long enough that no size down to the 70% floor can wrap it into the two
    # lines the cell's 17 pt row has room for.
    too_long = "Một trăm hai mươi ký tự dịch rất dài " * 6
    texts = {0: f"| Đặc điểm | Trạng thái |\n| --- | --- |\n| {too_long} | 5 |"}

    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.cells == 3 and stats.cells_kept == 1
    assert stats.failures == []
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        text = rebuilt[0].get_text()
        assert "Servers" in text
        assert "Đặc điểm" in text


def test_an_empty_source_cell_that_cannot_take_the_translation_is_not_a_failure(unicode_font):
    """A cell the source left empty has nothing to lose.

    pdfplumber reports a sliver box (1.4 pt wide) where a merged column
    collapsed, so a translation that cannot be printed there would otherwise be
    reported as a cell the rebuild dropped — while the source never had any text
    in it to drop.
    """
    pymupdf = pytest.importorskip("pymupdf")
    page = pymupdf.open().new_page()
    item = pdf_output._Placement(
        page=0,
        redact=[(147.9, 73.4, 149.3, 94.0)],
        box=(147.9, 73.4, 149.3, 94.0),
        size=10.0,
        text="VI:",
        kind="cell",
        align="left",
        source_text="",
    )
    assert pdf_output._insert_cell(page, item, pymupdf, unicode_font) == "source"


def test_a_cell_does_not_spill_below_the_page(tmp_path, unicode_font):
    """Không có nhánh 'dùng nốt chỗ tới lề dưới' cho ô: chữ phải nằm trong ô."""
    pymupdf = pytest.importorskip("pymupdf")
    src = _table_pdf(tmp_path)
    doc = load_document(src)
    boxes = [cell[:4] for row in doc.blocks[0].meta["pdf"]["cells"] for cell in row]
    out = tmp_path / "out.pdf"
    stats = rebuild_pdf(
        src,
        doc,
        {0: "| Đặc điểm | Trạng thái |\n| --- | --- |\n| Máy chủ | 5 |"},
        out,
        font_path=unicode_font,
    )

    assert stats.cells == 4 and stats.cells_kept == 0
    with pymupdf.open(out) as rebuilt:
        for word in rebuilt[0].get_text("words"):
            x0, top, x1, bottom = word[:4]
            assert any(
                x0 >= bx0 - 1 and x1 <= bx2 + 1 and top >= btop - 1 and bottom <= bbottom + 1
                for bx0, btop, bx2, bbottom in boxes
            ), f"chữ nằm ngoài ô: {word[4]!r}"


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
    items, _, _ = placements(
        _styled_doc(),
        {0: "Giới thiệu về Eocortex … 4", 1: "một\nhai", 2: "Dòng ngắn.", 3: "Dòng khác."},
        match_style=False,
    )
    assert [item.text for item in items][1] == "• một\n• hai"


def test_a_contents_entry_is_split_for_its_leader_and_number():
    items, _, _ = placements(_styled_doc(), {0: "Giới thiệu về Eocortex … 4"}, match_style=False)
    assert items[0].toc == ("Giới thiệu về Eocortex", "4")
    assert items[0].text == "Giới thiệu về Eocortex … 4"


def test_a_short_line_may_use_the_column_next_to_it():
    """A translation of a short line is longer than the English one far more
    often than not, and shrinking the type is what that otherwise costs."""
    items, _, _ = placements(
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
    items, _, _ = placements(
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
    items, _, _ = placements(doc, {0: "Tiêu đề tài liệu", 1: "Thân bài."}, match_style=False)
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
    items, _, _ = placements(
        _styled_doc(), {2: "Dòng ngắn."}, default_font=pinned, match_style=False
    )
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


# --------------------------- a whole table page -----------------------------


def test_a_real_table_page_is_rebuilt_with_its_cells_translated(tmp_path, unicode_font):
    """The end-to-end shape of the feature: a ruled table page, rebuilt.

    Every cell of this table fits its translation, so nothing falls back — the
    fallback is `test_a_cell_that_cannot_fit_keeps_its_source_text`'s subject.
    What is checked here is that the grid survives (the rules are not redacted,
    and no glyph lands outside a cell) and that the source words are gone.
    """
    from pdfdoc import write_table_pdf

    pymupdf = pytest.importorskip("pymupdf")
    src = write_table_pdf(
        tmp_path / "table.pdf",
        [["Feature", "State"], ["Servers", "5"], ["Backup", "Yes"]],
        col_widths=[220.0, 120.0],
    )
    doc = load_document(src)
    table = next(block for block in doc.blocks if block.meta.get("table"))
    cells = table.meta["pdf"]["cells"]
    assert [len(row) for row in cells] == [2, 2, 2]

    # A stand-in translation: every cell prefixed, the shape of the grid kept.
    source_rows = [line for line in table.text.splitlines() if not set(line) <= set("|- ")]
    translated = "\n".join(
        "| " + " | ".join(f"VI:{cell}" for cell in row.strip("|").split(" | ")) + " |"
        for row in source_rows
    )
    texts = {
        index: (translated if block is table else block.text)
        for index, block in enumerate(doc.blocks)
    }

    out = tmp_path / "out.pdf"
    stats = rebuild_pdf(src, doc, texts, out, font_path=unicode_font)
    assert stats.cells == 6 and stats.cells_kept == 0 and stats.failures == []

    with pymupdf.open(src) as original, pymupdf.open(out) as rebuilt:
        assert len(rebuilt[0].get_drawings()) == len(original[0].get_drawings()), "lưới còn nguyên"
        # The extractor breaks some cells after the colon; that is a choice of
        # the extraction, not text on the page, so normalise it before comparing.
        text = rebuilt[0].get_text().replace("VI: ", "VI:")
        assert "\xa0" not in text
        assert "VI:Feature" in text and "VI:Yes" in text
        assert "Feature" not in text.replace("VI:Feature", "")
        boxes = [cell[:4] for row in cells for cell in row]
        for word in rebuilt[0].get_text("words"):
            x0, top, x1, bottom = word[:4]
            assert any(
                x0 >= bx0 - 1 and x1 <= bx2 + 1 and top >= btop - 1 and bottom <= bbottom + 1
                for bx0, btop, bx2, bbottom in boxes
            ), f"chữ nằm ngoài ô: {word[4]!r}"


# ------------------------------- emphasis -----------------------------------


def test_plain_text_drops_emphasis_tokens():
    """The printed page never carries a token, whatever the block's kind."""
    assert plain_text("a ⟦BOLD⟧b⟦/BOLD⟧ c", "body") == "a b c"
    assert plain_text("⟦BOLD⟧a⟦/BOLD⟧b", "heading") == "ab"
    assert plain_text("- ⟦ITALIC⟧a⟦/ITALIC⟧", "list") == "a"


def _emphasis_doc(source_text: str, *, kind: str = "body") -> Document:
    """One block whose source line mixed styles, with the geometry to print it."""
    return Document(
        blocks=[
            Block(
                kind="paragraph",
                text=source_text,
                meta={
                    "pdf": {
                        "rects": [(0, 72.0, 100.0, 300.0, 112.0)],
                        "size": 10.0,
                        "kind": kind,
                        "font": "CMR12",
                        "bold": False,
                        "italic": False,
                        "color": (0.0, 0.0, 0.0),
                        "markers": [],
                    }
                },
            )
        ]
    )


def test_a_body_block_with_balanced_tokens_gets_runs():
    """The runs come from the translation *with* its tokens: stripping first
    would leave one plain run and lose the emphasis the tokens describe."""
    doc = _emphasis_doc("A ⟦BOLD⟧B⟦/BOLD⟧ C")
    items, _, _ = placements(doc, {0: "A ⟦BOLD⟧B⟦/BOLD⟧ C"}, match_style=False)

    assert items[0].runs == [("A ", False, False), ("B", True, False), (" C", False, False)]
    assert items[0].text == "A B C"
    assert items[0].face == "CMR12"


def test_a_block_whose_translation_lost_a_token_gets_no_runs():
    """Guard ở tầng dựng: lệch token thì in thường, và im lặng (lỗi ghi ở tầng
    pipeline, một lần cho cả chunk)."""
    doc = _emphasis_doc("A ⟦BOLD⟧B⟦/BOLD⟧ C")
    items, _, _ = placements(doc, {0: "A B ⟦BOLD⟧C"}, match_style=False)

    assert items[0].runs is None
    assert "⟦" not in items[0].text


def test_a_block_whose_source_had_no_tokens_gets_no_runs():
    """Model tự thêm cặp token: nguồn không nhấn mạnh thì không in nhấn mạnh."""
    doc = _emphasis_doc("A B C")
    items, _, _ = placements(doc, {0: "A ⟦BOLD⟧B⟦/BOLD⟧ C"}, match_style=False)

    assert items[0].runs is None
    assert items[0].text == "A B C"


def test_a_list_block_keeps_one_face_and_its_source_markers():
    """V1 chỉ in run cho block thân bài và heading.

    Một mục danh sách còn phải nhận lại dấu đầu dòng của nguồn (``_marked_text``)
    và dấu ``-`` của Markdown; đi đường run sẽ bỏ qua cả hai.
    """
    doc = _emphasis_doc("- ⟦BOLD⟧A⟦/BOLD⟧ B", kind="list")
    doc.blocks[0].meta["pdf"]["markers"] = ["•"]
    items, _, _ = placements(doc, {0: "- ⟦BOLD⟧A⟦/BOLD⟧ B"}, match_style=False)

    assert items[0].runs is None
    assert items[0].text == "• A B"


def test_tokens_inside_a_cell_are_stripped_before_the_cell_is_split():
    doc = _table_doc()
    table = "| ⟦BOLD⟧a⟦/BOLD⟧ | b |\n| --- | --- |\n| c | d |"
    items, _, _ = placements(doc, {0: table}, match_style=False)

    assert [item.text for item in items] == ["a", "b", "c", "d"]
    assert all("⟦" not in item.text for item in items)


def test_a_heading_block_with_tokens_does_not_print_its_hashes():
    """``plain_text`` strips the ``#``, so the run path must strip it too."""
    doc = _emphasis_doc("# ⟦BOLD⟧A⟦/BOLD⟧ B", kind="heading")
    items, _, _ = placements(doc, {0: "# ⟦BOLD⟧A⟦/BOLD⟧ B"}, match_style=False)

    assert items[0].runs == [("A", True, False), (" B", False, False)]
    assert "#" not in items[0].text


# ------------------------- emphasis: printing the runs ----------------------


def _spans(pdf_path) -> list[tuple[str, int, str]]:
    """``(text, flags, font)`` của mọi span trên trang đầu.

    Bit 16 là đậm, bit 2 là nghiêng. Cờ này sống sót khi nhúng bằng ``fontfile``
    **chỉ khi mỗi mặt có resource name riêng** — trùng tên thì PyMuPDF in bằng
    mặt đăng ký trước đó, nên `_resource_name` băm theo đường dẫn tệp font.
    """
    import pymupdf

    with pymupdf.open(pdf_path) as rebuilt:
        return [
            (span["text"], span["flags"], span["font"])
            for block in rebuilt[0].get_text("dict")["blocks"]
            if block.get("type") == 0
            for line in block["lines"]
            for span in line["spans"]
        ]


def test_a_bold_span_is_printed_bold_and_the_rest_plain(tmp_path, unicode_font):
    """Đã kiểm bằng tay: Times New Roman cho flags 4 (thường) / 20 (đậm)."""
    from pdfdoc import write_styled_pdf

    src = write_styled_pdf(tmp_path / "in.pdf", "Alpha ", "Beta", " Gamma")
    doc = load_document(src)
    assert "⟦BOLD⟧" in doc.blocks[0].text, "ingest phải phát token cho dòng trộn kiểu"
    rebuild_pdf(
        src, doc, {0: "Alpha ⟦BOLD⟧Beta⟦/BOLD⟧ Gamma"}, tmp_path / "out.pdf", font_path=unicode_font
    )

    spans = _spans(tmp_path / "out.pdf")
    assert any("Beta" in text and flags & 16 for text, flags, _ in spans), spans
    assert any("Alpha" in text and not flags & 16 for text, flags, _ in spans)
    assert "⟦" not in "".join(text for text, _, _ in spans), "token không được lộ lên trang"


def test_the_bold_run_uses_a_second_face_not_the_plain_one(tmp_path, unicode_font):
    """Hai lần in phải cho ra hai font **khác nhau** khi đọc lại.

    Nếu chúng trùng tên, một trang thật có hai mặt cùng tên sẽ âm thầm in bằng
    mặt thứ nhất, và nhấn mạnh biến mất mà không có lỗi nào.
    """
    from pdfdoc import write_styled_pdf

    src = write_styled_pdf(tmp_path / "in.pdf", "Alpha ", "Beta", " Gamma")
    doc = load_document(src)
    rebuild_pdf(
        src, doc, {0: "Alpha ⟦BOLD⟧Beta⟦/BOLD⟧ Gamma"}, tmp_path / "out.pdf", font_path=unicode_font
    )

    spans = _spans(tmp_path / "out.pdf")
    bold = {font for text, flags, font in spans if "Beta" in text and flags & 16}
    plain = {font for text, flags, font in spans if "Alpha" in text and not flags & 16}
    assert bold and plain and bold.isdisjoint(plain), spans


def test_a_missing_local_bold_face_loses_the_emphasis_but_not_the_text(
    tmp_path, unicode_font, monkeypatch
):
    """Máy không có mặt đậm cục bộ: mất nhấn mạnh, chữ còn nguyên, không raise."""
    from pdfdoc import write_styled_pdf

    monkeypatch.setattr(pdf_output, "match_font", lambda *a, **k: None)
    src = write_styled_pdf(tmp_path / "in.pdf", "Alpha ", "Beta", " Gamma")
    doc = load_document(src)
    stats = rebuild_pdf(
        src, doc, {0: "Alpha ⟦BOLD⟧Beta⟦/BOLD⟧ Gamma"}, tmp_path / "out.pdf", font_path=unicode_font
    )

    spans = _spans(tmp_path / "out.pdf")
    assert stats.failures == []
    assert any("Beta" in text for text, _, _ in spans), "chữ không được mất"
    assert not any(flags & 16 for _, flags, _ in spans), "không có mặt đậm thì không có cờ đậm"
    assert "⟦" not in "".join(text for text, _, _ in spans)


def test_a_long_block_continues_on_a_new_page_at_the_same_size(tmp_path, unicode_font):
    """Chữ dài hơn hộp thì **tràn sang trang sau ở đúng cỡ gốc**.

    `rebuild_pdf` xoá chữ nguồn trước khi in, nên một block in hụt là mất chữ
    chứ không để lại chữ nguồn; còn thu nhỏ cỡ chữ cho vừa hộp thì 10 pt thành
    5.5 pt, đè lên block dưới. Cách đúng là in phần vừa hộp ở cỡ gốc rồi nối
    tiếp phần còn lại sang trang mới, cũng ở cỡ gốc.
    """
    import pymupdf
    from pdfdoc import write_styled_pdf

    src = write_styled_pdf(tmp_path / "in.pdf", "Alpha ", "Beta", " Gamma")
    doc = load_document(src)
    source_size = doc.blocks[0].meta["pdf"]["size"]
    long_text = "Alpha ⟦BOLD⟧" + "rất dài " * 12 + "⟦/BOLD⟧ Gamma"
    stats = rebuild_pdf(src, doc, {0: long_text}, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == [], "chữ dài vẫn phải in được, chỉ là tràn sang trang sau"
    assert stats.continued >= 1, "phần không vừa hộp phải được in tiếp ở trang mới"
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert rebuilt.page_count == stats.pages
        first, second = rebuilt[0], rebuilt[1]
        size = first.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["size"]
        assert abs(size - source_size) < 0.5, f"chữ phải giữ nguyên cỡ gốc: {size} vs {source_size}"
        words = first.get_text("words") + second.get_text("words")
        height = first.rect.height
        assert all(word[1] >= 0 and word[3] <= height for word in words), (
            "chữ không được rời khỏi trang"
        )
        assert not any("⟦" in word[4] for word in words), "token không được lộ lên trang"
        assert "rất" in first.get_text() and "rất dài" in second.get_text(), (
            "chữ phải liền mạch qua hai trang, không mất chữ"
        )


def test_spilling_blocks_share_one_overleaf_page(tmp_path, unicode_font):
    """Hai block cùng tràn vào **một** trang chèn dùng chung, không phải mỗi
    block một trang gần như trống — 48 trang như vậy từng làm cuốn 69 trang
    thành 117.

    Đồng thời đó là test hồi quy cho refetch: sau khi block đầu tràn, block thứ
    hai vẫn phải in trên **trang nguồn** (index của trang nguồn không dịch chuyển
    khi trang chèn được thêm vào), ngược với bản cũ refetch lấy ngay trang chèn
    cuối và để một lỗ trống trên trang nguồn.
    """
    import pymupdf
    from pdfdoc import write_pdf

    # Đoạn phải wrap qua vài dòng: extractor tách block theo khoảng cách dòng,
    # và một đoạn đúng một dòng sẽ tự biến ước lượng đó thành norm (xem pdfdoc).
    _PARA_ONE = (
        "Alpha first paragraph wraps over several lines in the source, which "
        "is what marks it as its own block for the extractor to read."
    )
    _PARA_TWO = (
        "Beta second paragraph also wraps over several lines in the source, "
        "so the blank line between the two is clearly larger than the norm."
    )

    src = write_pdf(tmp_path / "in.pdf", f"{_PARA_ONE}\n\n{_PARA_TWO}\n")
    doc = load_document(src)
    assert len(doc.blocks) == 2, "cần đúng hai block trên một trang nguồn"
    texts = {0: "Alpha mở rộng " + "lời " * 80, 1: "Beta rất dài " + "câu " * 600}
    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == []
    assert stats.continued == 1, "hai block tràn phải dùng chung một trang chèn"
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert rebuilt.page_count == stats.pages == 2, "chỉ thêm một trang chèn"
        source_text = rebuilt[0].get_text()
        cont_text = rebuilt[1].get_text()
        assert "Beta" in source_text, "block dưới phải in trên trang nguồn sau khi block đầu tràn"
        assert "Alpha" in source_text, "đầu block đầu vẫn ở trang nguồn"
        # Hai tail xếp chồng theo thứ tự block trên cột chèn dùng chung.
        assert "lời" in cont_text and "câu" in cont_text, "cả hai tail phải có mặt trên trang chèn"
        assert cont_text.index("lời") < cont_text.index("câu"), (
            "tail phải giữ đúng thứ tự block"
        )
