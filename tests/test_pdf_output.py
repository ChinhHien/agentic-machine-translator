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


def test_a_merged_overlap_keeps_the_new_blocks_words_on_the_page():
    """The translator merged the replayed block with the first new block.

    The paragraph belongs to the replayed block, so nothing of this chunk owns
    it; blanking the new block instead would delete its translated words from
    the page. A repeat is visible, a hole is not.
    """
    c0 = _FakeChunk("c0", [0, 1], "A\n\nB")
    c1 = _FakeChunk("c1", [1, 2, 3], "B\n\nC\n\nD")
    out = block_translations([c0, c1], {"c0": "PA\n\nPB", "c1": "PB và PC\n\nPD"})
    assert out == {0: "PA", 1: "PB", 2: "PB và PC", 3: "PD"}


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


def _tight_triple() -> Document:
    """Three single-line blocks set tight: each line ends 2.3 pt before the
    next one starts, so the pad under every box hangs over the line below."""

    def line(text: str, top: float, bottom: float, x1: float) -> Block:
        return Block(
            kind="paragraph",
            text=text,
            meta={"pdf": {"rects": [(0, 72.0, top, x1, bottom)], "size": 10.0, "kind": "body"}},
        )

    return Document(
        blocks=[
            line("Backend Framework", 100.0, 112.0, 150.0),
            line("Frontend Framework", 114.3, 126.3, 500.0),
            line("Database", 128.6, 140.6, 500.0),
        ]
    )


def test_a_tight_set_line_widens_but_stops_at_the_line_below():
    """Both overlap decisions read the source line's own bottom, not the box's
    padded one. The pad hangs over the next line when the source sets its lines
    tight, and a box that counted that pad as room on its own line grew through
    the line below — the longer translation then printed over it — while the
    same pad refused the widening that would have kept the label on one line."""
    items, _, _ = placements(
        _tight_triple(),
        {0: "Khung máy chủ", 1: "Khung máy khách", 2: "Cơ sở dữ liệu"},
        match_style=False,
    )
    first = items[0]
    assert first.box[2] == 500.0  # widened into the free room to the right
    # source bottom 112.0 + the 6 pt pad, and no further: not grown to the line
    # after next (126.6), which would cover the line at 114.3.
    assert first.box[3] == 118.0


def test_a_block_over_the_page_break_gives_only_its_own_page_to_the_item():
    """A folded block whose lines run over the page break: the item printing
    these words lives on page 0, and page 1's coordinates — which start again
    near the top of *that* page — must not join its box, or the box spans a whole
    page and prints over everything already there. Page 1's lines are cleared
    where they stand, because their words were translated above."""
    doc = Document(
        blocks=[
            Block(
                kind="paragraph",
                text="First",
                meta={
                    "pdf": {"rects": [(0, 72.0, 100.0, 300.0, 112.0)], "size": 10.0, "kind": "body"}
                },
            ),
            Block(
                kind="paragraph",
                text="Rest",
                meta={
                    "pdf": {
                        "rects": [
                            (0, 72.0, 114.3, 300.0, 126.3),
                            (1, 72.0, 30.7, 300.0, 42.7),
                        ],
                        "size": 10.0,
                        "kind": "body",
                    }
                },
            ),
        ]
    )
    items, kept, merged = placements(doc, {0: "Đoạn đầu", 1: None})
    assert (kept, merged) == (0, 1)
    cover, clear = items
    # Only page 0's rects joined the box — not (72, 30.7, ...) from page 1.
    assert cover.box == (72.0, 100.0, 300.0, 126.3)
    assert cover.redact == [(72.0, 100.0, 300.0, 112.0), (72.0, 114.3, 300.0, 126.3)]
    assert (clear.page, clear.kind, clear.text) == (1, "clear", "")
    assert clear.redact == [(72.0, 30.7, 300.0, 42.7)]


def test_a_block_over_the_page_break_clears_and_resumes_where_it_stands():
    """A paragraph the source split over two pages: its translation starts in the
    first page's box and continues at the y the source continued at.

    The block's rects on page 1 were in ``redact`` but redaction runs on the
    first page only, so the English on page 1 stayed and the Vietnamese tail
    was carried into the margin above the running head. The lines on page 1 are
    now cleared where they are, and the tail is told where to resume.
    """
    doc = Document(
        blocks=[
            Block(
                kind="paragraph",
                text="Start of the paragraph",
                meta={
                    "pdf": {
                        "rects": [
                            (0, 72.0, 600.0, 500.0, 612.0),
                            (0, 72.0, 614.0, 500.0, 626.0),
                            (1, 72.0, 120.0, 500.0, 132.0),
                        ],
                        "size": 10.0,
                        "kind": "body",
                    }
                },
            ),
        ]
    )
    items, kept, _merged = placements(doc, {0: "Bắt đầu đoạn văn"})
    assert kept == 0
    cover, clear = items
    assert cover.page == 0
    assert cover.redact == [(72.0, 600.0, 500.0, 612.0), (72.0, 614.0, 500.0, 626.0)]
    assert cover.resume == (1, 120.0)
    assert (clear.page, clear.kind, clear.text) == (1, "clear", "")
    assert clear.redact == [(72.0, 120.0, 500.0, 132.0)]


def test_the_tail_resumes_at_the_source_continuation_not_the_top_margin(tmp_path, unicode_font):
    """Rebuild a paragraph split by the page break and read both pages back."""
    import pymupdf

    source = pymupdf.open()
    first = source.new_page(width=595.0, height=842.0)
    # The paragraph runs into the page's bottom edge, as the real book's does:
    # the extractor joins such a block with its continuation on the next page.
    y = 790.0
    for index in range(4):
        # No full stop: the next page's first line must read on from this one.
        first.insert_text(
            (72.0, y),
            f"Beta paragraph line {index} of the source page",
            fontname="helv",
            fontsize=10.0,
        )
        y += 12.0
    second = source.new_page(width=595.0, height=842.0)
    y = 120.0
    second.insert_text(
        (72.0, y), "continued on this page line 4 of the text", fontname="helv", fontsize=10.0
    )
    y += 12.0
    for index in range(5, 8):
        second.insert_text(
            (72.0, y),
            f"Beta paragraph line {index} of the source page",
            fontname="helv",
            fontsize=10.0,
        )
        y += 12.0
    for index in range(3):
        second.insert_text(
            (72.0, y + 40.0), f"Gamma paragraph line {index} here.", fontname="helv", fontsize=10.0
        )
        y += 12.0
    src = tmp_path / "in.pdf"
    source.save(str(src))
    source.close()

    doc = load_document(src)
    assert len(doc.blocks) == 2, "the paragraph over the break must be one block"
    texts = {0: "Đoạn dài " + "chữ " * 60, 1: "Gamma ngắn."}
    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == []
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        first_text = rebuilt[0].get_text()
        second_text = rebuilt[1].get_text()
        assert "of the source page" not in first_text
        assert "of the source page" not in second_text
        tail = [word for word in rebuilt[1].get_text("words") if word[4] == "chữ"]
        assert tail, "the translation continues on page 2"
        # The source's continuation began at y=120; the tail belongs there, not
        # above the page's top margin where the running head would sit (its
        # first line's bbox starts a little above the baseline, hence 100).
        assert min(word[1] for word in tail) >= 100.0
        assert max(word[3] for word in tail) <= 842.0 - 27.0 + 1
        gamma = [word for word in rebuilt[1].get_text("words") if word[4].startswith("Gamma")]
        assert gamma
        assert max(word[3] for word in tail) <= min(word[1] for word in gamma) + 1


def test_a_folded_block_printed_above_clears_a_page_of_its_own():
    """Every line of the folded block sits on the next page: the item above
    cannot take them into its box, but their words were still translated there,
    so they are cleared on their own page and nothing prints over them."""
    doc = Document(
        blocks=[
            Block(
                kind="paragraph",
                text="First",
                meta={
                    "pdf": {"rects": [(0, 72.0, 100.0, 300.0, 112.0)], "size": 10.0, "kind": "body"}
                },
            ),
            Block(
                kind="paragraph",
                text="Rest",
                meta={
                    "pdf": {"rects": [(1, 72.0, 50.0, 300.0, 90.0)], "size": 10.0, "kind": "body"}
                },
            ),
        ]
    )
    items, kept, merged = placements(doc, {0: "Đoạn đầu", 1: None})
    assert (kept, merged) == (0, 1)
    cover, clear = items
    assert cover.box == (72.0, 100.0, 300.0, 118.0)  # untouched by the far page
    assert (clear.page, clear.redact) == (1, [(72.0, 50.0, 300.0, 90.0)])


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


def test_a_pinned_font_is_not_second_guessed(unicode_font, tmp_path):
    """``--font-path``/``PDF_FONT_PATH`` means \"this face\", not \"a face like it\"."""
    pinned = tmp_path / "pinned.ttf"
    pinned.write_bytes(unicode_font.read_bytes())
    items, _, _ = placements(
        _styled_doc(), {2: "Dòng ngắn."}, default_font=pinned, match_style=False
    )
    assert items[0].font_file == str(pinned)


def test_a_pinned_face_is_set_aside_for_a_symbol_it_cannot_print(tmp_path, monkeypatch):
    """A chosen font is kept for everything it can print.

    Pinning picks the typography, not a page full of boxes: when the block
    carries a symbol the pinned face has no glyph for (``∈`` is in no Arial),
    the block is printed in a face that does have it.
    """
    pinned = tmp_path / "pinned.ttf"
    pinned.write_bytes(b"\x00\x01")
    wide = tmp_path / "wide.ttf"
    wide.write_bytes(b"\x00\x01")
    monkeypatch.setattr(pdf_output, "_WIDE_FONT_CANDIDATES", (str(wide),))
    plain = tuple(sorted({ord(ch) for ch in "f = O(n)" if not ch.isspace()}))
    monkeypatch.setattr(
        pdf_output,
        "_covers_codepoints",
        lambda font, codepoints: font == str(wide) or codepoints == plain,
    )
    assert pdf_output._printable_face(str(pinned), "f ∈ O(n)", pinned) == str(wide)
    assert pdf_output._printable_face(str(pinned), "f = O(n)", pinned) == str(pinned)


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


def test_a_long_block_fills_the_room_below_it_at_the_source_size(tmp_path, unicode_font):
    """Vượt khung của khối gốc nhưng trang còn trống: khối **dồn xuống dưới** ở
    đúng cỡ gốc — không tràn sang trang chèn, không thu nhỏ chữ.

    Đây là quy tắc dồn dòng trong trang của bản dựng lại: một bản dịch dài hơn
    dòng tiếng Anh không mất đuôi sang trang chèn chỉ vì khung của nó cắt ngắn
    hơn chữ; trang còn chỗ thì chữ in xuống chỗ đó, cỡ của nguồn giữ nguyên
    (in 10 pt tại 5.5 pt là thứ một trang dựng lại không thể chịu được).
    """
    import pymupdf
    from pdfdoc import write_styled_pdf

    src = write_styled_pdf(tmp_path / "in.pdf", "Alpha ", "Beta", " Gamma")
    doc = load_document(src)
    source_size = doc.blocks[0].meta["pdf"]["size"]
    long_text = "Alpha ⟦BOLD⟧" + "rất dài " * 12 + "⟦/BOLD⟧ Gamma"
    stats = rebuild_pdf(src, doc, {0: long_text}, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == [], "chữ dài vẫn phải in được"
    assert stats.continued == 0, "trang còn trống thì không được tràn sang trang chèn"
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert rebuilt.page_count == 1, "không có trang chèn nào"
        page = rebuilt[0]
        text = " ".join(page.get_text().split())
        assert text.count("rất dài") == 12 and text.endswith("Gamma"), "chữ không được mất"
        words = page.get_text("words")
        assert not any("⟦" in word[4] for word in words), "token không được lộ lên trang"
        height = page.rect.height
        assert all(word[1] >= 0 and word[3] <= height for word in words), (
            "chữ không được rời khỏi trang"
        )
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    assert abs(span["size"] - source_size) < 0.5, (
                        f"chữ phải giữ nguyên cỡ gốc: {span['size']} vs {source_size}"
                    )


def test_a_long_block_pushes_the_block_below_further_down_the_page(tmp_path, unicode_font):
    """Khối dài hơn khung gốc dồn khối dưới xuống — cùng trang, không đè lên
    nhau, không tràn sang trang chèn, không đổi cỡ chữ.

    Đây là hành vi "dồn dòng" cụ thể nhất: khối trên lấy thêm chỗ của khoảng
    trắng mà nguồn để giữa hai đoạn, và khối dưới (và các khối sau nữa) lùi
    xuống theo.
    """
    import pymupdf
    from pdfdoc import write_pdf

    _PARA_ONE = (
        "Alpha first paragraph wraps over several lines in the source, which "
        "is what marks it as its own block for the extractor to read."
    )
    _PARA_TWO = (
        "Beta second paragraph also wraps over several lines in the source, "
        "so the blank line between the two is clearly larger than the norm."
    )
    _PARA_THREE = (
        "Gamma third paragraph also wraps over several lines in the source, "
        "so its block is separated from the second one by the same gap again."
    )

    src = write_pdf(tmp_path / "in.pdf", f"{_PARA_ONE}\n\n{_PARA_TWO}\n\n{_PARA_THREE}\n")
    doc = load_document(src)
    assert len(doc.blocks) == 3, "cần đúng ba block trên một trang nguồn"
    source_size = doc.blocks[0].meta["pdf"]["size"]
    pushed_top = doc.blocks[1].meta["pdf"]["rects"][0][2]
    texts = {0: "Alpha mở rộng " + "lời " * 100, 1: "Beta rất ngắn."}
    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == []
    assert stats.continued == 0, "trang còn trống thì không được tràn sang trang chèn"
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert rebuilt.page_count == 1, "không có trang chèn nào"
        page = rebuilt[0]
        words = page.get_text("words")
        # Chỉ chữ của khối trên ("lời"), không lẫn chữ tiếng Anh còn giữ lại.
        alpha = [word for word in words if word[4] in {"Alpha", "mở", "rộng", "lời"}]
        beta = [word for word in words if word[4] == "Beta"]
        assert alpha and beta, "cả hai khối đều phải có mặt"
        assert beta[0][1] > pushed_top + 5, (
            "khối dưới phải bị dồn xuống dưới vị trí nguồn của nó, không đứng yên"
        )
        assert max(word[3] for word in alpha) <= beta[0][1] + 1, (
            "khối trên không được đè lên khối dưới"
        )
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    assert abs(span["size"] - source_size) < 0.5, "phải in đúng cỡ nguồn"


def test_a_block_whose_tail_would_spill_keeps_the_source_size(tmp_path, unicode_font):
    """Khối dài hơn khung (khối tiếng Anh bên dưới giữ chỗ, nên khung không nới
    được) **không** bị thu nhỏ chữ: phần đuôi sang trang chèn, và mọi chữ vẫn in
    đúng cỡ nguồn.

    10pt in thành 6pt không còn là cùng một tài liệu, và người đọc thấy chỗ chữ
    nhỏ trước khi thấy thêm một trang. Chỉ ô bảng mới được thu nhỏ (70%), vì ô
    không có chỗ nào khác để đi.
    """
    import pymupdf
    from pdfdoc import write_pdf

    _PARA_ONE = (
        "Alpha first paragraph wraps over several lines in the source, which "
        "is what marks it as its own block for the extractor to read."
    )
    _PARA_TWO = (
        "Beta second paragraph also wraps over several lines in the source, "
        "so the blank line between the two is clearly larger than the norm."
    )

    # Block 1 stays English (it is absent from `texts`): an anchor the block
    # above may neither push down nor print over, so its box cannot grow.
    src = write_pdf(tmp_path / "in.pdf", f"{_PARA_ONE}\n\n{_PARA_TWO}\n")
    doc = load_document(src)
    source_size = doc.blocks[0].meta["pdf"]["size"]
    texts = {0: "Alpha mo rong " + "loi " * 150}
    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == []
    assert stats.continued >= 1, "phần đuôi phải sang trang chèn thay vì bị thu nhỏ"
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert rebuilt.page_count == stats.pages, "trang chèn phải được tính đúng"
        texts_per_page = [page.get_text() for page in rebuilt]
        assert sum(text.count("loi") for text in texts_per_page) == 150, "không được mất chữ nào"
        assert "loi" in texts_per_page[0], "đầu khối vẫn ở trang nguồn"
        assert "loi" in "".join(texts_per_page[1:]), "đuôi phải có mặt trên trang chèn"
        for page in rebuilt:
            sizes = [
                span["size"]
                for block in page.get_text("dict")["blocks"]
                for line in block.get("lines", [])
                for span in line["spans"]
            ]
            assert all(abs(size - source_size) < 0.5 for size in sizes), (
                "chữ phải in đúng cỡ nguồn, không được thu nhỏ"
            )
            height = page.rect.height
            assert all(word[1] >= 0 and word[3] <= height for word in page.get_text("words"))


def test_a_block_needing_less_than_half_the_size_still_continues(tmp_path, unicode_font):
    """Khối dài gấp nhiều lần khung: đuôi sang trang chèn ở đúng cỡ nguồn.

    Không có ngưỡng thu nhỏ nào nữa — khối bình thường luôn in đúng cỡ gốc.
    """
    from pdfdoc import write_pdf

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
    texts = {0: "Alpha rat dai " + "cau " * 900}
    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == []
    assert stats.continued >= 1, "past the 50% floor the tail still moves on"


def test_a_page_with_no_room_left_continues_the_whole_block_overleaf(tmp_path, unicode_font):
    """Trang đầy tới đáy: khối không có lấy một dòng chỗ phải in **nguyên khối**
    sang trang chèn, ở đúng cỡ gốc — không in đè lên gì cả, và cũng không in
    một dòng lạc lõng rồi mất phần còn lại."""
    import pymupdf
    from pdfdoc import write_pdf

    _PARA_ONE = (
        "Alpha first paragraph wraps over several lines in the source, which "
        "is what marks it as its own block for the extractor to read."
    )
    _PARA_TWO = (
        "Beta second paragraph also wraps over several lines in the source, "
        "so the blank line between the two is clearly larger than the norm."
    )
    _PARA_THREE = (
        "Gamma third paragraph also wraps over several lines in the source, "
        "so its block is separated from the second one by the same gap again."
    )

    src = write_pdf(tmp_path / "in.pdf", f"{_PARA_ONE}\n\n{_PARA_TWO}\n\n{_PARA_THREE}\n")
    doc = load_document(src)
    # Cả ba khối đều được dịch, nên không khối nào là mốc giữ chỗ: khối đầu dài
    # enough để lấp đầy trang — kể cả khi đựngthu nhỏ đựngthớ 50% — và hai khối
    # sau không còn một dòng chỗ.
    texts = {
        0: "Alpha rất dài " + "câu " * 7000,
        1: "Beta kết thúc.",
        2: "Gamma cũng ngắn.",
    }
    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == []
    assert stats.continued >= 1
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert rebuilt.page_count == stats.pages, "trang chèn phải được tính đúng"
        texts_per_page = [page.get_text() for page in rebuilt]
        source_text, cont_text = texts_per_page[0], "".join(texts_per_page[1:])
        assert "Alpha" in source_text, "đầu khối đầu vẫn ở trang nguồn"
        assert "Beta" not in source_text, "khối không có chỗ thì không được in một phần"
        assert "Beta kết thúc." in cont_text and "Gamma cũng ngắn." in cont_text, (
            "khối không có chỗ phải in nguyên khối sang trang chèn"
        )
        # Không mất chữ: đủ 7000 "câu" qua các trang.
        assert sum(text.count("câu") for text in texts_per_page) == 7000


def test_spilling_blocks_share_one_overleaf_page(tmp_path, unicode_font):
    """Hai block cùng tràn vào **một** trang chèn dùng chung, không phải mỗi
    block một trang gần như trống — 44 trang như vậy từng làm cuốn 69 trang
    thành 113.

    Đồng thời đó là test hồi quy cho refetch: sau khi block đầu tràn, block
    cuối vẫn phải in trên **trang nguồn** (index của trang nguồn không dịch
    chuyển khi trang chèn được thêm vào), ngược với bản cũ refetch lấy ngay
    trang chèn cuối và để một lỗ trống trên trang nguồn.

    Block giữa được **giữ nguyên tiếng Anh** (không có trong `texts`): nó vừa
    là mốc neo — khối đầu dồn xuống phải dừng trên nó, phần đuôi sang trang
    chèn thay vì in đè lên chữ còn giữ lại — vừa chứng minh chữ nguồn không bị
    mất.
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
    _PARA_THREE = (
        "Gamma third paragraph also wraps over several lines in the source, "
        "so its block is separated from the second one by the same gap again."
    )

    src = write_pdf(tmp_path / "in.pdf", f"{_PARA_ONE}\n\n{_PARA_TWO}\n\n{_PARA_THREE}\n")
    doc = load_document(src)
    assert len(doc.blocks) == 3, "cần đúng ba block trên một trang nguồn"
    # Block giữa (index 1) không có trong texts: giữ nguyên tiếng Anh.
    # Khối đầu phải đựngthản đựngthả đựngthủ đựngthử chụngthì đựngthu đựngthỏmprức
    # đựngthớ 50% cỡ nguồn — ngọathôn ngọathôn chụngthì đựngthu đựngthỏmprức —
    # đựngthì đựngthả đựngthủ đựngthử chụngthì đựngthả đựngthủ chụngthử...
    texts = {0: "Alpha mở rộng " + "lời " * 500, 2: "Gamma rất dài " + "câu " * 900}
    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == []
    assert stats.kept == 1, "block giữa phải được giữ nguyên tiếng Anh"
    assert stats.continued >= 1, "hai block tràn phải đựngthả đựngthủ chụngthử"
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert rebuilt.page_count == stats.pages, (
            "trang chèn phải đựngthống đựngthống đựngthống đựngthống"
        )
        texts_per_page = [page.get_text() for page in rebuilt]
        source_text, cont_text = texts_per_page[0], "".join(texts_per_page[1:])
        assert "Alpha" in source_text, "đầu block đầu vẫn ở trang nguồn"
        assert "Beta second paragraph" in source_text, (
            "chữ tiếng Anh còn giữ lại không được bị dồn dòng đè lên"
        )
        assert "Gamma" in source_text, (
            "block cuối phải in trên trang nguồn sau khi block đầu tràn (refetch)"
        )
        # Hai tail xếp chồng theo thứ tự block trên cột chèn dùng chung.
        assert "lời" in cont_text and "câu" in cont_text, "cả hai tail phải có mặt trên trang chèn"
        assert cont_text.index("lời") < cont_text.index("câu"), "tail phải giữ đúng thứ tự block"
        # Không mất chữ: đủ 900 "câu" của block cuối.
        assert sum(text.count("câu") for text in texts_per_page) == 900


def test_a_block_above_a_table_stops_at_the_grid(tmp_path, unicode_font):
    """Dồn dòng không được dồn lên lưới bảng: các ô đứng yên chỗ bản gốc vẽ,
    chữ của khối trên dừng trên bảng và phần đuôi tiếp tục sang trang chèn."""
    pymupdf = pytest.importorskip("pymupdf")
    from pdfdoc import write_table_pdf

    src = write_table_pdf(
        tmp_path / "table.pdf",
        [["Feature", "State"], ["Servers", "5"]],
        col_widths=[240.0, 70.0],
        prose=(
            "Alpha first paragraph wraps over several lines in the source, which "
            "is what marks it as its own block for the extractor to read here.\n\n"
            "Beta second paragraph also wraps over several lines in the source, "
            "so the blank line between the two is clearly larger than the norm."
        ),
        top=340.0,
    )
    doc = load_document(src)
    table = next(block for block in doc.blocks if block.meta.get("table"))
    cells = [cell[:4] for row in table.meta["pdf"]["cells"] for cell in row]
    texts = {
        index: (
            "| Tính năng | Trạng thái |\n| --- | --- |\n| Máy chủ | 5 |"
            if block is table
            # Khối ngay trên bảng: dài hơn khoảng trống giữa nó và lưới.
            else "Beta mở rộng " + "lời " * 600
            if index == len(doc.blocks) - 2
            else "Alpha một đoạn ngắn."
        )
        for index, block in enumerate(doc.blocks)
    }
    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == []
    assert stats.cells == 4 and stats.cells_kept == 0, "lưới phải được dựng lại đủ ô"
    assert stats.continued >= 1, "khối không vừa lưới phải tiếp tục sang trang chèn"
    with pymupdf.open(src) as original, pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert len(rebuilt[0].get_drawings()) == len(original[0].get_drawings()), "lưới còn nguyên"
        text = rebuilt[0].get_text()
        assert "Máy chủ" in text and "Tính năng" in text, "dịch ô phải in vào ô"
        assert "lời" in text, "chữ của khối trên phải có mặt trên trang nguồn"
        # Không chữ nào của khối trên lọt vào ô bảng.
        for word in rebuilt[0].get_text("words"):
            for bx0, btop, bx1, bbottom in cells:
                if (
                    word[0] >= bx0 - 1
                    and word[2] <= bx1 + 1
                    and word[1] >= btop - 1
                    and word[3] <= bbottom + 1
                ):
                    assert word[4] != "lời", f"chữ khối trên lọt vào ô: {word[4]!r}"
                    break


def test_a_block_the_source_set_low_keeps_the_room_the_page_used(tmp_path, unicode_font):
    """Khối mà bản gốc đặt tận đáy trang vẫn được in xuống đúng chỗ đó.

    Nguồn in chữ xuống tới y≈841 trên trang này, nhưng margin dựng lại là 28 pt
    tính từ đáy trang (y=814): một khối dồn dòng bị cắt ngắn hơn bản gốc thì đuôi
    của nó mất chỗ và phải tràn sang trang chèn — đọc thành một cuốn sách bị ngắt
    trang vô cớ. Giới hạn dưới vì thế không ngắn hơn chính phòng mà bản gốc đã
    dùng cho khối đó.
    """
    pymupdf = pytest.importorskip("pymupdf")

    source = pymupdf.open()
    page = source.new_page(width=595.0, height=842.0)
    y = 800.0
    for index in range(3):
        page.insert_text(
            (72.0, y),
            f"Line {index} of a paragraph the source sets near its bottom edge.",
            fontname="helv",
            fontsize=10.0,
        )
        y += 12.0
    src = tmp_path / "low.pdf"
    source.save(str(src))
    source.close()
    doc = load_document(src)
    assert len(doc.blocks) == 1, "cần đúng một khối ở đáy trang"
    # Ba dòng chữ: vượt qua mốc 814 (lề dựng lại), nhưng vẫn nằm trong chính
    # khung mà bản gốc đặt cho khối (tới y≈832).
    long_text = "Dòng dịch dài hơn dòng nguồn và chạy sát đáy trang " * 3
    stats = rebuild_pdf(src, doc, {0: long_text}, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == []
    assert stats.continued == 0, (
        "chữ phải in hết xuống phòng bản gốc đã dùng, không được tràn sang trang chèn"
    )
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert rebuilt.page_count == 1, "không có trang chèn nào"
        page = rebuilt[0]
        words = page.get_text("words")
        assert words, "chữ phải được in"
        # Vượt qua mốc 814 (lề dựng lại) — bản gốc cũng in tới đó.
        assert max(word[3] for word in words) > 814.0, (
            "khối đặt gần đáy trang phải được in xuống phòng mà bản gốc đã dùng"
        )
        assert max(word[3] for word in words) <= page.rect.height, "chữ không được rời khỏi trang"
        assert " ".join(page.get_text().split()).count("sát") == 3, "không mất chữ"


def test_landing_stands_below_a_box_in_its_first_line():
    """Nơi một khối bị dồn xuống bắt đầu và kéo dài tới đâu.

    Một khối bị dồn xuống không được in đè lên những gì trang còn giữ: nếu một
    khối neo đứng ngay trong dòng đầu của nó, khối phải lùi xuống dưới khối neo
    ấy; nếu khối neo nằm dưới, khối dừng lại trước nó; và khối neo ở cột khác
    thì không liên quan.
    """
    page_bottom = 800.0
    anchors = [(72.0, 110.0, 300.0, 150.0)]
    # No room for even the first line before the anchor: stand below it.
    assert pdf_output._landing(100.0, 72.0, 300.0, 10.0, anchors, page_bottom) == (152.0, 800.0)
    # The same anchor, with the start well above it: reach down to it, not past.
    lower = [(72.0, 200.0, 300.0, 240.0)]
    assert pdf_output._landing(100.0, 72.0, 300.0, 10.0, lower, page_bottom) == (100.0, 198.0)
    # Another column is not in the way.
    beside = [(310.0, 110.0, 500.0, 150.0)]
    assert pdf_output._landing(100.0, 72.0, 300.0, 10.0, beside, page_bottom) == (100.0, 800.0)


def test_a_rebuilt_page_keeps_the_source_geometry_and_type(tmp_path):
    """Định dạng của trang nguồn phải còn nguyên sau khi dựng lại.

    Một trang có đủ kiểu chữ mà người đọc nhận ra ngay nếu sai: tiêu đề đậm cỡ
    lớn, đoạn thân thụt lề, một đoạn chữ màu. Bản dịch ở đây ngắn, nên không
    khối nào phải dồn xuống hay tràn trang: vị trí, cỡ chữ, độ đậm và màu của
    từng khối phải trùng bản gốc, và trang vẫn đúng một trang cùng khổ.
    """
    import pymupdf

    source = pymupdf.open()
    page = source.new_page(width=595.0, height=842.0)
    page.insert_text((72.0, 120.0), "Chapter Two", fontname="hebo", fontsize=17.0)
    page.insert_text(
        (90.0, 170.0), "Body text of the first paragraph.", fontname="helv", fontsize=11.0
    )
    page.insert_text((90.0, 182.0), "It carries a second line.", fontname="helv", fontsize=11.0)
    page.insert_text(
        (72.0, 240.0),
        "A paragraph printed in colour.",
        fontname="helv",
        fontsize=11.0,
        color=(0.8, 0.0, 0.0),
    )
    src = tmp_path / "in.pdf"
    source.save(str(src))
    source.close()

    doc = load_document(src)
    assert len(doc.blocks) == 3, "cần ba khối trên trang nguồn"
    texts = {
        0: "Chương Hai",
        1: "Đoạn thân thứ nhất.",
        2: "Một đoạn in bằng màu.",
    }
    # Không ghim font: đây là đường chạy thật, nơi khối tự tìm lại họ chữ và độ
    # đậm của bản gốc.
    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf")
    assert stats.failures == []
    assert stats.continued == 0, "bản dịch ngắn thì không được tràn trang"

    def spans_of(page, needle):
        found = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if needle in span["text"]
        ]
        assert found, f"không thấy chữ {needle!r} trên trang"
        return found[0]

    with pymupdf.open(src) as original, pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert rebuilt.page_count == original.page_count == 1
        assert rebuilt[0].rect == original[0].rect, "khổ trang phải giữ nguyên"
        for needle, source_span in (
            ("Chương", spans_of(original[0], "Chapter Two")),
            ("thân", spans_of(original[0], "Body text")),
            ("màu", spans_of(original[0], "printed in colour")),
        ):
            span = spans_of(rebuilt[0], needle)
            assert abs(span["origin"][0] - source_span["origin"][0]) < 1.0, (
                f"lề trái của {needle!r} phải giữ nguyên"
            )
            # Dòng đầu đứng đúng chỗ: sai số còn lại là độ cao ascender của mặt
            # chữ dùng để in so với mặt chữ gốc (dưới 2pt ở mọi cỡ), không phải
            # một khối bị dồn đi chỗ khác.
            assert abs(span["origin"][1] - source_span["origin"][1]) < 2.0, (
                f"dòng đầu của {needle!r} phải đứng đúng chỗ"
            )
            assert abs(span["size"] - source_span["size"]) < 0.2, (
                f"cỡ chữ của {needle!r} phải giữ nguyên"
            )
            assert span["color"] == source_span["color"], f"màu chữ của {needle!r} phải giữ nguyên"
        assert "Bold" in spans_of(rebuilt[0], "Chương")["font"], "tiêu đề phải in đậm"
        assert "Bold" not in spans_of(rebuilt[0], "thân")["font"], "thân bài không đậm"


def test_a_page_break_never_strands_a_tail_on_a_page_of_its_own(tmp_path, unicode_font):
    """Trang trước tràn khỏi đáy: phần đuôi in **đầu trang sau**, chữ của trang
    sau lùi xuống dưới nó — không sinh thêm một trang chỉ có một dòng.

    Đây là lỗi người đọc thấy trên cuốn tài liệu thật: một trang gần như trắng,
    chỉ còn một gạch đầu dòng lạc lõng, rồi trang sau mới tiếp tục. Trang chèn
    riêng cho phần đuôi là thứ phải biến mất: văn bản chảy sang trang sau như
    một dòng chữ bình thường, đúng cỡ nguồn.
    """
    import pymupdf

    source = pymupdf.open()
    first = source.new_page(width=595.0, height=842.0)
    # Ba dòng mỗi khối: extractor tách block theo khoảng cách dòng, và một đoạn
    # đúng một dòng sẽ tự biến khoảng trắng của nó thành chuẩn (xem pdfdoc).
    y = 150.0
    for index in range(3):
        first.insert_text(
            (72.0, y), f"Alpha paragraph line {index} of the page.", fontname="helv", fontsize=10.0
        )
        y += 12.0
    # Khối cuối nằm sát đáy trang: bản dịch dài hơn không còn chỗ nào dưới nó.
    y = 790.0
    for index in range(3):
        first.insert_text(
            (72.0, y), f"Beta last paragraph line {index} here.", fontname="helv", fontsize=10.0
        )
        y += 12.0

    second = source.new_page(width=595.0, height=842.0)
    y = 120.0
    for index in range(3):
        second.insert_text(
            (72.0, y), f"Gamma next page line {index} of text.", fontname="helv", fontsize=10.0
        )
        y += 12.0
    src = tmp_path / "in.pdf"
    source.save(str(src))
    source.close()

    doc = load_document(src)
    assert len(doc.blocks) == 3, "cần hai khối ở trang đầu và một khối ở trang sau"
    texts = {
        0: "Alpha ngắn.",
        1: "Beta dài " + "chữ " * 40,
        2: "Gamma ngắn.",
    }
    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == []
    assert stats.continued == 0, "phần đuôi phải chảy sang trang sau, không cần trang chèn"
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert rebuilt.page_count == 2, "không được thêm trang chỉ cho phần đuôi"
        first_page, second_page = rebuilt[0], rebuilt[1]
        assert "Beta" in first_page.get_text(), "đầu khối vẫn ở trang nguồn"
        tail = [word for word in second_page.get_text("words") if word[4] == "chữ"]
        gamma = [word for word in second_page.get_text("words") if word[4] == "Gamma"]
        assert tail and gamma, "trang sau phải có cả phần đuôi lẫn chữ của chính nó"
        assert min(word[1] for word in tail) < min(word[1] for word in gamma), (
            "phần đuôi phải in trên chữ của trang sau"
        )
        assert max(word[3] for word in tail) <= min(word[1] for word in gamma) + 1, (
            "chữ của trang sau phải lùi xuống dưới phần đuôi, không đè lên"
        )
        assert " ".join(rebuilt[0].get_text().split()).count("chữ") + len(tail) == 40, (
            "không được mất chữ nào của phần đuôi"
        )
        for page in rebuilt:
            height = page.rect.height
            assert all(word[1] >= 0 and word[3] <= height for word in page.get_text("words"))
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        assert abs(span["size"] - 10.0) < 0.5, "phải in đúng cỡ nguồn"
