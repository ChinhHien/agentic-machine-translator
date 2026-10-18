"""Tests for the layout-aware PDF extractor.

The extractor's input is what pdfplumber reports per glyph (text, x-offset,
size, font), so the tests build pdfplumber-shaped lines directly: that exercises
classification, word-gap reconstruction, code indentation and paragraph
grouping without needing a hand-crafted PDF.
"""

from __future__ import annotations

from polyglot_lc.ingest import load_document
from polyglot_lc.ingest.md import parse_markdown
from polyglot_lc.ingest.model import Block, Document
from polyglot_lc.ingest.pdf_layout import (
    _build_units,
    _line_spacing,
    _page_lines,
    _table_grid,
    bind_page_geometry,
    line_text,
    looks_like_code,
    normalize_glyphs,
    normalize_toc_entry,
    normalize_toc_lines,
    pages_to_markdown,
    render_markdown,
)

BODY = "ABCDEF+CMR12"
MONO = "GHIJKL+CMTT12"
BOLD = "MNOPQR+CMBX12"
BULLET = "unknown"


def _chars(
    tokens: list[str], x0: float, size: float, font: str, char_w: float = 6.0, word_gap: float = 3.0
) -> list[dict]:
    """Glyphs for *tokens*, with `word_gap` between words and none inside one."""
    chars: list[dict] = []
    x = x0
    for token in tokens:
        for ch in token:
            chars.append({"text": ch, "x0": x, "x1": x + char_w, "size": size, "fontname": font})
            x += char_w
        x += word_gap
    return chars


def _line(
    tokens: list[str], top: float, x0: float = 100.0, size: float = 12.0, font: str = BODY
) -> dict:
    return {"chars": _chars(tokens, x0, size, font), "top": top}


def _words(text: str) -> list[str]:
    """One line's tokens, written out as the words a reader would see.

    The extractor never sees spaces — it rebuilds them from the gaps between
    glyph runs — so a fixture's line is a list of words either way; this just
    spares every fixture the same twelve-element list.
    """
    return text.split()


class _StubPage:
    """Minimal pdfplumber page: text lines, a height, and table boxes."""

    def __init__(
        self, lines: list[dict], *, height: float = 0.0, tables: list | None = None
    ) -> None:
        self._lines = sorted(lines, key=lambda ln: ln["top"])
        self.height = height
        self._tables = tables or []

    def extract_text_lines(self, layout: bool = False) -> list[dict]:
        # `layout` mirrors pdfplumber's signature; the stub ignores it.
        return self._lines

    def find_tables(self) -> list:
        return self._tables


class _Row:
    """pdfplumber's TableRow: only `cells`, which is all the extractor reads."""

    def __init__(self, cells: list) -> None:
        self.cells = cells


def _even_grid(
    bbox: tuple[float, float, float, float], nrows: int, ncols: int
) -> list[list[tuple]]:
    """Split a table box into an even grid, for tests that need no exact boxes."""
    x0, top, x1, bottom = bbox
    dx, dy = (x1 - x0) / ncols, (bottom - top) / nrows
    return [
        [(x0 + c * dx, top + r * dy, x0 + (c + 1) * dx, top + (r + 1) * dy) for c in range(ncols)]
        for r in range(nrows)
    ]


class _StubTable:
    """Minimal pdfplumber table: a bounding box, a cell matrix and cell boxes."""

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        rows: list[list[str]],
        cells: list[list[tuple[float, float, float, float] | None]] | None = None,
    ) -> None:
        self.bbox = bbox
        self._rows = rows
        self._cells = cells if cells is not None else _even_grid(bbox, len(rows), len(rows[0]))

    def extract(self) -> list[list[str]]:
        return self._rows

    @property
    def rows(self) -> list[_Row]:
        return [_Row(row) for row in self._cells]


def _page(lines: list[dict], *, height: float = 0.0, tables: list | None = None):
    return _StubPage(lines, height=height, tables=tables)


# --------------------------- word-gap reconstruction ------------------------
def test_line_text_rebuilds_missing_spaces():
    """A TeX PDF emits no space glyphs; the gap between words is the evidence."""
    line = _line(["bestfor", "smallproblems", "and"], top=100.0)
    assert line_text(line["chars"]) == "bestfor smallproblems and"


def test_line_text_keeps_words_without_internal_spaces():
    line = _line(["ArrayList", "methods"], top=100.0)
    assert line_text(line["chars"]) == "ArrayList methods"


# ----------------------------- classifications ------------------------------
def test_monospace_run_becomes_code_with_indentation():
    lines = [
        _line(["public", "E", "get(int", "index)", "{"], top=200.0, font=MONO),
        _line(["if", "(index", "<", "0)"], top=215.2, x0=124.6, font=MONO),
        _line(["}"], top=230.4, font=MONO),
    ]
    md = pages_to_markdown([_page(lines)])
    assert "```java" in md
    assert "public E get(int index) {" in md
    assert "\n    if (index < 0)\n" in md  # indent restored from x-offset


def test_prose_line_dominated_by_a_url_is_not_code():
    """TeX sets URLs in the typewriter face; a sentence is still a sentence."""
    assert not looks_like_code("sort (see http://thinkdast.com/selectsort):")
    assert looks_like_code("return array[index];")
    assert looks_like_code("// other methods omitted")
    lines = [
        _line(["read", "about", "it", "at", "http://thinkdast.com/types."], top=200.0, font=MONO),
    ]
    md = pages_to_markdown([_page(lines)])
    assert "```" not in md
    assert "read about it at http://thinkdast.com/types." in md


def test_undersized_mono_comment_line_is_still_code():
    lines = [_line(["//", "other", "methods", "omitted"], top=200.0, font=MONO)]
    md = pages_to_markdown([_page(lines)])
    assert md.strip() == "```\n// other methods omitted\n```"


def test_oversized_run_becomes_heading_with_level_from_size():
    lines = [
        _line(["3.1", "Classifying", "MyArrayList", "methods"], top=200.0, size=17.2, font=BOLD),
    ]
    md = pages_to_markdown([_page(lines)])
    assert md.strip() == "## 3.1 Classifying MyArrayList methods"


def test_chapter_title_lines_merge_into_one_heading():
    """ "Chapter 3" and "ArrayList" are two oversized lines of one title."""
    lines = [
        _line(["Chapter", "3"], top=210.0, size=24.8, font=BOLD),
        _line(["ArrayList"], top=273.0, size=24.8, font=BOLD),
    ]
    md = pages_to_markdown([_page(lines)])
    assert md.strip() == "# Chapter 3 ArrayList"


def test_cover_fields_that_each_open_with_a_bold_label_are_separate_blocks():
    """A title page sets its fields at prose leading, one field per line.

    The lines carry no sentence-ending punctuation and their gaps are the
    page's normal leading, so spacing reads them as one wrapped paragraph and
    the whole field list arrives glued into a single block: translated as one
    run-on paragraph, and printed as one shapeless blob on the rebuilt cover.
    A bold, colon-terminated label opening *every* line is what tells them
    apart: a wrapped continuation never opens with one.
    """
    fields = [
        ("Project Title:", "Design of a Web System"),
        ("Project Category:", "Software Engineering"),
        ("Programming Languages:", "Python, JavaScript"),
        ("Backend Framework:", "Django"),
    ]
    lines = [
        _styled_line(
            [(_words(label), BOLD_FACE), (_words(value), BODY)], top=100.0 + 14.5 * i
        )
        for i, (label, value) in enumerate(fields)
    ]
    md = pages_to_markdown([_page(lines)])
    assert md.count("\n\n") == len(fields) - 1
    for label, value in fields:
        assert f"⟦BOLD⟧{label}⟦/BOLD⟧ {value}" in md


def test_a_wrapped_paragraph_of_bold_lead_ins_still_stays_one_block():
    """One labelled line before regular prose is a definition, not a field run.

    Only two *consecutive* labelled lines break the block: a lone bold label
    may be a definition line whose text reads on into the paragraph below it,
    and a prose wrap never opens with a bold label of its own.
    """
    lines = [
        _styled_line(
            [(["Deployment", "Steps:"], BOLD_FACE), (_words("The application is deployed"), BODY)],
            top=100.0,
        ),
        _line(_words("behind a reverse proxy in production."), top=114.5),
    ]
    md = pages_to_markdown([_page(lines)])
    assert "\\n" not in md.strip()
    assert "⟦BOLD⟧Deployment Steps:⟦/BOLD⟧ The application is deployed" in md


def test_running_head_is_dropped():
    lines = [
        _line(["10", "Chapter", "2", "Analysis"], top=91.0, font=BOLD),
        _line(["Body", "text", "here."], top=127.0),
    ]
    md = pages_to_markdown([_page(lines)])
    assert "10 Chapter 2 Analysis" not in md
    assert "Body text here." in md


# ------------------------- document-derived parameters ---------------------
def test_margin_band_is_a_fraction_of_this_page_height():
    """A landscape sheet must not get a portrait book's 110-point band.

    On this sheet the running head and the first content line both sit inside
    110 pt; only the page number is a running head, so the section title and the
    prose under it have to survive.
    """
    page = _page(
        [
            _line(["4"], top=20.2, size=10.0),
            _line(["About", "Eocortex"], top=46.5, size=14.0, font=BOLD),
            _line(["Eocortex", "is", "a", "global", "provider."], top=72.6, size=10.0),
        ],
        height=595.0,
    )
    md = pages_to_markdown([page])
    assert "4" not in md.split("\n")
    assert "About Eocortex" in md
    assert "Eocortex is a global provider." in md


def test_page_number_at_the_top_of_a_landscape_page_is_dropped():
    page = _page(
        [
            _line(["42"], top=30.0, size=10.0),
            _line(["Real", "content", "here."], top=200.0),
        ],
        height=595.0,
    )
    md = pages_to_markdown([page])
    assert "42" not in md
    assert "Real content here." in md


def test_content_line_carrying_a_year_is_not_a_running_head():
    """The page number is the evidence; a number inside a line is not."""
    page = _page(
        [
            _line(
                ["Windows", "Server", "2016", "/", "2019", "/", "2022", "is", "supported."],
                top=40.0,
                size=10.0,
            ),
        ],
        height=595.0,
    )
    md = pages_to_markdown([page])
    assert "Windows Server 2016 / 2019 / 2022 is supported." in md


# ------------------------------ running feet -------------------------------
def test_a_running_foot_is_not_glued_to_the_next_page_s_first_line():
    """Chân trang (số trang) bị dán với dòng đầu trang sau thành một khối.

    Số trang nằm dưới cùng trang; sang trang sau, dòng đầu trang lại ở toạ độ
    *nhỏ* hơn dòng cuối trang trước, nên phép đo khoảng cách đọc hai dòng đó là
    cùng một đoạn. Khối sinh ra chỉ sở hữu hình chữ nhật của số trang — một ô
    rộng vài point — và bản dịch của cả khối bị in vào đó.
    """
    pages = [
        _page(
            [
                _line(["Body", "text", "that", "ends"], top=520.0),
                _line(["7"], top=560.0, x0=375.0),
            ],
            height=595.0,
        ),
        _page([_line(["Next", "page", "text."], top=60.0)], height=595.0),
    ]
    md = pages_to_markdown(pages)
    assert "Next page text." in md
    assert "7" not in md, "số trang là trang trí của trang, không phải nội dung"
    doc = _bound_doc(pages)
    assert _pages_of(doc) == [[0], [1]], "không khối nào được nằm trên hai trang"


def test_a_paragraph_the_page_break_cut_in_half_reads_on():
    """Ngoại lệ duy nhất của luật trên: câu bị ngắt giữa dòng vẫn nối lại.

    Không có dấu kết câu ở cuối trang trước và dòng đầu trang sau bắt đầu bằng
    chữ thường — đó là một đoạn văn, không phải hai khối.
    """
    pages = [
        _page([_line(["The", "sentence", "continues"], top=520.0)], height=595.0),
        _page([_line(["on", "the", "next", "page."], top=60.0)], height=595.0),
    ]
    md = pages_to_markdown(pages)
    assert "The sentence continues on the next page." in md


def test_two_headings_on_different_pages_are_not_one_title():
    """Với số trang đã bỏ, không còn gì xen giữa hai dòng tiêu đề lớn.

    Luật gộp "Chapter 3" / "ArrayList" chỉ đúng khi hai dòng là *một* tiêu đề bị
    ngắt dòng — cùng một trang. Tiêu đề phần ở đầu trang này và tiêu đề chương ở
    đầu trang sau là hai tiêu đề, và trước đây chỉ được tách ra nhờ dòng số trang
    nằm giữa chúng.
    """
    pages = [
        _page([_line(["Part", "I.", "Introduction"], top=100.0, size=24.0)], height=595.0),
        _page([_line(["1.", "Introduction"], top=100.0, size=24.0)], height=595.0),
    ]
    md = pages_to_markdown(pages)
    assert md.count("# ") == 2, md
    assert "# Part I. Introduction" in md
    assert "# 1. Introduction" in md


def test_a_page_without_a_height_keeps_every_line():
    """Trang không cho biết chiều cao thì không dòng nào bị coi là chân trang.

    ``_footer_band`` trả vô cực trong trường hợp đó: thà giữ lại một số trang còn
    hơn bỏ nhầm một dòng nội dung ở đáy một trang mình không đo được.
    """
    page = _page([_line(["Body", "text"], top=520.0), _line(["7"], top=560.0, x0=375.0)])
    md = pages_to_markdown([page])
    assert "7" in md
    assert "Body text" in md


def test_heading_threshold_follows_the_document_font_size():
    """A 14 pt heading over a 10 pt body is a heading; no constant says so."""
    lines = [
        _line([f"Body line {i} of prose."], top=120.0 + 12.0 * i, size=10.0) for i in range(24)
    ]
    lines.append(_line(["Integration", "facilities"], top=420.0, size=14.0, font=BOLD))
    md = pages_to_markdown([_page(lines)])
    assert "## Integration facilities" in md
    assert "## Body line 0 of prose." not in md


# --------------------------------- tables -----------------------------------
def test_table_is_emitted_as_markdown_and_not_repeated_as_prose():
    """Table rows must arrive as rows: flattening them loses the columns."""
    table = _StubTable(
        bbox=(40.0, 120.0, 500.0, 200.0),
        rows=[
            # "LS" and "ST" are each reported twice by a merged header cell.
            ["Feature", "Description", "LS", "LS", "ST", "ST"],
            [
                "Servers per system",
                "Maximum number of servers per system",
                "5",
                "5",
                "Not limited",
                "Not limited",
            ],
        ],
    )
    page = _page(
        [
            _line(["Intro", "paragraph."], top=70.0),
            _line(["Servers", "per", "system", "Maximum", "number", "of", "servers"], top=130.0),
            _line(["After", "the", "table."], top=250.0),
        ],
        height=595.0,
        tables=[table],
    )
    md = pages_to_markdown([page])
    assert "| Feature | Description | LS | ST |" in md
    assert "| Servers per system | Maximum number of servers per system | 5 | Not limited |" in md
    # The lines inside the table's box were replaced by the table, not doubled.
    assert "Servers per system Maximum number of servers" not in md
    assert "Intro paragraph." in md
    assert "After the table." in md


def test_a_single_column_region_is_not_a_table():
    """A framed paragraph is not a table; it must flow as text."""
    table = _StubTable(
        bbox=(40.0, 120.0, 500.0, 200.0), rows=[["Abandoned Objects"], ["Detection"]]
    )
    page = _page(
        [
            _line(["Abandoned", "Objects"], top=130.0),
            _line(["Detection"], top=150.0),
        ],
        height=595.0,
        tables=[table],
    )
    md = pages_to_markdown([page])
    assert "|" not in md
    assert "Abandoned Objects" in md
    assert "Detection" in md


def _bound_doc(pages: list) -> Document:
    """The document the extractor produces, with each block's geometry bound."""
    units = _build_units(pages)
    doc = parse_markdown(render_markdown(units), source_format="pdf")
    bind_page_geometry(doc, units)
    return doc


def _layout_fixture() -> tuple[Document, list]:
    """Three pages: a heading and a paragraph each, and a numbered list.

    Each block owns its own line(s) — page 2's list owns three units, one per
    item — so a block's geometry can be told apart from its neighbour's. Every
    page opens with a heading because a paragraph *continues* onto the next
    page until a line of another kind ends it (that is what a real page break
    looks like to the extractor: the next line is higher up the page).
    """
    pages = [
        _page(
            [
                _line(["Chapter", "One"], top=100.0, size=24.0),
                _line(["Alpha", "paragraph", "text."], top=150.0),
            ],
            height=595.0,
        ),
        _page(
            [
                _line(["Chapter", "Two"], top=100.0, size=24.0),
                _line(["Beta", "paragraph", "text."], top=150.0),
                _line(["1.", "First", "item"], top=200.0),
                _line(["2.", "Second", "item"], top=215.0),
                _line(["3.", "Third", "item"], top=230.0),
            ],
            height=595.0,
        ),
        _page(
            [
                _line(["Chapter", "Three"], top=100.0, size=24.0),
                _line(["Gamma", "paragraph", "text."], top=150.0),
            ],
            height=595.0,
        ),
    ]
    units = _build_units(pages)
    return parse_markdown(render_markdown(units), source_format="pdf"), units


def _pages_of(doc: Document) -> list[list[int]]:
    """The source page of every line of every bound block, in order."""
    return [
        [int(rect[0]) for rect in block.meta["pdf"]["rects"]]
        for block in doc.blocks
        if "pdf" in block.meta
    ]


def test_every_block_is_bound_to_the_units_it_was_parsed_from():
    """Not to the unit in the same position: to the unit holding its text.

    Ran on the book this was written for, the positional walk drifted: of the
    572 blocks whose source page could be pinned down, 250 carried geometry
    from 1 to 7 pages away — a paragraph bound to one line of a page six pages
    on is a page of 5.5 pt type printed over another page's words.
    """
    doc, units = _layout_fixture()
    assert bind_page_geometry(doc, units) == len(doc.blocks) == 7
    assert [block.kind for block in doc.blocks] == [
        "heading",
        "paragraph",
        "heading",
        "paragraph",
        "list",
        "heading",
        "paragraph",
    ]
    assert _pages_of(doc) == [[0], [0], [1], [1], [1, 1, 1], [2], [2]]
    assert [round(rect[2]) for rect in doc.blocks[4].meta["pdf"]["rects"]] == [200, 215, 230]
    assert doc.blocks[4].meta["pdf"]["markers"] == ["1.", "2.", "3."]


def test_a_block_with_no_unit_of_its_own_takes_no_other_block_s_lines():
    """The one thing the geometry must never do: move onto someone else's text.

    A block the Markdown parser produced without units of its own used to be
    matched by kind to whatever unit the walk had reached, so it was printed
    over the *next* block's source lines and every block after it moved on by
    one. An unmatched block is now left unbound — it keeps its source text —
    and the blocks after it still own their own lines.
    """
    doc, units = _layout_fixture()
    rogue = Block(kind="paragraph", text="A paragraph no unit produced.")
    doc.blocks.insert(2, rogue)
    assert bind_page_geometry(doc, units) == 7
    assert "pdf" not in rogue.meta
    assert _pages_of(doc) == [[0], [0], [1], [1], [1, 1, 1], [2], [2]]


def test_table_cell_boxes_line_up_with_the_markdown_matrix():
    table = _StubTable(
        bbox=(40.0, 120.0, 500.0, 200.0),
        rows=[["Feature", "State"], ["Servers", "5"]],
        cells=[
            [(40.0, 120.0, 270.0, 160.0), (270.0, 120.0, 500.0, 160.0)],
            [(40.0, 160.0, 270.0, 200.0), (270.0, 160.0, 500.0, 200.0)],
        ],
    )
    doc = _bound_doc(
        [
            _page(
                [_line(["Feature", "State"], top=130.0), _line(["Servers", "5"], top=170.0)],
                height=595.0,
                tables=[table],
            )
        ]
    )

    cells = doc.blocks[0].meta["pdf"]["cells"]
    assert [len(row) for row in cells] == [2, 2], "one cell per cell of the Markdown table"
    assert cells[0][0][:4] == [40.0, 120.0, 270.0, 160.0]
    assert cells[1][1][:4] == [270.0, 160.0, 500.0, 200.0]
    assert doc.blocks[0].meta["pdf"]["kind"] == "table"


def test_merged_columns_merge_their_boxes():
    """A merged cell repeats its text in every column it spans; its box spans too."""
    table = _StubTable(
        bbox=(40.0, 120.0, 500.0, 200.0),
        rows=[["Feature", "LS", "LS"], ["Servers", "5", "5"]],
        cells=[
            [
                (40.0, 120.0, 300.0, 160.0),
                (300.0, 120.0, 400.0, 160.0),
                (400.0, 120.0, 500.0, 160.0),
            ],
            [
                (40.0, 160.0, 300.0, 200.0),
                (300.0, 160.0, 400.0, 200.0),
                (400.0, 160.0, 500.0, 200.0),
            ],
        ],
    )
    rows, cells = _table_grid(table)
    assert rows == [["Feature", "LS"], ["Servers", "5"]]
    assert [row[1] for row in cells] == [
        (300.0, 120.0, 500.0, 160.0),
        (300.0, 160.0, 500.0, 200.0),
    ]


def test_cell_typography_comes_from_the_source_lines():
    """Each cell is set the way the source set it, header row included."""
    table = _StubTable(
        bbox=(40.0, 120.0, 500.0, 200.0), rows=[["Feature", "State"], ["Servers", "5"]]
    )
    doc = _bound_doc(
        [
            _page(
                [
                    _line(["Feature", "State"], top=130.0, x0=40.5, size=12.0, font="Verdana-Bold"),
                    _line(["Servers", "5"], top=170.0, x0=42.0, size=9.0),
                ],
                height=595.0,
                tables=[table],
            )
        ]
    )
    header, body = doc.blocks[0].meta["pdf"]["cells"]
    assert header[0][5] == 12.0 and header[0][7] is True  # the source line's size and weight
    assert body[0][5] == 9.0 and body[0][7] is False
    assert header[0][4] == 0.5  # the source's own inset, held at the floor


def test_a_cell_without_a_box_is_null():
    table = _StubTable(
        bbox=(40.0, 120.0, 500.0, 200.0),
        rows=[["Feature", "State"], ["Servers", "5"]],
        cells=[
            [(40.0, 120.0, 270.0, 160.0), None],
            [(40.0, 160.0, 270.0, 200.0), (270.0, 160.0, 500.0, 200.0)],
        ],
    )
    _rows, cells = _table_grid(table)
    assert cells[0][1] is None and cells[1][1][:2] == (270.0, 160.0)


def test_a_row_whose_cell_count_disagrees_with_extract_gets_no_boxes():
    """A phantom column must not be lined up against the wrong cell."""
    table = _StubTable(
        bbox=(40.0, 120.0, 500.0, 200.0),
        rows=[["Feature", "State"], ["Servers", "5"]],
        cells=[
            [(40.0, 120.0, 500.0, 160.0)],
            [(40.0, 160.0, 270.0, 200.0), (270.0, 160.0, 500.0, 200.0)],
        ],
    )
    _rows, cells = _table_grid(table)
    assert cells == [
        [None, None],
        [(40.0, 160.0, 270.0, 200.0), (270.0, 160.0, 500.0, 200.0)],
    ]


def test_a_table_split_across_pages_gets_one_block_per_page():
    pages = [
        _page(
            [_line(["Alpha", "1"], top=130.0)],
            height=595.0,
            tables=[_StubTable((40.0, 120.0, 500.0, 200.0), [["Alpha", "1"], ["Beta", "2"]])],
        ),
        _page(
            [_line(["Gamma", "3"], top=130.0)],
            height=595.0,
            tables=[_StubTable((40.0, 120.0, 500.0, 200.0), [["Gamma", "3"], ["Delta", "4"]])],
        ),
    ]
    doc = _bound_doc(pages)
    tables = [block for block in doc.blocks if block.meta.get("table")]
    assert len(tables) == 2
    assert tables[0].text.startswith("| Alpha | 1 |")
    assert tables[1].text.startswith("| Gamma | 3 |")
    assert tables[1].meta["pdf"]["cells"][1][1][:2] == [270.0, 160.0]


def test_paragraph_gap_splits_but_line_spacing_does_not():
    lines = [
        _line(["First", "line", "of", "the", "paragraph."], top=127.0),
        _line(["Wrapped", "second", "line."], top=142.2),
        _line(["A", "brand", "new", "paragraph."], top=169.0),
    ]
    md = pages_to_markdown([_page(lines)])
    assert "First line of the paragraph. Wrapped second line." in md
    assert md.count("\n\n") == 1
    assert md.endswith("A brand new paragraph.")


def test_a_diagram_page_measures_its_prose_and_not_its_figure_labels():
    """The labels of a figure are set smaller and tighter than the prose around
    them, and on a diagram page they can outnumber the prose lines outright: the
    book this was written against gives eleven 8.45 pt gaps between 7.2 pt labels
    and twelve 11.96 pt gaps between 10 pt prose lines on page 11.

    A yardstick taken from every gap on the page then sat at 8.45 pt, *under* the
    real 11.96 pt leading — and since a break is "more than 1.35 x the yardstick",
    every line of every real paragraph read as a fresh paragraph: the three-line
    paragraph below arrived as three blocks, and each list item as one block per
    line. The rhythm is the gap that repeats between lines of the page's own body
    size, so the labels' tighter rhythm never answers for the prose.
    """
    labels = [
        _line(["JRuby", "JRuby"], top=100.0, size=9.0),
        _line(["Wrappers"], top=108.45, size=9.0),
        _line(["Clojure"], top=116.9, size=9.0),
        _line(["Sesame", "APIs"], top=125.35, size=9.0),
        _line(["RDF"], top=133.8, size=9.0),
    ]
    prose = [
        _line(_words("First line of the paragraph, which runs on to the edge."), top=200.0),
        _line(_words("Wrapped second line of it."), top=215.2),
        _line(_words("Third line of it."), top=230.4),
        _line(_words("Fourth line of it."), top=245.6),
        _line(_words("Fifth line of it."), top=260.8),
        _line(_words("A brand new paragraph starts here."), top=288.0),
    ]
    page = _page(labels + prose)
    assert abs(_line_spacing(_page_lines(page, 0)) - 15.2) < 0.05
    blocks = pages_to_markdown([page]).split("\n\n")
    paragraph = next(b for b in blocks if b.startswith("First line of the paragraph"))
    assert paragraph == (
        "First line of the paragraph, which runs on to the edge. Wrapped second line "
        "of it. Third line of it. Fourth line of it. Fifth line of it."
    )
    assert "A brand new paragraph starts here." in blocks


def test_a_figure_label_is_its_own_block_and_a_footnote_is_not():
    """A figure's labels are set smaller *and* narrower than the body text, and a
    run of them read as one paragraph owns a scatter of rectangles: the rebuilt
    PDF then prints one translation across the whole scatter. Each label is a
    block of its own instead.

    Being small is not enough to make a line a label. A note is set smaller too,
    and its last line is short and narrow — just like a label's. What tells them
    apart is that the note's sentence started on the line above: a line that
    reads on from the one above it is never a label, and one whose line above
    was left mid-sentence is a continuation whichever letter it opens with.
    """
    page = _page(
        [
            _line(_words("Body line one that spans the text block."), top=100.0),
            _line(_words("Body line two of the same paragraph here."), top=115.2),
            _line(_words("Body line three of the same paragraph."), top=130.4),
            _line(_words("Body line four of the same paragraph."), top=145.6),
            _line(["RDF"], top=200.0, size=9.0),
            _line(["Filters"], top=208.45, size=9.0),
            _line(["Store"], top=216.9, size=9.0),
            # A note whose last line is short, narrow, and starts lowercase.
            _line(_words("Footnote one with a full measure of words and"), top=260.0, size=10.5),
            _line(_words("matching tail"), top=275.2, size=10.5),
            # ... and one whose last line opens with a capital, the line above it
            # having been left mid-sentence.
            _line(_words("Footnote two refers to RDF Examples"), top=300.0, size=10.5),
            _line(["Apress", "2009"], top=315.2, size=10.5),
        ]
    )
    assert pages_to_markdown([page]).split("\n\n") == [
        (
            "Body line one that spans the text block. Body line two of the same "
            "paragraph here. Body line three of the same paragraph. Body line four of "
            "the same paragraph."
        ),
        "RDF",
        "Filters",
        "Store",
        "Footnote one with a full measure of words and matching tail",
        "Footnote two refers to RDF Examples Apress 2009",
    ]


def test_hyphenated_line_break_is_rejoined():
    lines = [
        _line(["the", "previ-"], top=127.0),
        _line(["ous", "chapter."], top=142.2),
    ]
    md = pages_to_markdown([_page(lines)])
    assert "the previous chapter." in md


def test_bullet_lines_become_list_items_with_wrapped_continuation():
    lines = [
        # The PDF repeats the bullet glyph before the item text.
        _line(["(cid:136)(cid:136)(cid:136)", "Constant", "time:"], top=200.0, font=BULLET),
        _line(["depend", "on", "the", "size."], top=215.2),
        _line(["(cid:136)(cid:136)(cid:136)", "Linear:"], top=240.0, font=BULLET),
    ]
    md = pages_to_markdown([_page(lines)])
    assert md.split("\n") == [
        "- Constant time: depend on the size.",
        "- Linear:",
    ]


# ------------------------------ glyph artefacts -----------------------------
def test_normalize_glyphs_collapses_doubled_ligature_and_expands():
    """This PDF doubles the ligature glyph and writes it as one codepoint."""
    assert normalize_glyphs("the eﬀﬀect of Proﬁﬁling") == "the effect of Profiling"


# ------------------------------- contents page -------------------------------
def test_toc_entry_keeps_its_text_and_loses_only_the_leader():
    """The leader is layout; the entry is content the reader asked to translate."""
    assert normalize_toc_entry("1.3 The List interface . . . . . 3") == "1.3 The List interface … 3"
    md = "# Contents\n\n1.3 The List interface . . . . . . . . . 3\n\nReal body text.\n"
    out = normalize_toc_lines(md)
    assert "The List interface" in out
    assert "Real body text." in out
    assert "....." not in out
    assert "The List interface … 3" in out


def test_contents_entries_survive_the_extractor():
    """Every entry must reach the parser, whether or not it has a page number."""
    toc = [
        _line(["0.1", "Prerequisites", ".", ".", ".", ".", ".", "xii"], top=200.0),
        _line(["1", "Interfaces", ".", ".", ".", ".", ".", ".", "1"], top=218.0),
        _line(["1.1", "List", "ADT"], top=236.0),
    ]
    md = normalize_toc_lines(pages_to_markdown([_page(toc)]))
    assert "Prerequisites" in md
    assert "Interfaces" in md
    assert "List ADT" in md
    assert "........." not in md
    assert md.count("…") == 2  # both leaders collapsed, neither entry dropped


def test_roman_numbered_entry_is_not_glued_to_the_line_above():
    """Front matter is roman-numbered; such an entry used to read as body text,
    so it was appended to the previous paragraph and two entries became one."""
    lines = [
        _line(["Preface", ".", ".", ".", ".", ".", ".", "xi"], top=200.0),
        _line(["0.1", "Prerequisites", ".", ".", ".", ".", ".", "xii"], top=218.0),
    ]
    md = pages_to_markdown([_page(lines)])
    assert "Preface . . . . . . xi" in md
    assert "xi 0.1" not in md
    assert md.count("0.1") == 1


# --------------------------- source appearance ------------------------------
# What the rebuild needs to replace text without changing the page: the face it
# was set in, its colour, and the bullets a list's items carried.


def _geometry(markdown: str, tmp_path) -> dict[int, dict]:
    from pdfdoc import write_pdf

    doc = load_document(write_pdf(tmp_path / "faces.pdf", markdown))
    return {i: (b.meta.get("pdf") or {}) for i, b in enumerate(doc.blocks)}


def test_a_block_records_the_face_and_colour_the_source_set_it_in(tmp_path):
    geo = _geometry("# Tiêu đề\n\nThân bài bình thường.\n", tmp_path)
    heading, body = geo[0], geo[1]
    assert heading["font"] == "Helvetica-Bold"
    assert heading["bold"] is True and heading["italic"] is False
    assert heading["color"] == (0.0, 0.0, 0.0)
    assert body["font"] == "Helvetica"
    assert body["bold"] is False


def test_a_list_records_the_marker_of_every_item(tmp_path):
    geo = _geometry("Một câu mở đầu.\n\n- mục một\n- mục hai\n", tmp_path)
    assert geo[1]["kind"] == "list"
    assert geo[1]["markers"] == ["-", "-"]


def test_a_numbered_list_records_its_labels(tmp_path):
    geo = _geometry("Một câu mở đầu.\n\n1. mục đầu\n2. mục sau\n", tmp_path)
    assert geo[1]["kind"] == "list"
    assert geo[1]["markers"] == ["1.", "2."]


# ------------------------- emphasis spans in a line -------------------------
#: Faces whose names `_font_style` reads as bold/italic (it looks for substrings,
#: which is what a real PDF carries: "ABCDEF+Verdana-Bold").
BOLD_FACE = "STUVWX+Verdana-Bold"
ITALIC_FACE = "YZABCD+Verdana-Italic"


def _styled_line(
    parts: list[tuple[list[str], str]], top: float, x0: float = 100.0, size: float = 12.0
) -> dict:
    """One source line set in several styles: each part is (words, face).

    pdfplumber reports one flat list of glyphs per line, each carrying its own
    font, so a mixed-style line is exactly this: glyphs that disagree about their
    face, with a word gap between the runs.
    """
    chars: list[dict] = []
    x = x0
    for tokens, font in parts:
        piece = _chars(tokens, x, size, font)
        chars.extend(piece)
        x = max(ch["x1"] for ch in piece) + 3.0
    return {"chars": chars, "top": top}


def test_a_mixed_style_line_marks_only_the_bold_span():
    page = _StubPage(
        [_styled_line([(["Alpha"], BODY), (["Beta"], BOLD_FACE), (["Gamma"], BODY)], top=150.0)],
        height=595.0,
    )
    assert _build_units([page])[0].text == "Alpha ⟦BOLD⟧Beta⟦/BOLD⟧ Gamma"


def test_an_italic_span_is_marked_with_its_own_pair():
    page = _StubPage(
        [_styled_line([(["Alpha"], BODY), (["Beta"], ITALIC_FACE), (["Gamma"], BODY)], top=150.0)],
        height=595.0,
    )
    assert _build_units([page])[0].text == "Alpha ⟦ITALIC⟧Beta⟦/ITALIC⟧ Gamma"


def test_a_uniform_style_line_gets_no_tokens():
    page = _StubPage([_line(["Alpha", "Beta", "Gamma"], top=150.0)], height=595.0)
    assert "⟦" not in _build_units([page])[0].text


def test_a_bold_span_wrapped_over_two_lines_becomes_one_pair():
    """Cụm đậm vắt qua mối nối dòng phải thành MỘT cặp, không hai."""
    page = _StubPage(
        [
            _styled_line([(["Alpha"], BODY), (["Beta"], BOLD_FACE)], top=150.0),
            _styled_line([(["Gamma"], BOLD_FACE), (["Delta"], BODY)], top=162.0),
        ],
        height=595.0,
    )
    assert _build_units([page])[0].text == "Alpha ⟦BOLD⟧Beta Gamma⟦/BOLD⟧ Delta"


def test_geometry_and_block_font_are_unchanged():
    """Sentinel là thông tin THÊM: meta["pdf"] phải y như trước."""
    page = _StubPage(
        [_styled_line([(["Alpha"], BODY), (["Beta"], BOLD_FACE)], top=150.0)], height=595.0
    )
    units = _build_units([page])
    doc = parse_markdown(render_markdown(units), source_format="pdf")
    bind_page_geometry(doc, units)
    geo = doc.blocks[0].meta["pdf"]
    # `_face` reads the bare name (it drops the subset prefix), and the first
    # line of this block is the plain one, so the block stays regular.
    assert geo["font"] == BODY.split("+")[-1] and geo["bold"] is False
    assert set(geo) == {"rects", "size", "kind", "font", "bold", "italic", "color", "markers"}
