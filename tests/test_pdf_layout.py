"""Tests for the layout-aware PDF extractor.

The extractor's input is what pdfplumber reports per glyph (text, x-offset,
size, font), so the tests build pdfplumber-shaped lines directly: that exercises
classification, word-gap reconstruction, code indentation and paragraph
grouping without needing a hand-crafted PDF.
"""

from __future__ import annotations

from polyglot_lc.ingest.pdf_layout import (
    line_text,
    looks_like_code,
    normalize_glyphs,
    normalize_toc_entry,
    normalize_toc_lines,
    pages_to_markdown,
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


class _StubTable:
    """Minimal pdfplumber table: a bounding box and a cell matrix."""

    def __init__(self, bbox: tuple[float, float, float, float], rows: list[list[str]]) -> None:
        self.bbox = bbox
        self._rows = rows

    def extract(self) -> list[list[str]]:
        return self._rows


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
