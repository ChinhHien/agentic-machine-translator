# Dịch chữ trong bảng của PDF dựng lại — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bản PDF dựng lại mang bản dịch trong từng ô của bảng, trên đúng lưới (đường kẻ, nền, cột) của bản gốc.

**Architecture:** Ingest ghi thêm hộp, lề trong, cỡ chữ, dạng chữ và canh lề của **từng ô** vào `block.meta["pdf"]["cells"]`. Tầng dựng PDF thôi bỏ qua block bảng: nó tách bản dịch Markdown (vốn đã có) thành ma trận ô, rồi in mỗi ô vào hộp của chính nó — `_insert_cell` thu cỡ chữ tới sàn 70% và rơi về chữ gốc của ô khi vẫn không vừa. Đường kẻ và nền là vector art của bản gốc nên không bị tái tạo, chỉ chữ bị thay. Không phát sinh request LLM nào.

**Tech Stack:** Python 3.11, pdfplumber (đọc), PyMuPDF (dựng), pydantic (Document/Block), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-12-table-cells-design.md`

## Global Constraints

- Không thêm dependency nào; không đổi prompt, chunker, checkpoint, output `.md` hay web UI.
- Không phát sinh request LLM nào cho việc này.
- Tám khoá cũ của `meta["pdf"]` (`rects`, `size`, `kind`, `font`, `bold`, `italic`, `color`, `markers`) giữ nguyên nghĩa; chỉ thêm `cells`.
- Mọi giá trị trong `cells` là số, chuỗi, `None` hoặc list (để `meta` vẫn (de)serialize được).
- Toạ độ dùng đúng hệ của `rects`: `(page, x0, top, x1, bottom)`, gốc trên-trái, đơn vị pt.
- Sàn cỡ chữ của ô: `_CELL_MIN_SCALE = 0.70`. Sàn của block thường vẫn là `_MIN_SCALE = 0.55`.
- Hằng số mới: `_CELL_DEFAULT_SIZE = 10.0`, lề trong kẹp `[0.5, 2.0]` pt, sai số canh giữa `2.0` pt.
- Test mới không được đọc `input/` (thư mục này không được git theo dõi), không cần mạng, không cần API key. Test cần font Unicode dùng fixture `unicode_font` sẵn có (tự skip khi máy không có font).
- Mỗi commit kết thúc bằng footer:
  `🤖 Generated with Codebuff` / `Co-Authored-By: Codebuff <noreply@codebuff.com>`

## Review Focus

Năm trường hợp spec ngụ ý nhưng chưa thành test riêng — mỗi dòng có test ở task sở hữu:

1. `table.rows[i].cells` không khớp số cột của `extract()` (cột "ảo") → hàng đó giữ tiếng gốc, không lệch ô (**Task 1**).
2. Ô có hộp `None` (merge lạ, pdfplumber không cho hộp) → ô đó giữ tiếng gốc, không crash (**Task 1**, **Task 2**).
3. Chữ dịch chứa dấu `|` → tách thành nhiều ô → sai hình dạng → **giữ nguyên cả bảng** (**Task 2**).
4. Ô nguồn có chữ nhưng chữ dịch rỗng → không in, **không redact**, chữ nguồn còn nguyên (**Task 2**).
5. Ô sát đáy trang với chữ dịch dài → không tràn khỏi ô và không tràn khỏi trang (**Task 3**).

---

### Task 1: Ingest ghi hộp và kiểu chữ của từng ô

**Files:**
- Modify: `src/polyglot_lc/ingest/pdf_layout.py`
- Test: `tests/test_pdf_layout.py`

**Interfaces:**
- Consumes: không (task đầu).
- Produces:
  - `_table_grid(table) -> tuple[list[list[str]], list[list[Cell | None]]] | None` — thay `_table_rows`; `Cell = tuple[float, float, float, float]`.
  - `_Table` (dataclass): `bbox: tuple[float, float, float, float]`, `markdown: str`, `boxes: list[list[Cell | None]]`.
  - `_page_tables(page, page_no) -> list[_Table]` (trước đây trả tuple 5 phần).
  - `_cell_entry(lines: list[_RawLine], box: Cell | None) -> list | None` → `[x0, top, x1, bottom, pad, size, align, bold, italic]`.
  - `metadata: block.meta["pdf"]["cells"]` = `list[list[list | None]]`, 1:1 với ô của `block.text`.
  - Slot `cells` trên `_RawLine` và `_Unit`.

- [ ] **Step 1: Mở rộng stub và viết test thất bại**

Thêm vào `_StubTable` trong `tests/test_pdf_layout.py` (giữ nguyên `extract()` cũ để các test hiện có chạy được):

```python
class _Row:
    """pdfplumber's TableRow: chỉ có `cells` mà extractor đọc."""

    def __init__(self, cells: list) -> None:
        self.cells = cells


def _grid(bbox, nrows: int, ncols: int) -> list[list[tuple]]:
    """Chia đều hộp bảng thành lưới, cho test nào không cần hộp chính xác."""
    x0, top, x1, bottom = bbox
    dx, dy = (x1 - x0) / ncols, (bottom - top) / nrows
    return [
        [(x0 + c * dx, top + r * dy, x0 + (c + 1) * dx, top + (r + 1) * dy) for c in range(ncols)]
        for r in range(nrows)
    ]


class _StubTable:
    def __init__(self, bbox, rows, cells: list[list[tuple | None]] | None = None) -> None:
        self.bbox = bbox
        self._rows = rows
        self._cells = cells if cells is not None else _grid(bbox, len(rows), len(rows[0]))

    def extract(self) -> list[list[str]]:
        return self._rows

    @property
    def rows(self) -> list[_Row]:
        return [_Row(row) for row in self._cells]
```

Và test đầu tiên:

```python
def test_table_cell_boxes_line_up_with_the_markdown_matrix():
    table = _StubTable(
        bbox=(40.0, 120.0, 500.0, 200.0),
        rows=[["Feature", "State"], ["Servers", "5"]],
        cells=[
            [(40.0, 120.0, 270.0, 160.0), (270.0, 120.0, 500.0, 160.0)],
            [(40.0, 160.0, 270.0, 200.0), (270.0, 160.0, 500.0, 200.0)],
        ],
    )
    page = _page(
        [_line(["Feature", "State"], top=130.0), _line(["Servers", "5"], top=170.0)],
        height=595.0,
        tables=[table],
    )
    units = _build_units([page])
    doc = parse_markdown(render_markdown(units), source_format="pdf")
    bind_page_geometry(doc, units)

    cells = doc.blocks[0].meta["pdf"]["cells"]
    assert [len(row) for row in cells] == [2, 2], "một ô cho mỗi ô của bảng Markdown"
    assert cells[0][0][:4] == [40.0, 120.0, 270.0, 160.0]
    assert cells[1][1][:4] == [270.0, 160.0, 500.0, 200.0]
    assert doc.blocks[0].meta["pdf"]["kind"] == "table"
```

Bổ sung import: `parse_markdown` từ `polyglot_lc.ingest.md`, và `_build_units`, `bind_page_geometry`, `render_markdown` từ `polyglot_lc.ingest.pdf_layout`.

- [ ] **Step 2: Chạy test để chắc nó thất bại**

Run: `python -m pytest tests/test_pdf_layout.py::test_table_cell_boxes_line_up_with_the_markdown_matrix -v`
Expected: FAIL — `KeyError: 'cells'`.

- [ ] **Step 3: Cài đặt phần ingest**

Trong `src/polyglot_lc/ingest/pdf_layout.py`:

- hằng số mới cạnh các hằng số khác: `_CELL_DEFAULT_SIZE = 10.0`, `_CELL_PAD_MIN = 0.5`, `_CELL_PAD_MAX = 2.0`, `_CELL_PAD_DEFAULT = 2.0`, `_CELL_ALIGN_TOLERANCE = 2.0`;
- `_table_rows` → `_table_grid`: giữ nguyên đường đi chuẩn hoá hiện có (gộp cột trùng, bỏ hàng rỗng, trả `None` khi <2 hàng hoặc <2 cột), nhưng đọc `table.extract()` cho chữ và `table.rows[i].cells[j]` cho hộp, rồi áp **cùng** phép gộp cột và bỏ hàng lên ma trận hộp (gộp cột → hộp là bao của các hộp bị gộp). Nếu `table` không có `rows`, hoặc một hàng có số ô khác số ô chữ của hàng đó, thì hàng đó nhận `None` cho mọi ô (giữ tiếng gốc) thay vì ghép sai;
- `_cell_entry(lines, box)`: `None` khi `box is None`; ngược lại lấy các `_RawLine` có tâm trong `box`, rồi `size` = max cỡ (mặc định `_CELL_DEFAULT_SIZE` khi không có dòng nào), `bold`/`italic` của dòng đầu, `pad` = min khoảng cách từ dòng tới bốn mép ô kẹp `[_CELL_PAD_MIN, _CELL_PAD_MAX]` (mặc định `_CELL_PAD_DEFAULT`), `align` = `"center"` khi tâm dòng lệch tâm ô dưới `_CELL_ALIGN_TOLERANCE`, `"right"` khi dòng dính mép phải mà không dính mép trái, còn lại `"left"`;
- `_Table` dataclass + `_page_tables` trả `list[_Table]`;
- `_page_lines`: dựng danh sách `_cell_entry` cho từng bảng **trước** khi lọc bỏ các dòng nằm trong bảng, rồi truyền `cells=` vào `_RawLine(...)` của bảng;
- `_RawLine`: thêm `cells` vào `__slots__` và một tham số từ khoá `cells: list | None = None` (`self.cells = cells or []`); `_Unit`: thêm slot `cells` (mặc định `[]`) và copy trong `emit("table", …)` / `take` không cần đổi;
- `bind_page_geometry`: ghi thêm `"cells": owned[0].cells` **chỉ khi** `owned[0].kind == "table"` và `owned[0].cells` khác rỗng.

- [ ] **Step 4: Chạy test để chắc nó pass**

Run: `python -m pytest tests/test_pdf_layout.py -v`
Expected: PASS, cả file (không test cũ nào đổi kết quả). Phép kiểm chứng "Markdown không đổi"
(spec §8) chính là test cũ `test_table_is_emitted_as_markdown_and_not_repeated_as_prose`: nó so chuỗi
Markdown chính xác, nên refactor mà làm đổi chữ sẽ làm nó đỏ.

- [ ] **Step 5: Thêm bốn test biên của task này**

```python
def test_merged_columns_merge_their_boxes():
    """Cột bị gộp (ô merge lặp chữ ở mọi cột) gộp cả hộp."""
    table = _StubTable(
        bbox=(40.0, 120.0, 500.0, 200.0),
        rows=[["Feature", "LS", "LS"], ["Servers", "5", "5"]],
        cells=[
            [(40.0, 120.0, 300.0, 160.0), (300.0, 120.0, 400.0, 160.0), (400.0, 120.0, 500.0, 160.0)],
            [(40.0, 160.0, 300.0, 200.0), (300.0, 160.0, 400.0, 200.0), (400.0, 160.0, 500.0, 200.0)],
        ],
    )
    md, cells = _table_grid(table)
    assert md == [["Feature", "LS"], ["Servers", "5"]]
    assert [row[1][:4] for row in cells] == [
        [300.0, 120.0, 500.0, 160.0],
        [300.0, 160.0, 500.0, 200.0],
    ]


def test_cell_typography_comes_from_the_source_lines():
    table = _StubTable(bbox=(40.0, 120.0, 500.0, 200.0), rows=[["Feature", "State"], ["Servers", "5"]])
    page = _page(
        [
            _line(["Feature", "State"], top=130.0, x0=46.0, size=12.0, font=BOLD),
            _line(["Servers", "5"], top=170.0, x0=42.0, size=9.0),
        ],
        height=595.0,
        tables=[table],
    )
    units = _build_units([page])
    doc = parse_markdown(render_markdown(units), source_format="pdf")
    bind_page_geometry(doc, units)
    header, body = doc.blocks[0].meta["pdf"]["cells"]
    assert header[0][5] == 12.0 and header[0][7] is True   # cỡ và đậm của dòng nguồn
    assert body[0][5] == 9.0 and body[0][7] is False
    assert header[0][4] == 0.5                              # lề trong suy từ dòng nguồn, kẹp sàn


def test_a_cell_without_a_box_is_null():
    table = _StubTable(
        bbox=(40.0, 120.0, 500.0, 200.0),
        rows=[["Feature", "State"], ["Servers", "5"]],
        cells=[[(40.0, 120.0, 270.0, 160.0), None], [(40.0, 160.0, 270.0, 200.0), (270.0, 160.0, 500.0, 200.0)]],
    )
    md, cells = _table_grid(table)
    assert cells[0][1] is None


def test_a_row_whose_cell_count_disagrees_with_extract_gets_no_boxes():
    """Cột "ảo" của pdfplumber không được ghép bừa vào ô sai."""
    table = _StubTable(
        bbox=(40.0, 120.0, 500.0, 200.0),
        rows=[["Feature", "State"], ["Servers", "5"]],
        cells=[[(40.0, 120.0, 500.0, 160.0)], [(40.0, 160.0, 270.0, 200.0), (270.0, 160.0, 500.0, 200.0)]],
    )
    md, cells = _table_grid(table)
    assert cells == [None, [(40.0, 160.0, 270.0, 200.0), (270.0, 160.0, 500.0, 200.0)]]


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
    units = _build_units(pages)
    doc = parse_markdown(render_markdown(units), source_format="pdf")
    bind_page_geometry(doc, units)
    tables = [block for block in doc.blocks if block.meta.get("table")]
    assert len(tables) == 2
    assert tables[0].meta["pdf"]["cells"][0][0][0] == 40.0
    assert tables[1].meta["pdf"]["cells"][1][1][0] == 270.0 + 90.0  # ô của trang thứ hai
```

- [ ] **Step 6: Chạy cả file test**

Run: `python -m pytest tests/test_pdf_layout.py -v`
Expected: PASS (5 test mới + toàn bộ test cũ).

- [ ] **Step 7: Commit**

```bash
git add src/polyglot_lc/ingest/pdf_layout.py tests/test_pdf_layout.py
git commit -m "feat(ingest): record each table cell's box, size and alignment"
```

---

### Task 2: Dựng PDF sinh placement cho từng ô

**Files:**
- Modify: `src/polyglot_lc/pipeline/pdf_output.py`
- Test: `tests/test_pdf_output.py`

**Interfaces:**
- Consumes: `block.meta["pdf"]["cells"]` từ Task 1.
- Produces:
  - `_markdown_cells(text) -> list[list[str]] | None` — tách bảng Markdown thành ma trận ô.
  - `_Placement.align: str | None` (mặc định `None` = hành vi hiện nay) và `_Placement.source_text: str | None`.
  - `placements()` sinh `_Placement(kind="cell", …)` cho từng ô in được, với `box` = hộp in (hộp ô thụt vào `pad`), `redact` = danh sách một phần tử là **hộp ô**, `align`, `size`, `source_text` = chữ gốc của ô.

- [ ] **Step 1: Viết các test thất bại**

Thêm vào `tests/test_pdf_output.py` (fixture block dùng chung cho cả task này):

```python
def _table_block() -> Block:
    """Một bảng 2x2 đã có hình học và hộp ô, như ingest thật ghi lại."""
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


def test_table_cells_are_placed_cell_by_cell():
    doc = Document(blocks=[_table_block()])
    items, kept = placements(doc, {0: "| Tính năng | Trạng thái |\n| --- | --- |\n| Máy chủ | 5 |"})

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
    items, kept = placements(Document(blocks=[_table_block()]), {0: "| Tính năng |\n| --- |\n| Máy chủ |"})
    assert items == [] and kept == 1


def test_a_translation_that_splits_a_cell_on_a_pipe_keeps_the_table():
    text = "| a | b |\n| --- | --- |\n| c | d | e |"  # dấu | trong chữ dịch sinh thêm một ô
    items, kept = placements(Document(blocks=[_table_block()]), {0: text})
    assert items == [] and kept == 1


def test_a_cell_without_translation_is_not_printed():
    items, kept = placements(
        Document(blocks=[_table_block()]), {0: "| a |  |\n| --- | --- |\n| c | d |"}
    )
    assert [item.text for item in items] == ["a", "c", "d"]
    assert [item.redact for item in items][1] == [(40.0, 146.0, 270.0, 172.0)]


def test_a_table_without_a_translation_is_kept():
    items, kept = placements(Document(blocks=[_table_block()]), {})
    assert items == [] and kept == 1


def test_a_cell_holding_a_code_placeholder_prints_it_as_text():
    items, kept = placements(
        Document(blocks=[_table_block()]),
        {0: "| ⟦CODE_1⟧ | b |\n| --- | --- |\n| c | d |"},
    )
    assert kept == 0 and items[0].text == "⟦CODE_1⟧"


def test_a_cell_without_a_box_is_left_alone():
    doc = Document(blocks=[_table_block()])
    doc.blocks[0].meta["pdf"]["cells"][1][1] = None
    items, _ = placements(doc, {0: "| a | b |\n| --- | --- |\n| c | d |"})
    assert [item.text for item in items] == ["a", "b", "c"]


def test_an_overlapping_cell_is_printed_once():
    doc = Document(blocks=[_table_block()])
    cells = doc.blocks[0].meta["pdf"]["cells"]
    cells[1][0] = list(cells[0][0])  # merge dọc mà pdfplumber không gộp
    items, _ = placements(doc, {0: "| a | b |\n| --- | --- |\n| c | d |"})
    assert [item.text for item in items] == ["a", "b", "d"]
```

Và viết lại test cũ `test_tables_code_and_geometryless_blocks_are_kept_as_source` — giữ đúng tên spec §8:

```python
def test_cells_and_code_and_geometryless_blocks_are_kept_as_source():
    items, kept = placements(_doc_with_geometry(), {0: "Xin chào", 1: "x", 2: "x", 3: "x"})
    assert [item.text for item in items] == ["Xin chào"]
    assert kept == 3
```

- [ ] **Step 2: Chạy test để chắc chúng thất bại**

Run: `python -m pytest tests/test_pdf_output.py -k "cell or table" -v`
Expected: FAIL — `placements` trả `items == []` cho block bảng.

- [ ] **Step 3: Cài đặt**

Trong `src/polyglot_lc/pipeline/pdf_output.py`:

- `_markdown_cells(text)`: giữ các dòng (sau `strip`) bắt đầu bằng `|` và có ít nhất hai dấu `|`; bỏ dòng phân cách (mọi ô khớp `^:?-{3,}:?$`); tách mỗi dòng theo `|`, bỏ phần tử rỗng đầu/cuối, `strip()` và gộp khoảng trắng trong từng ô; trả `None` khi còn dưới 2 hàng hoặc dưới 2 cột;
- `_Placement`: thêm hai trường `align: str | None = None` và `source_text: str | None = None`;
- `placements()`: thay dòng `if block.meta.get("table") …: kept += 1; continue` bằng một hàm `_table_placements(geo, block, raw, match_style, default_font, printed_by_page)` trả `list[_Placement] | None`; `None` nghĩa là "giữ nguyên cả bảng" (`kept += 1`). Hàm từ chối khi: `cells` rỗng; hình dạng `_markdown_cells(raw)` khác `cells`; mọi ô đều không in được. Với ô in được: bỏ qua khi chữ dịch rỗng hoặc hộp `None`; bỏ qua khi hộp nằm trong hoặc bằng một hộp đã in trước đó **trên cùng trang**; ngược lại tạo `_Placement(kind="cell", page=…, box=hộp thụt `pad`, source_box=hộp ô, redact=[hộp ô], size=size của ô, text=chữ dịch, align=align của ô, source_text=chữ gốc của ô, font_file=match_font(font của block, bold của ô, italic của ô) khi `match_style`, color=màu của block)`;
- giữ nguyên đường đi của block thường, và để `_insert` (Task 3) lo phần in.

- [ ] **Step 4: Chạy test để chắc chúng pass**

Run: `python -m pytest tests/test_pdf_output.py -v`
Expected: PASS, cả file.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/pipeline/pdf_output.py tests/test_pdf_output.py
git commit -m "feat(pdf): place each translated table cell in its own source box"
```

---

### Task 3: In ô với sàn 70% và rơi về chữ gốc

**Files:**
- Modify: `src/polyglot_lc/pipeline/pdf_output.py`, `tests/pdfdoc.py`
- Test: `tests/test_pdf_output.py`

**Interfaces:**
- Consumes: `_Placement(kind="cell", align=…, source_text=…)` từ Task 2.
- Produces: `_CELL_MIN_SCALE = 0.70`; `_insert_cell(page, item, pymupdf, font_file) -> str | None` trả `"translated"`, `"source"` hoặc `None`; `RebuildStats.cells: int` và `RebuildStats.cells_kept: int`; trong `tests/pdfdoc.py`: `write_table_pdf(path, rows, *, col_widths=None, row_height=18.0, left=72.0, top=150.0, size=9.0) -> Path`.

- [ ] **Step 1: Viết `write_table_pdf` trong `tests/pdfdoc.py`**

Helper này là thứ hai test dưới cần, nên làm trước. Yêu cầu: mỗi hàng cao `row_height`, mỗi cột rộng theo `col_widths` (mặc định 90 pt), kẻ **đường ngang ở mọi ranh giới hàng và đường dọc ở mọi ranh giới cột** bằng `page.draw_line` — pdfplumber chỉ bắt được bảng khi có đường kẻ thật — rồi in chữ từng ô bằng `page.insert_text` tại `(ô.x0 + 3, ô.top + 12)` với `fontname="hebo"` cho hàng đầu và `"helv"` cho các hàng sau.

- [ ] **Step 2: Viết các test thất bại**

```python
def _one_cell_pdf(tmp_path, *, text: str = "Feature", width: float = 60.0):
    """PDF một trang với bảng 1x1 đủ rộng cho *text* nhưng không rộng hơn."""
    from pdfdoc import write_table_pdf

    return write_table_pdf(tmp_path / "table.pdf", [[text]], col_widths=[width])


def test_a_cell_that_cannot_fit_keeps_its_source_text(tmp_path, unicode_font):
    pymupdf = pytest.importorskip("pymupdf")
    src = _one_cell_pdf(tmp_path)
    doc = load_document(src)
    long_translation = "| " + "Một trăm hai mươi ký tự dịch rất dài " * 3 + " |"
    texts = {0: "| Feature |\n| --- |\n" + long_translation}

    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.cells == 0 and stats.cells_kept == 1
    assert stats.failures == []
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert "Feature" in rebuilt[0].get_text()


def test_a_cell_does_not_spill_below_the_page(tmp_path, unicode_font):
    """Không có nhánh 'tràn xuống lề dưới' cho ô: chữ phải nằm trong ô."""
    pymupdf = pytest.importorskip("pymupdf")
    src = _one_cell_pdf(tmp_path)
    doc = load_document(src)
    box = doc.blocks[0].meta["pdf"]["cells"][0][0][:4]
    stats = rebuild_pdf(src, doc, {0: "| Nội dung |\n| --- |\n| Nội dung |"}, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.cells == 1
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        for word in rebuilt[0].get_text("words"):
            x0, top, x1, bottom = word[:4]
            assert x0 >= box[0] - 1 and x1 <= box[2] + 1
            assert top >= box[1] - 1 and bottom <= box[3] + 1
```

- [ ] **Step 3: Chạy test để chắc chúng thất bại**

Run: `python -m pytest tests/test_pdf_output.py -k "cannot_fit or spill" -v`
Expected: FAIL — `AttributeError: 'RebuildStats' object has no attribute 'cells'`.

- [ ] **Step 4: Cài đặt**

- `_CELL_MIN_SCALE = 0.70` cạnh `_MIN_SCALE`;
- `_insert_cell(page, item, pymupdf, font_file)`: giống vòng lặp của `_insert` nhưng `floor = max(item.size * _CELL_MIN_SCALE, 4.0)` và **không** có nhánh "dùng nốt chỗ tới lề dưới"; hộp luôn là `item.box`. Vòng 1 thử `item.text` (bản dịch); nếu không vừa, vòng 2 thử `item.source_text` ở cỡ gốc của ô, rồi ở sàn; trả `"translated"`, `"source"`, hoặc `None` khi cả hai đều không vừa;
- `_insert`: khi `item.kind == "cell"` thì gọi `_insert_cell` và trả kết quả đó; các block khác giữ nguyên (đổi kiểu trả về của `_insert` thành `str | None`, `True` → `"block"`);
- `_Placement.align`: khi khác `None`, dùng để chọn `pymupdf.TEXT_ALIGN_LEFT/CENTER/RIGHT` thay cho suy đoán canh giữa hiện có (suy đoán đó vẫn áp cho block heading);
- `rebuild_pdf`: `"translated"` → `stats.cells += 1`, `stats.placed += 1`; `"source"` → `stats.cells_kept += 1`, `stats.placed += 1`; `None` → vào `stats.failures` như hiện nay; dòng log cuối thêm số ô (`{} cell(s) translated, {} kept as source`);
- docstring module: bảng chuyển từ danh sách "deliberately left alone" sang danh sách được dựng lại, nêu rõ hai luật (sàn 70%, rơi về chữ gốc của ô) và điều kiện giữ nguyên cả bảng khi bản dịch sai hình dạng.

- [ ] **Step 5: Chạy test để chắc chúng pass**

Run: `python -m pytest tests/test_pdf_output.py -v`
Expected: PASS, cả file.

- [ ] **Step 6: Commit**

```bash
git add src/polyglot_lc/pipeline/pdf_output.py tests/pdfdoc.py tests/test_pdf_output.py
git commit -m "feat(pdf): print a cell within its box, falling back to the source cell"
```

---

### Task 4: Test tích hợp trên trang bảng thật và README

**Files:**
- Modify: `tests/test_pdf_output.py`, `README.md`
- Test: `tests/test_pdf_output.py`

**Interfaces:**
- Consumes: `rebuild_pdf` (Task 3), `block.meta["pdf"]["cells"]` (Task 1), `write_table_pdf` (Task 3).
- Produces: không có interface mới cho code sản phẩm; chỉ test và tài liệu.

- [ ] **Step 1: Viết test tích hợp**

```python
def test_a_real_table_page_is_rebuilt_with_its_cells_translated(tmp_path, unicode_font):
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

    # Bản dịch giả lập: mỗi ô được thêm tiền tố, giữ nguyên số hàng/cột.
    source_rows = [line for line in table.text.splitlines() if not set(line) <= set("|- ")]
    translated = "\n".join(
        "| " + " | ".join(f"VI:{cell}" for cell in row.strip("| ").split(" | ")) + " |"
        for row in source_rows
    )
    texts = {index: (translated if block is table else block.text) for index, block in enumerate(doc.blocks)}

    stats = rebuild_pdf(src, doc, texts, tmp_path / "out.pdf", font_path=unicode_font)
    assert stats.cells == 6 and stats.cells_kept == 0 and stats.failures == []

    with pymupdf.open(src) as original, pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        assert len(rebuilt[0].get_drawings()) == len(original[0].get_drawings()), "lưới còn nguyên"
        text = rebuilt[0].get_text().replace("\xa0", " ")
        assert "VI:Feature" in text and "VI:Yes" in text
        assert "Feature" not in text.replace("VI:Feature", "")
        boxes = [cell[:4] for row in cells for cell in row]
        for word in rebuilt[0].get_text("words"):
            x0, top, x1, bottom = word[:4]
            assert any(
                x0 >= bx0 - 1 and x1 <= bx2 + 1 and top >= btop - 1 and bottom <= bbottom + 1
                for bx0, btop, bx2, bbottom in boxes
            ), f"chữ nằm ngoài ô: {word[4]!r}"
```

- [ ] **Step 2: Chạy test để chắc nó pass (hoặc thất bại đúng chỗ)**

Run: `python -m pytest tests/test_pdf_output.py::test_a_real_table_page_is_rebuilt_with_its_cells_translated -v`
Expected: PASS. Nếu khẳng định `get_drawings()` không đổi bị FAIL, thu hẹp `redact` của ô về đúng bao các
dòng chữ nguồn (xem spec §10) rồi chạy lại. Ở đây mọi ô đều in được nên `cells_kept == 0`; vế (d)
của spec §8 — "chữ nguồn còn lại chỉ thuộc các ô fallback" — là test của Task 3.

- [ ] **Step 3: Cập nhật README**

Ba chỗ nói bảng giữ nguyên tiếng gốc: dòng về "bảng biểu" ở đầu file, mục mô tả PDF dựng lại (danh sách "hai loại đó vẫn…"), và phần ghi chú so với pdf2zh. Sửa thành: chữ trong ô được dịch và in lại vào đúng ô của lưới gốc; ô không đủ chỗ giữ tiếng gốc; code vẫn là loại duy nhất không được dịch.

- [ ] **Step 4: Chạy toàn bộ suite và lint**

Run: `python -m pytest -q; python -m ruff check src tests webui; python -m ruff format --check src tests webui`
Expected: tất cả PASS/sạch (exit code 0).

- [ ] **Step 5: Commit**

```bash
git add tests/test_pdf_output.py README.md
git commit -m "test(pdf): verify a rebuilt table page keeps its grid and translates its cells"
```
