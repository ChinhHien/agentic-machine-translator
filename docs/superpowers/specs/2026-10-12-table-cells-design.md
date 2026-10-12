# Spec: dịch chữ trong bảng của bản PDF dựng lại

- Ngày: 2026-10-12
- Trạng thái: đã tự soát (hợp đồng, ràng buộc, biên, tiêu chí); **chờ duyệt spec**
- Phạm vi: `src/polyglot_lc/ingest/pdf_layout.py`, `src/polyglot_lc/pipeline/pdf_output.py`, `tests/test_pdf_layout.py`, `tests/test_pdf_output.py`
- Sub-project: **B** (PDF reconstruction) trong bảy việc đã tách — phần bảng

## 1. Mục tiêu

Bản PDF dựng lại phải **đọc được bảng bằng tiếng đích mà lưới y hệt bản gốc**. Hiện nay bảng là phần
duy nhất còn nguyên tiếng nguồn trong PDF: chữ trong ô **đã** được dịch (translator đã giữ bảng
Markdown), nhưng tầng dựng PDF cố ý bỏ qua mọi block `meta["table"]` với lý do "ý nghĩa của bảng nằm ở
lưới, dàn lại chữ mà mất cột thì còn tệ hơn". Lý do đó vẫn đúng cho việc **dàn lại**; nó không còn
đúng khi ta in chữ vào **chính từng ô** của lưới gốc.

Sau khi xong, phải trả lời được ba câu, chỉ từ kết quả chạy:

1. **Ô nào đã được thay bằng bản dịch, ô nào còn tiếng gốc, và vì sao?** — đếm được (`RebuildStats`).
2. **Lưới có còn nguyên không?** — đường kẻ và nền là vector art của bản gốc, không tái tạo, không xoá.
3. **Có tốn thêm bao nhiêu?** — **không thêm request LLM nào**; prompt, chunking và ngân sách không đổi.

Tiêu chí thành công (đo được): đọc lại PDF đầu ra bằng PyMuPDF — phần lớn ô mang chữ tiếng Việt, số
đối tượng vector không đổi so với bản gốc, và không có chữ nào nằm ngoài hộp ô của nó.

## 2. Quyết định đã chốt với người dùng

| Quyết định | Chốt |
|---|---|
| Hướng làm | **In lại từng ô lên chính lưới gốc.** Không dịch từng ô bằng request riêng, không vẽ lại lưới mới |
| Ô không đủ chỗ | **Ngắt dòng + thu cỡ chữ, sàn 70%.** Vẫn không vừa thì **ô đó** giữ tiếng gốc |
| Bản dịch sai hình dạng | **Giữ nguyên cả bảng bằng tiếng gốc**, không ghép ô theo vị trí |

Điều ba quyết định này **loại trừ**, và không được lén thêm vào khi code: request LLM theo từng ô;
tái tạo đường kẻ/độ rộng cột; ghép ô theo chỉ số khi số hàng/cột lệch; đổi prompt, verifier hay
critic; đụng vào Markdown, chunker, checkpoint hay web UI; và cuộn lại lịch sử của bảng Markdown đã
dịch (bản dịch vẫn là một chuỗi duy nhất trong checkpoint, như mọi block khác).

## 3. Ràng buộc đã xác minh trong repo

| Sự thật | Ở đâu |
|---|---|
| Bảng được bắt bằng pdfplumber `find_tables()` và phát thành **một block** (kind `table`), text là Markdown pipe table | `pdf_layout.py:_page_tables:593`, `md.py:markdown_table:64` |
| `_table_rows` gộp cột trùng và bỏ hàng rỗng **trước khi** render Markdown | `pdf_layout.py:556` |
| Dòng `_RawLine` của bảng mang `size = 0.0` và hộp bao toàn bảng | `pdf_layout.py:_page_lines:611` |
| `_Unit.take` ghi `rects`/`size`/`font`; `bind_page_geometry` ghi `meta["pdf"]` gồm `rects`, `size`, `kind`, `font`, `bold`, `italic`, `color`, `markers` | `pdf_layout.py:bind_page_geometry:933` |
| `Block` là pydantic `BaseModel`, `meta` là dict tự do | `ingest/model.py:10` |
| Tầng dựng **bỏ qua** block bảng: `kept += 1` và không in gì | `pdf_output.py:485` |
| Docstring module liệt bảng vào phần "deliberately left alone" | `pdf_output.py:19` |
| `_insert` thu cỡ chữ theo bước 5% / 0.25 pt tới sàn `_MIN_SCALE = 0.55`, và có nhánh cuối **tràn xuống lề dưới** | `pdf_output.py:_insert:559`, `_MIN_SCALE:153` |
| `rebuild_pdf` redact theo từng rect của placement rồi gọi `apply_redactions(images=NONE, graphics=LINE_ART_NONE, text=REMOVE)` | `pdf_output.py:rebuild_pdf:616` |
| Translator đã buộc giữ bảng: "dịch chữ trong ô, giữ hàng, giữ `|` và dòng `| --- |`" | `agents/translator.py:78` |
| Critic/Verifier cũng coi hình dạng bảng là bất biến | `agents/critic.py:72`, `agents/verifier.py:74` |
| Đường `.md` đặt `meta["table"] = True` nhưng **không có hình học** | `md.py:165` |

## 4. Hợp đồng dữ liệu: `meta["pdf"]["cells"]`

```python
block.meta["pdf"] = {
    # ... tám khoá cũ giữ nguyên nghĩa ...
    "cells": [[cell, ...], ...],   # chỉ có ở block bảng
}
# cell = None | [x0, top, x1, bottom, pad, size, align, bold, italic]
```

- `cells[i][j]` **1:1** với ô `[i][j]` của `block.text` sau khi bỏ dòng phân cách `| --- |`.
- `x0, top, x1, bottom`: hộp **của ô**, cùng hệ toạ độ và gốc trên-trái như `rects`.
- `pad`: lề trong tính bằng pt; hộp in = hộp ô thụt vào `pad` mỗi phía.
- `size`, `bold`, `italic`: cỡ chữ và dạng chữ **của ô đó** trong bản gốc (hàng tiêu đề thường to/đậm
  hơn hàng dữ liệu, nên không thể lấy theo block — `size` của block bảng là `0.0`).
- `align`: `"left"` / `"center"` / `"right"` — cách ô nguồn canh chữ.
- `None`: ô không có hộp dùng được (pdfplumber trả `None`, ô hợp nhất lạ) → ô đó giữ tiếng gốc.

Bất biến: `cells` chỉ xuất hiện khi `kind == "table"`; số hàng và số cột khớp `block.text`; mọi toạ độ
nằm trong hộp bảng; toàn bộ giá trị là số, chuỗi, `None` hoặc list nên `meta` vẫn (de)serialize được.
Block không phải bảng không có khoá `cells`, và `_page_geometry` (vùng chiếm chỗ) **không** cần biết
tới ô: hộp bảng đã có trong `rects`.

## 5. Ingest: từ hộp bảng tới hộp ô

1. `_table_rows(table)` tách thành `_table_grid(table) -> tuple[list[list[str]], list[list[Box|None]]]`:
   đọc `table.extract()` cho chữ và `table.rows[i].cells[j]` cho hộp (cùng chỉ số), rồi áp **đúng**
   phép chuẩn hoá hiện có lên cả hai ma trận — gộp hai cột thì hộp gộp là bao của hai hộp, hàng rỗng
   bị bỏ thì bỏ cả hộp. Điều kiện "một cột / một hàng thì không phải bảng" và giá trị trả `None` giữ
   nguyên. `markdown_table(rows)` vẫn nhận đúng ma trận chữ như trước, nên **Markdown không đổi**.
2. Với mỗi ô, suy kiểu chữ từ các `_RawLine` **đã phân loại** của trang mà tâm nằm trong hộp ô (đọc
   trước khi chúng bị lọc bỏ vì nằm trong bảng): `size` = cỡ lớn nhất, `bold`/`italic` của dòng đầu,
   `pad` = khoảng cách nhỏ nhất từ dòng tới mép ô kẹp trong `[0.5, 2.0]` pt, `align` = `"center"` nếu
   tâm dòng lệch tâm ô dưới 2 pt, `"right"` nếu dòng dính mép phải mà không dính mép trái, còn lại
   `"left"`. Ô không có dòng nào → `size` = hằng số mới `_CELL_DEFAULT_SIZE = 10.0`, `pad` = 2.0,
   `align` = `"left"`.
3. `_RawLine` thêm slot `cells`; `_page_lines` gắn ma trận hộp cho dòng `kind == "table"`; `_Unit`
   thêm slot `cells`; `emit("table", …)` chuyển tiếp; `bind_page_geometry` ghi `"cells"` khi
   `owned[0].kind == "table"`.

## 6. Dựng PDF: in lại từng ô

1. `placements()`: block bảng rời khỏi nhánh `kept` và đi vào nhánh bảng:
   - không có `cells`, hoặc không có `texts[index]`, hoặc bản dịch không tách được thành bảng, hoặc
     số hàng/cột khác `cells` → `kept += 1` (**giữ nguyên cả bảng**, không ghép lệch);
   - ngược lại, mỗi ô có chữ dịch khác rỗng **và** có hộp → một `_Placement` với `kind = "cell"`:
     `box` = hộp in của ô, `size` = cỡ của ô, `redact = [hộp ô]`, `align` = canh lề của ô, mặt chữ là
     `match_font(<font của block>, <bold của ô>, <italic của ô>)` — tên mặt chữ nằm ở mức block, còn
     độ đậm/nghiêng lấy theo ô — rơi về font mặc định nếu không tìm được, màu theo màu của block. Ô
     có chữ dịch rỗng: không sinh placement, **không redact**;
   - ô có hộp **nằm trong hoặc bằng** hộp của một ô đã sinh placement trước đó **trên cùng trang**
     thì bỏ qua (để không in chồng lên nhau) và không redact riêng — hộp đó đã nằm trong vùng redact
     của ô in rồi, nên không sót chữ nguồn.
2. `_Placement` thêm trường `align` với mặc định `None` = **hành vi hiện nay** (mặc định canh trái, và
   block heading vẫn tự suy "canh giữa" như cũ); chỉ ô đặt tường minh `"left"`/`"center"`/`"right"`,
   và `_insert` dùng giá trị đó khi nó khác `None`.
3. `_insert` định tuyến sang `_insert_cell` khi `kind == "cell"`: cùng vòng lặp thu cỡ chữ, nhưng sàn
   `_CELL_MIN_SCALE = 0.70` và **không** có nhánh tràn xuống lề dưới. Nếu bản dịch không vừa ở sàn:
   in **chữ gốc của chính ô đó** ở cỡ gốc của ô (bản gốc đã in được ở đó), rồi thử tiếp ở sàn nếu vẫn
   không vừa; chữ gốc này lấy từ ma trận chữ mà `block.text` đã cho, nên không cần đo trước khi redact.
   Ô nào phải rơi về chữ gốc thì đếm vào `RebuildStats.cells_kept`; ô không in được cả chữ gốc (bất
   thường) vào `RebuildStats.failures`.
4. `RebuildStats` thêm `cells` (số ô đã in bản dịch) và `cells_kept`; dòng log cuối nêu cả hai.
5. Docstring module: bỏ bảng khỏi danh sách "deliberately left alone", mô tả luật in theo ô và luật
   rơi về chữ gốc. Không đổi gì khác trong tệp.

## 7. Sai lệch, biên và lỗi

| Trường hợp | Hành vi |
|---|---|
| Bản dịch không phải bảng, hoặc số hàng/cột khác `cells` | Giữ nguyên **cả bảng** như hôm nay; `kept += 1` |
| Block bảng không có bản dịch (chunk hỏng, retry cạn) | Giữ nguyên cả bảng |
| Một ô có hộp `None` | Ô đó không in, **không redact** → chữ nguồn của ô còn nguyên |
| Chữ dịch dài hơn ô | Ngắt dòng + thu cỡ chữ tới sàn 70%; vẫn không vừa → in chữ nguồn của ô đó |
| Ô nguồn và chữ dịch đều rỗng | Không in, không redact |
| Hai ô có hộp trùng nhau (merge dọc mà pdfplumber không gộp) | Ô đầu in như thường; ô sau bị bỏ qua khi hộp của nó nằm trong hoặc bằng hộp đã in, để không in chồng |
| Bảng trải qua hai trang | pdfplumber trả hai bảng → hai block, xử lý độc lập; hàng tiêu đề lặp ở trang sau là bảng riêng |
| Bảng đến từ tài liệu `.md` | Không có `meta["pdf"]` → không dựng lại, đúng như hiện nay |
| Ô chứa `⟦CODE_n⟧` | Placeholder đã được pipeline bảo vệ: mã không bị dịch, chữ quanh nó dịch như thường |
| Bảng lồng nhau | Đã bị `_page_tables` loại từ trước; không đổi |

## 8. Kiểm thử

**Ingest** (`tests/test_pdf_layout.py`):
- `test_table_cell_boxes_line_up_with_the_markdown_matrix` — số hàng/cột của `cells` khớp `block.text`
  và toạ độ ô dùng cùng hệ với `rects`.
- `test_merged_columns_merge_their_boxes` — bảng 11 cột đọc thành 6 (cột bị gộp) thì `cells` cũng 6
  cột, hộp là bao của các hộp bị gộp.
- `test_cell_typography_comes_from_the_source_lines` — ô tiêu đề nhận cỡ/đậm của dòng nguồn trong ô.
- `test_a_cell_without_a_box_is_null` — ô pdfplumber trả `None` → phần tử `None`, không raise.
- `test_markdown_output_is_unchanged` — chuỗi Markdown của bảng y như trước khi refactor.

**Output** (`tests/test_pdf_output.py`):
- `test_table_cells_are_placed_cell_by_cell` — mỗi ô có chữ dịch sinh đúng một placement, đúng hộp,
  đúng cỡ và canh lề của ô, và `redact` là hộp ô.
- `test_a_cell_that_cannot_fit_keeps_its_source_text` — hộp hẹp + chữ dịch dài → `_insert_cell` in
  chữ gốc và `cells_kept` tăng.
- `test_a_translated_table_with_a_different_shape_keeps_the_source_table` — bản dịch thiếu một cột →
  không placement nào cho block đó, `kept` tăng.
- `test_an_overlapping_cell_is_printed_once` — hai ô cùng hộp (merge dọc pdfplumber không gộp) → một
  placement, và `redact` của nó phủ cả hộp chung.
- `test_cells_and_code_and_geometryless_blocks_are_kept_as_source` — viết lại test cũ: bảng **không có
  `cells`** (như bảng từ `.md`) vẫn `kept`, code và block không hình học vẫn `kept`.

**Tích hợp** trên PDF thật trong `input/` (ví dụ `eocortex-4.6-specification.pdf`), **không cần LLM và
không cần mạng**: dựng bản dịch giả lập bằng cách thêm một tiền tố cố định vào chữ của mỗi ô, gọi
`rebuild_pdf`, rồi đọc lại bằng PyMuPDF và khẳng định (a) chữ trong ô mang tiền tố đó, (b) số đối
tượng của `page.get_drawings()` **không đổi** so với bản gốc — đây là phép kiểm chứng cho luật
"redaction chỉ xoá chữ, giữ line art và nền", (c) không glyph nào nằm ngoài hộp ô của nó, (d) chữ
nguồn còn lại trong bảng chỉ thuộc các ô đã đếm vào `cells_kept`. Kiểm tra chất lượng dịch thật là
việc chạy tay trên một lượt có API key, không phải test tự động.

Cuối cùng: toàn bộ `pytest -q` và `ruff check` + `ruff format --check` phải sạch.

## 9. Tiêu chí nghiệm thu

1. Bảng trong PDF dựng lại có chữ tiếng đích ở những ô đã in được, và `RebuildStats` nói rõ số ô in
   bản dịch / số ô giữ tiếng gốc.
2. Số đối tượng vector của mỗi trang bảng **không đổi** sau khi dựng lại (lưới và nền còn nguyên).
3. Không có glyph nào nằm ngoài hộp ô mà nó thuộc về (kể cả ô fallback).
4. Bảng sai hình dạng, thiếu hộp, hoặc thiếu bản dịch đều **không** tạo ra chữ lệch ô: cả bảng (hoặc
   riêng ô đó) giữ tiếng gốc và được đếm.
5. Không phát sinh request LLM nào; output `.md`, prompt, chunker, checkpoint và web UI không đổi.
6. Suite và ruff sạch.

## 10. Rủi ro đã biết

- **Bảng trộn hai ngôn ngữ.** Trên tài liệu có ô ngắn (nhãn, số) việc thu 70% là đủ; ô nhiều chữ có
  thể phải giữ tiếng gốc, và tỷ lệ đó đo được bằng `cells_kept`. Nếu tỷ lệ cao trên tài liệu thật thì
  phải **nói ra**, không im lặng: đó là dữ liệu để quyết định có hạ sàn hay không.
- **Số cột của pdfplumber so với `extract()`.** Ở bảng có cột "ảo" hai bên có thể lệch nhau; khi đó
  `cells` không khớp bản dịch và cả bảng được giữ nguyên — an toàn nhưng mất tính năng cho bảng đó.
  Cách xử lý thật (dùng hộp làm nguồn chuẩn thay vì chỉ số) là việc khác, không thuộc bản này.
- **Redaction trên hộp ô.** `graphics=LINE_ART_NONE` được cho là giữ line-art, nhưng điều đó chưa từng
  được kiểm chứng trên một hộp *nằm trong* bảng; test tích hợp ở mục 8(b) là chỗ kiểm chứng. Nếu sai,
  thu hẹp `redact` về đúng bao các dòng chữ của ô thay vì cả hộp ô.
- **Nền ô tô màu.** Nếu nền là đối tượng không thuộc line-art và bị xoá, bảng sẽ mất nền — cùng phép
  kiểm chứng (b) và cùng cách khắc phục.

## 11. Việc để sau (không thuộc bản này)

- Canh phải chữ số trong ô theo **kiểu số**, thay vì chỉ theo vị trí của bản gốc.
- Bảng không có đường kẻ mà pdfplumber không bắt được: vẫn là chữ thường, không phải bảng.
- Nói cho người đọc biết ô nào giữ tiếng gốc (ví dụ một mục trong khối "Vấn đề" của web UI).
- Mặt chữ riêng cho từng ô (hiện dùng mặt chữ của block; bảng thật gần như luôn cùng một face).
