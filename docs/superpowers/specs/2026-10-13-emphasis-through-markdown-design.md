# Spec: đậm/nghiêng đi theo Markdown tới bản PDF dựng lại

- Ngày: 2026-10-13
- Trạng thái: đã tự soát (placeholder, mâu thuẫn, mơ hồ, phạm vi); **chờ duyệt spec**
- Phạm vi: `src/polyglot_lc/emphasis.py` (mới), `src/polyglot_lc/ingest/pdf_layout.py`,
  `src/polyglot_lc/agents/translator.py`, `src/polyglot_lc/agents/verifier.py`,
  `src/polyglot_lc/pipeline/pdf_output.py`, `src/polyglot_lc/pipeline/assembler.py`,
  `src/polyglot_lc/orchestrator/graph.py`, `tests/`
- Sub-project: **B** (PDF reconstruction) — phần kiểu chữ trong dòng

## 1. Mục tiêu

Bản PDF dựng lại phải **giữ được cụm đậm/nghiêng nằm giữa một câu**. Hiện nay mỗi block chỉ có
**một** kiểu chữ, lấy từ **dòng đầu tiên** của nó, nên một câu như *"**Eocortex ST** được thiết kế
để…"* được in trọn bằng kiểu chiếm phần lớn ký tự và cụm đậm biến mất — đúng thứ người đọc dùng để
nhận ra tên riêng và thuật ngữ.

Ba sự thật đã kiểm chứng trong repo quyết định hình dạng lời giải:

1. Kiểu chữ được chốt ở mức **dòng** rồi bị **rút gọn về dòng đầu** khi ghi vào block
   (`_face` → `_RawLine` → `_Unit.take` → `meta["pdf"]["font"]`). Thông tin kiểu theo *span* bị vứt
   ngay tại `_face`.
2. Đậm **chưa từng** xuất hiện trong chuỗi Markdown đi qua các tầng. Nếu model có phát ra `**`,
   `plain_text("**Eocortex ST** …", "body")` trả về nguyên hai dấu sao, tức trang in sẽ hiện `**`.
   Không có đường nào để "đậm chảy xuống" nếu không sửa hợp đồng chuỗi.
3. `TextWriter.fill_textbox` **không** phân tích markup `<b>`/`<i>`: thử nghiệm cho thấy nó in nguyên
   thẻ ra trang (`'Alpha\xa0<b>beta\xadgamma</b>…'`). Vậy phần in run phải do ta tự làm.

Sau khi xong, phải trả lời được ba câu **chỉ từ kết quả chạy**:

1. **Cụm nào trên trang là đậm/nghiêng?** — đọc lại PDF bằng PyMuPDF và kiểm cờ `span["flags"]`.
2. **Có mất chữ nào không?** — chuỗi ký tự của block sau khi bỏ token phải khớp giữa đường `.md` và
   đường PDF.
3. **Có tốn thêm bao nhiêu?** — **không thêm request LLM nào**: prompt không dài thêm một lượt gọi,
   chunking, ngân sách và checkpoint không đổi.

## 2. Quyết định đã chốt với người dùng

| Quyết định | Chốt |
|---|---|
| Cách biểu diễn | **Sentinel `⟦BOLD⟧…⟦/BOLD⟧`, `⟦ITALIC⟧…⟦/ITALIC⟧`**, theo gương của `⟦CODE_n⟧` |
| Phạm vi kiểu | **Đậm + nghiêng** |
| Cơ chế in | **Cách 1**: chỉ block *có* run mới dùng layout tự viết; block thường giữ nguyên `insert_textbox` |
| Khi sentinel lệch | **Không tốn lượt dịch lại**: bỏ hết token của block đó, in thường, ghi lỗi |
| Ô bảng | **Không đổi trong v1**: mỗi ô vẫn một mặt chữ; token trong ô bị gỡ (xem §8) |

Điều các quyết định này **loại trừ**, và không được lén thêm khi code: đổi `insert_textbox` cho mọi
block; request LLM theo từng span; căn chỉnh (alignment) run theo toạ độ nguồn; đổi `⟦CODE_n⟧`; đụng
vào lưới bảng, checkpoint hay web UI.

### 2.1 Thay đổi so với thiết kế đã duyệt (cần bạn xác nhận khi duyệt spec)

Thiết kế trình bày trước nói "chèn lại theo vị trí nếu đếm được, dùng lại kiểu
`repair_code_placeholders`" và "`hard_checks` thêm đối chiếu multiset sentinel". Đọc code thì cả hai
vế phải sửa:

- `repair_code_placeholders` làm việc **theo đoạn**: nó coi mỗi `⟦CODE_n⟧` là *một đoạn riêng* và chèn
  lại theo số đoạn văn phía trước. Sentinel của ta nằm **trong** một đoạn, nên cơ chế đó không áp
  được; và **chèn lại theo vị trí nguồn là sai về ngữ nghĩa** — bản dịch có thể dài/ngắn khác, nên vị
  trí ký tự của nguồn không tương ứng với bản dịch.
- Vì vậy: khi sentinel **cân bằng và đếm khớp nguồn**, ta **tôn trọng vị trí do model đặt** — model đã
  làm phép căn chỉnh, và đó chính là điều ta muốn. Khi sentinel **lệch hoặc không cân bằng**, ta **gỡ
  toàn bộ token của block đó** và in thường (không mất chữ, không đoán bừa vị trí).
- Và vì quyết định "không tốn lượt dịch lại", đối chiếu sentinel **không** được đưa vào `hard_checks`
  (thứ cổng này điều khiển việc gọi lại). Nó là một phép kiểm riêng ở đường dựng/ghi, có ghi lỗi.

## 3. Ràng buộc đã xác minh trong repo

| Sự thật | Ở đâu |
|---|---|
| `_face(chars)` chọn mặt chữ **chiếm đa số** của một dòng và trả `{font, bold, italic, color}`; thông tin theo span bị mất tại đây | `pdf_layout.py:_face:255` |
| `_RawLine` mang `font`, `bold`, `italic`, `color` của **dòng đó** | `pdf_layout.py:_RawLine:301` |
| `_Unit.take` chỉ nhận bộ kiểu của **dòng đầu tiên** (`if not self.font`) | `pdf_layout.py:_Unit.take:402` |
| Block ghi `meta["pdf"]["font"|"bold"|"italic"]` từ `owned[0]` | `pdf_layout.py:bind_page_geometry:1074`, dòng ghi khoá ở `1109-1112` |
| Văn bản của một unit được dựng ở `flush()` bằng `_join_wrapped([rendered …])` — đây là chỗ duy nhất ghép các dòng nguồn của một đoạn | `pdf_layout.py:_build_units:906`, `flush:932`, `_join_wrapped:829` |
| `render_markdown(units)` ghép các unit thành chuỗi Markdown | `pdf_layout.py:render_markdown:1025` |
| `hard_checks` so **multiset** `⟦CODE_n⟧` và **mức heading** giữa nguồn và bản dịch | `verifier.py:hard_checks:35` |
| `repair_code_placeholders` chèn lại marker theo **số đoạn văn** phía trước, và marker là **cả một đoạn** | `pipeline/placeholders.py:43`, `_prose_offset:29` |
| Lỗi placeholder được ghi qua `ctx.record_error(step=…, cause="placeholder", …)` | `orchestrator/graph.py:collect_node` |
| Translator buộc giữ `⟦CODE_1⟧` trong prompt Rules | `agents/translator.py:72-74` |
| Verifier coi `⟦CODE_n⟧` là một phần của mục "Structure" | `agents/verifier.py:82` |
| `plain_text` chỉ gỡ dấu heading và dấu mục danh sách | `pdf_output.py:plain_text:352` |
| Số phận block là **cả dòng chỉ có `⟦CODE_n⟧`** thì giữ nguyên tiếng nguồn | `pdf_output.py:_CODE_PLACEHOLDER_RE:152`, dùng ở `placements:677` |
| `_insert` in cả block bằng **một** `fontname` qua `insert_textbox`, hạ cỡ tới `_MIN_SCALE` | `pdf_output.py:_insert:814`, `_MIN_SCALE:178` |
| `match_font(name, bold, italic)` đã chọn được mặt đậm/nghiêng cùng họ, có `_covers_vietnamese` | `pdf_output.py:match_font:326` |
| Markdown đích được ghi bằng `write_atomic(out_path, assemble(...))` | `coordinator.py:715`, `786` |
| `segment/` và `pipeline/` đều được `ingest/` và `orchestrator/` dùng chung, nhưng **`ingest/` không import `pipeline/` hay `segment/`** | `grep` trên `src/polyglot_lc/ingest/*.py` |
| Prompt mock nhận diện agent bằng chuỗi con (`"QA verifier"`, `"translation critic"`, `⟦CODE_n⟧`) | `llm/client.py:_mock_json_reply:664`, test ghép ở `tests/test_agents.py` cuối file |

## 4. Hợp đồng dữ liệu: ngữ pháp sentinel

Module **mới** `src/polyglot_lc/emphasis.py`, **không import gì** trong dự án (để cả `ingest/` lẫn
`pipeline/` đều dùng được mà không tạo phụ thuộc vòng):

```python
BOLD_OPEN, BOLD_CLOSE = "⟦BOLD⟧", "⟦/BOLD⟧"
ITALIC_OPEN, ITALIC_CLOSE = "⟦ITALIC⟧", "⟦/ITALIC⟧"
TOKENS: tuple[str, ...]          # bốn token trên
PAIR_RE: re.Pattern[str]         # (⟦BOLD⟧|⟦ITALIC⟧)(.*?)(⟦/BOLD⟧|⟦/ITALIC⟧), không tham lam

def is_balanced(text: str) -> bool
def counts(text: str) -> dict[str, int]        # {"⟦BOLD⟧": n, "⟦/BOLD⟧": m, ...}
def strip(text: str) -> str                    # gỡ mọi token, giữ nguyên phần chữ
def to_markdown(text: str) -> str              # ⟦BOLD⟧x⟦/BOLD⟧ -> **x**, italic -> *x*
def to_runs(text: str) -> list[Run]            # Run = (text, bold, italic) sau khi gỡ token
```

Bất biến:

- **Cân bằng**: mỗi token mở có đúng một token đóng cùng loại phía sau; cho phép lồng nhau.
- **Multiset khớp nguồn** là điều kiện để tôn trọng vị trí model đặt.
- Token **không bao giờ** xuất hiện trong `.md` đích (đã chuyển thành `**`/`*`) hay trên trang PDF (đã
  gỡ). Chuỗi mang token chỉ tồn tại ở tầng trong: `block.text`, chunk text, prompt, checkpoint, TM.
- `strip(to_markdown(x)) == strip(x)` — chuyển đổi không làm mất chữ.

## 5. Ingest: từ glyph tới run

1. `_RawLine` thêm slot `spans: list[tuple[str, bool, bool]]` (chữ, đậm, nghiêng), dựng ngay tại
   `_classify` từ chính `chars`: dùng `_font_of` và `_font_style` (đã có) trên từng ký tự, rồi gộp các
   ký tự liền kề **cùng** cặp `(bold, italic)` thành một span. Chỉ gộp trong phạm vi một dòng.
2. `_page_lines`/`_classify` giữ `spans` cho dòng `body`, `bullet`, `numbered`, `heading`; dòng `table`
   và `code` để `spans` rỗng (ô bảng có cờ riêng; code không dịch).
3. `_build_units.flush()` ghép các dòng của một đoạn bằng một hàm **join theo run** mới
   (`_join_runs(parts: list[list[Run]]) -> list[Run]`), áp **đúng** luật của `_join_wrapped` hiện tại
   (nối bằng một dấu cách; gộp "previ-" + "ous" khi phần sau bắt đầu bằng chữ thường) nhưng ở mức run,
   rồi **gộp hai run cùng kiểu nằm hai bên mối nối**. Nhờ vậy run đậm trải qua nhiều dòng nguồn thành
   **một** cặp token, không sinh cặp thừa ở mỗi mép dòng.4. Sau khi ghép, đoạn chỉ sinh token khi **thật sự trộn kiểu**: chuỗi run của nó chứa **từ hai cặp `(bold, italic)` khác nhau trở lên**. Đoạn chỉ có một cặp duy nhất (kể cả khi cặp đó là đậm) → text y như hôm nay, **không** token. Hệ quả tốt:
   heading vốn đậm toàn bộ không có token và vẫn in đúng như hiện nay; tài liệu đơn kiểu không bị xáo
   trộn chút nào.
5. Chỉ **đoạn `body`/`list`/`heading`** được phát token. `code`, `table`, `toc` giữ nguyên (mục lục là
   pymupdf dựng lại dấu chấm dẫn, không đụng).
6. `bind_page_geometry` **không đổi**: `font/bold/italic` của block vẫn lấy từ `owned[0]` và vẫn là
   kiểu dùng để in khi block không có run.

## 6. Prompt và tầng agent

1. `translator.py` (khối `Rules`): thêm một gạch đầu dòng — giữ nguyên các cặp
   `⟦BOLD⟧…⟦/BOLD⟧` / `⟦ITALIC⟧…⟦/ITALIC⟧` quanh **đúng cụm chữ được nhấn mạnh**, dịch phần chữ bên
   trong, **không dịch tên token**, không tự thêm cặp mới. Đặt cùng chỗ đang giải thích `⟦CODE_1⟧`.
2. `verifier.py`: mục "Structure" nêu thêm hai cặp token này, để phần "No additions" không báo chúng
   là nội dung thêm. **Không** đổi danh sách check, **không** thêm mục mới.
3. `hard_checks` **không đổi** (xem §2.1): lệch sentinel không kích hoạt gọi lại.
4. Critic **không đổi**.
5. `_mock_json_reply` khoá theo chuỗi con nên các sửa trên không đổi hành vi mock; test ghép cuối
   `tests/test_agents.py` vẫn phải xanh.

## 7. Dựng lại PDF: layout theo run

1. `plain_text` trả chuỗi **đã gỡ token** như hiện nay (dùng `emphasis.strip`), để mọi nhánh cũ giữ
   nguyên chữ.
2. `_Placement` thêm `runs: list[Run] | None = None`. `placements()` đặt `runs` khi **cả hai** điều
   kiện đúng: text nguồn của block có token, **và** bản dịch đã qua `emphasis.strip` còn ra run hợp lệ
   (`to_runs` trả ≥ 2 kiểu khác nhau). Ngược lại `runs = None` → hành vi hôm nay.
3. `_insert` rẽ nhánh: `runs is None` → giữ nguyên toàn bộ thân hàm hiện tại. `runs` khác `None` →
   gọi `_insert_runs`, thực hiện:
   - **ngắt dòng tham lam** theo bề rộng khung, dùng `pymupdf.Font(fontfile=<font của run>).text_length`
     cho từng run; không cắt giữa một từ; bề rộng khung là `item.box[2] - item.box[0]`;
   - **hạ cỡ chữ** từ `item.size` xuống tới `max(item.size * _MIN_SCALE, 4.0)` theo đúng bước 5%/0.25 pt
     như hiện nay, và "vừa" nghĩa là số dòng cần thiết × `lineheight` (1.15) không vượt chiều cao khung;
     block `heading` vẫn giữ luật canh giữa theo `source_box` như hiện nay;
   - **đặt từng run** bằng `page.insert_text` tại `(x, baseline)` với `fontname = _resource_name(font)`
     và `fontfile = font` của run đó; `x` cho run đầu là mép trái khung (cộng phần canh giữa nếu áp
     dụng), các run sau nối tiếp theo bề rộng đo được;
   - mặt chữ của run lấy từ `match_font(<font của block>, bold, italic)`; `PDF_FONT_PATH`/`font_path`
     ghim vẫn thắng (khi ghim, mọi run dùng mặt đã ghim — đúng như `placements` đang làm với
     `match_style=False`).
   - Trả `"block"` khi đã in hết, `None` khi ngay ở sàn vẫn không vừa (đếm vào `failures` như hiện nay).
4. **Không** có nhánh tràn xuống lề dưới cho block có run: layout tự viết đã biết chiều cao của chính
   nó, nên "vừa" là điều kiện thật; giữ nguyên nhánh tràn cho block thường để không đổi hành vi cũ.
5. Phần vá ToUnicode (§2 của lượt trước) **đã** phủ trường hợp này: nó sửa map của **mọi** font mà ta
   chèn trên trang, kể cả nhiều mặt đậm/nghiêng khác nhau. Không cần sửa gì thêm.
6. `_insert_cell`, lưới bảng, `_toc_line`, `_marked_text` **không đổi**. Dấu đầu dòng của mục danh sách
   vẫn in bằng mặt chữ của block; các run của item nối tiếp sau nó.

## 8. Đường `.md` đích

`assemble()`/`assemble_partial()` chuyển token thành Markdown thật trước khi ghi:
`emphasis.to_markdown` biến `⟦BOLD⟧x⟦/BOLD⟧` → `**x**` và `⟦ITALIC⟧x⟦/ITALIC⟧` → `*x*`. Làm ở cuối
hai hàm, **một chỗ duy nhất** cho cả ghi đầy đủ và ghi từng phần, để người đọc `.md` nhận được nhấn
mạnh chuẩn Markdown thay vì token. PDF **không** dùng đường này (nó đi qua `to_runs`).

## 9. Sai lệch, biên và lỗi

| Trường hợp | Hành vi |
|---|---|
| Sentinels cân bằng, đếm khớp nguồn | Tôn trọng vị trí model đặt; in run theo đó |
| Sentinels lệch (thiếu/thừa/không cân bằng) | Gỡ **toàn bộ** token của block đó, in thường, `record_error(step="collect", cause="emphasis", …)` một dòng |
| Model dịch tên token (ví dụ `⟦ĐẬM⟧`) | Token lạ không khớp ngữ pháp → `counts` lệch → rơi vào hàng trên; không có `⟦` nào in ra trang |
| Model bỏ hẳn token, phần chữ còn nguyên | Hàng trên; **không mất chữ**, chỉ mất nhấn mạnh |
| Đoạn nguồn không trộn kiểu | Không token, `runs = None`, hành vi y hôm nay |
| Token trong **ô bảng** (chỉ có thể do model tự thêm, vì ingest không phát token cho bảng) | `_table_placements` gỡ token trước khi tách ô — v1 chưa in run trong ô; chữ trong ô không đổi, chỉ mất nhấn mạnh |
| Token trong block `code` / cả dòng `⟦CODE_n⟧` | Không xảy ra: hai loại này không được phát token (§5.5) và không dịch |
| Không tìm được mặt đậm/nghiêng trên máy | Run đó in bằng mặt mặc định → mất nhấn mạnh, **không** mất chữ; ghi `logger.debug` |
| Đoạn chỉ có một từ đậm, không có chữ thường | Chuỗi run có hai kiểu → có token; layout chạy bình thường |
| Nhấn mạnh vắt qua mối nối dòng nguồn | Run được gộp ở §5.3 → một cặp token, không sinh cặp thừa |
| Block không dịch (chunk hết lượt thử lại) | Nhánh `kept` không in và không gỡ gì; token chỉ tồn tại ở tầng chuỗi, còn trang gốc là chữ thật nên không có token nào trên trang |
| Tài liệu vào từ `.md`/`.docx` | Không có `meta["pdf"]` → không phát token ở ingest, không dựng lại; không đổi |

## 10. Kiểm thử

**`tests/test_emphasis.py`** (mới, thuần chuỗi, không cần PyMuPDF):
- `test_counts_and_balance` — đếm đúng, phát hiện cặp lệch và lồng nhau.
- `test_strip_keeps_every_character` — `strip` chỉ gỡ token, không đụng chữ có dấu tiếng Việt.
- `test_to_markdown_round_trips` — `strip(to_markdown(x)) == strip(x)`.
- `test_to_runs_marks_only_the_emphasised_span` — "A **B** C" → ba run, chỉ run giữa đậm.

**Ingest** (`tests/test_pdf_layout.py`):
- `test_a_mixed_style_line_marks_only_the_bold_span` — dựng một trang tổng hợp có một dòng với một cụm
  đậm giữa câu → `block.text` có đúng một cặp `⟦BOLD⟧…⟦/BOLD⟧` quanh cụm đó.
- `test_a_uniform_style_line_gets_no_tokens` — dòng một kiểu → `block.text` **không** chứa `⟦`.
- `test_a_bold_span_wrapped_over_two_lines_becomes_one_pair` — cụm đậm vắt qua hai dòng nguồn → đúng
  một cặp, không có cặp rỗng ở mối nối.
- `test_geometry_and_block_font_are_unchanged` — `meta["pdf"]` của block vẫn như trước khi có token.

**Dựng lại** (`tests/test_pdf_output.py`):
- `test_a_bold_span_is_printed_bold_and_the_rest_plain` — PDF tổng hợp có cụm đậm giữa câu, dựng lại,
  đọc `get_text("dict")` → span của cụm mang cờ đậm, phần còn lại không; hư không lộ `⟦` trên trang.
- `test_runs_that_do_not_fit_shrink_to_the_floor` — khung hẹp + bản dịch dài → cỡ chữ tụt nhưng không
  xuống dưới `_MIN_SCALE`, chữ không tràn khỏi khung.
- `test_a_block_without_runs_still_uses_insert_textbox` — block một kiểu đi đúng nhánh cũ (so khớp
  hành vi với test hồi quy hiện có).
- `test_lopsided_tokens_print_plain_and_record_a_failure` — bản dịch thiếu một token đóng → chữ in ra
  đầy đủ, không có `⟦`, và lỗi được ghi.
- `test_tokens_inside_a_table_cell_are_stripped` — bảng có token trong ô → ô vẫn in bản dịch, không có
  `⟦` nào trên trang.

**Ghép tầng** (`tests/test_agents.py`, `tests/test_pipeline.py`):
- `test_the_translator_prompt_describes_the_emphasis_tokens` — prompt có tên hai cặp token.
- `test_the_markdown_output_has_markdown_emphasis` — `assemble` (mock) cho `⟦BOLD⟧x⟦/BOLD⟧` → `.md` chứa
  `**x**`, không chứa `⟦`.

Cuối cùng: toàn bộ `pytest -q` và `ruff check` + `ruff format --check` phải sạch (mốc hiện tại:
492 passed, ruff sạch sau phần 1–2).

## 11. Tiêu chí nghiệm thu

1. Trên một PDF tổng hợp có cụm đậm giữa câu: PDF dựng lại có **đúng cụm đó** mang cờ đậm khi đọc
   `get_text("dict")`, phần còn lại thường.
2. Không ký tự nào của token (`⟦`, `BOLD`, `/BOLD`) xuất hiện trong `get_text()` của trang đã dựng lại.
3. Bỏ token ở cả hai đường (`.md` và PDF) cho ra **cùng một chuỗi chữ**; không mất chữ ở bất kỳ nhánh
   lệch nào.
4. Block **không** nhấn mạnh giữ nguyên hành vi: các test hồi quy PDF hiện có vẫn xanh mà không phải
   sửa.
5. Tài liệu đơn kiểu (mọi dòng một kiểu) không sinh token nào trong chunk text; số request LLM không
   đổi.
6. Suite và ruff sạch.

## 12. Rủi ro đã biết

- **Chất lượng ngắt dòng của ta thấp hơn `insert_textbox`.** Chỉ block có run mới chịu, và bước hạ cỡ
  chữ vẫn bảo đảm không tràn. Nếu tệ trên tài liệu thật, cách chữa là gọi `insert_textbox` cho phần
  chữ thường và chèn run đậm **lên trên** — phức tạp hơn, không làm ở v1.
- **Token làm model bối rối** (dịch tên token, đặt lệch). Đã có lưới an toàn: gỡ toàn bộ và in thường;
  tỷ lệ này **đo được** (đếm số lần ghi `cause="emphasis"`) nên có dữ liệu để chọn cách chữa, không
  đoán.
- **`.md` đích và PDF khác nhau đôi chút**: `.md` có `**`, PDF có cờ đậm thật. Cùng nội dung chữ; chỉ
  cách biểu diễn nhấn mạnh khác.
- **Chuỗi trong kho (TM, checkpoint, reasoning bank)** nay mang token. TM lưu cặp nguồn→đích nên vẫn
  khớp; đây là thay đổi định dạng **bên trong** kho, chấp nhận được nhưng phải nói ra.
- **Độ dài prompt tăng** ~10 ký tự mỗi cụm nhấn mạnh. Không đo được đáng kể.
- **Font đậm/nghiêng cục bộ có thể thiếu** → mất nhấn mạnh, không mất chữ (đã ghi ở §9).

## 13. Việc để sau (không thuộc bản này)

- Run **bên trong ô bảng** (v1 chỉ một mặt chữ mỗi ô).
- Gạch chân, gạch ngang, chỉ số trên/dưới, màu — chưa vào dòng chảy.
- Canh chỉnh run theo toạ độ nguồn (hiện chỉ kiểu chữ đi theo, không đi theo vị trí).
- Dùng lại token nhấn mạnh trong `consistency`/SkillOpt như một tín hiệu chất lượng.
