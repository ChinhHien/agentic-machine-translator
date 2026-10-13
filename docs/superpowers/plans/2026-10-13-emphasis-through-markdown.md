# Đậm/nghiêng đi theo Markdown — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cụm đậm/nghiêng nằm giữa một câu của bản gốc được in đậm/nghiêng trên trang PDF dựng lại, không mất chữ nào.

**Architecture:** Ingest phát **sentinel** `⟦BOLD⟧…⟦/BOLD⟧` / `⟦ITALIC⟧…⟦/ITALIC⟧` quanh đúng span đậm/nghiêng, ở tầng đã ghép các dòng nguồn của một đoạn. Sentinel đi cùng chuỗi qua chunk → prompt → bản dịch → checkpoint như `⟦CODE_n⟧` vẫn đi. Hai biên đầu ra tự xử: đường `.md` đổi sentinel thành `**`/`*`; đường PDF gỡ sentinel, và với block có nhấn mạnh thì **tự ngắt dòng theo run** rồi in từng run bằng mặt chữ đậm/nghiêng tương ứng. Block không nhấn mạnh giữ nguyên `insert_textbox`. Không phát sinh request LLM nào.

**Tech Stack:** Python 3.11, pdfplumber (đọc), PyMuPDF (dựng), pydantic (Document/Block), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-13-emphasis-through-markdown-design.md`

## Global Constraints

- Không thêm dependency nào; không phát sinh request LLM nào; không đổi chunker, checkpoint, web UI.
- Chuỗi token cố định, đúng từng ký tự: `⟦BOLD⟧`, `⟦/BOLD⟧`, `⟦ITALIC⟧`, `⟦/ITALIC⟧`.
- `src/polyglot_lc/emphasis.py` **không import gì** trong dự án (để `ingest/` dùng được mà không tạo phụ thuộc vòng).
- Tám khoá của `meta["pdf"]` giữ nguyên nghĩa và **không thêm khoá mới**.
- Sàn cỡ chữ `_MIN_SCALE = 0.55`, bước hạ `max(size*0.05, 0.25)`, `lineheight=1.15` — không đổi.
- Block **không** có run vẫn in bằng `insert_textbox`; mỗi ô bảng vẫn một mặt chữ (v1).
- `placements()` giữ nguyên chữ ký và số giá trị trả về `(items, kept)`; không thêm field vào `RebuildStats`.
- Lỗi sentinel được ghi **một lần**, ở tầng pipeline (`step="collect"`, `cause="emphasis"`); guard ở tầng dựng là phòng ngừa, **im lặng**.
- `cause="emphasis"` phải được thêm vào tập **đóng** `CAUSES` (`src/polyglot_lc/ui/trace.py:59`) — không thêm thì `record_error` ghi nó thành `unknown`. Kéo theo `tests/test_trace.py:50` đổi `len(CAUSES) == 11` thành `== 12`; đây là thay đổi test **do hành vi được yêu cầu** (tập đóng có thêm một phần tử thật), không phải để test xanh.
- `emphasis.strip` chỉ gỡ **bốn** token của mình: `⟦CODE_n⟧` không bị đụng, nên `test_a_cell_holding_a_code_placeholder_prints_it_as_text` phải tiếp tục xanh.
- Test không đọc `input/`, không cần mạng, không cần API key; test cần font Unicode dùng fixture `unicode_font` sẵn có.
- Mốc hiện tại phải giữ: `python -m pytest -q` → 492 passed; `ruff check src tests webui` và `ruff format --check src tests webui` sạch.
- Mỗi commit kết thúc bằng footer:
  `🤖 Generated with Codebuff` / `Co-Authored-By: Codebuff <noreply@codebuff.com>`

## Review Focus

Năm trường hợp spec ngụ ý nhưng chưa thành test riêng — mỗi dòng có test ở task sở hữu:

1. Model làm mất **một** token của một cặp → block đó phải in thường, **không** mất chữ, **không** lộ `⟦` (**Task 1**, **Task 4**, **Task 6**).
2. Máy không có mặt đậm/nghiêng cục bộ → mất nhấn mạnh nhưng chữ còn nguyên, không raise (**`test_a_missing_local_bold_face_loses_the_emphasis_but_not_the_text`**, Task 5).
3. Trang có hai mặt chữ mà dùng **cùng** một resource name → mặt thứ hai in bằng mặt thứ nhất, nhấn mạnh biến mất âm thầm (**`test_the_bold_run_uses_a_second_face_not_the_plain_one`**, Task 5).
4. Block có run mà ngay ở sàn vẫn không vừa khung → `failures` tăng, **không** chữ nào in ra ngoài khung (**Task 5**).
5. Bản dịch của bảng có token nằm **trong ô** → ô vẫn in bản dịch, không lộ token (**Task 4**).

---

### Task 1: Module ngữ pháp sentinel

**Files:**
- Create: `src/polyglot_lc/emphasis.py`
- Test: `tests/test_emphasis.py`

**Interfaces:**
- Consumes: không (task đầu).
- Produces:
  - `BOLD_OPEN: str = "⟦BOLD⟧"`, `BOLD_CLOSE: str = "⟦/BOLD⟧"`, `ITALIC_OPEN: str = "⟦ITALIC⟧"`, `ITALIC_CLOSE: str = "⟦/ITALIC⟧"`, `TOKENS: tuple[str, ...]` (bốn token).
  - `Run = tuple[str, bool, bool]` — `(text, bold, italic)`.
  - `counts(text: str) -> dict[str, int]` — mỗi token một khoá, kể cả khi bằng 0.
  - `is_balanced(text: str) -> bool`
  - `strip(text: str) -> str`
  - `to_markdown(text: str) -> str`
  - `to_runs(text: str) -> list[Run]`
  - `agrees_with(source: str, translation: str) -> bool` — `is_balanced(translation)` và mọi khoá của `counts(translation)` bằng `counts(source)`.

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_emphasis.py
from polyglot_lc import emphasis

def test_counts_counts_each_token():
    text = "⟦BOLD⟧a⟦/BOLD⟧ ⟦ITALIC⟧b⟦/ITALIC⟧ ⟦BOLD⟧c⟦/BOLD⟧"
    assert emphasis.counts(text)["⟦BOLD⟧"] == 2
    assert emphasis.counts(text)["⟦/ITALIC⟧"] == 1
    assert emphasis.counts("plain")["⟦ITALIC⟧"] == 0

def test_is_balanced_rejects_a_lone_token_and_accepts_nesting():
    assert emphasis.is_balanced("⟦BOLD⟧a⟦/BOLD⟧") is True
    assert emphasis.is_balanced("⟦BOLD⟧a⟦ITALIC⟧b⟦/ITALIC⟧c⟦/BOLD⟧") is True
    assert emphasis.is_balanced("⟦BOLD⟧a") is False
    assert emphasis.is_balanced("⟦BOLD⟧a⟦/ITALIC⟧") is False

def test_strip_keeps_every_character():
    text = "⟦BOLD⟧Eocortex ST⟦/BOLD⟧ được thiết kế để"
    assert emphasis.strip(text) == "Eocortex ST được thiết kế để"

def test_to_markdown_makes_real_markdown_and_never_leaves_a_token():
    assert emphasis.to_markdown("a ⟦BOLD⟧b⟦/BOLD⟧ c") == "a **b** c"
    assert emphasis.to_markdown("a ⟦ITALIC⟧b⟦/ITALIC⟧ c") == "a *b* c"
    # Nested: the inner pair must survive the outer substitution.
    assert emphasis.to_markdown("⟦BOLD⟧a⟦ITALIC⟧b⟦/ITALIC⟧c⟦/BOLD⟧") == "**a*b*c**"
    # A lone token carries no emphasis to convert: it is dropped, not printed.
    assert "⟦" not in emphasis.to_markdown("a ⟦BOLD⟧b")

def test_to_runs_marks_only_the_emphasised_span():
    runs = emphasis.to_runs("A ⟦BOLD⟧B⟦/BOLD⟧ C")
    assert runs == [("A ", False, False), ("B", True, False), (" C", False, False)]

def test_to_runs_never_leaves_a_token():
    assert "⟦" not in "".join(text for text, _, _ in emphasis.to_runs("⟦BOLD⟧a⟦/ITALIC⟧"))

def test_agrees_with_requires_equal_counts():
    assert emphasis.agrees_with("⟦BOLD⟧a⟦/BOLD⟧", "⟦BOLD⟧a⟦/BOLD⟧") is True
    assert emphasis.agrees_with("plain", "⟦BOLD⟧a⟦/BOLD⟧") is False
    assert emphasis.agrees_with("⟦BOLD⟧a⟦/BOLD⟧", "a ⟦BOLD⟧b") is False
```

- [ ] **Step 2: Chạy test để chắc nó thất bại**

Run: `python -m pytest tests/test_emphasis.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot_lc.emphasis'`

- [ ] **Step 3: Viết `src/polyglot_lc/emphasis.py`**

Định nghĩa hằng số và `Run` như Interfaces. Thuật toán:

- `_PAIR_RE = re.compile(r"⟦(BOLD|ITALIC)⟧(.*?)⟦/\1⟧", re.DOTALL)` — backreference buộc cặp **cùng loại**.
- `is_balanced`: lặp `text = _PAIR_RE.sub("", text)` tới khi không đổi, rồi `không còn token nào trong TOKENS`. Cách này tự nhiên loại cặp lệch loại và cặp không lồng đúng.
- `to_markdown`: lặp `_PAIR_RE.sub` với hàm thay `⟦BOLD⟧x⟦/BOLD⟧ -> **x**` (và `*x*`) cho tới khi không còn cặp nào, rồi `strip` phần token còn sót.
- `to_runs`: quét một lượt, giữ một **stack** trạng thái `(bold, italic)`; token mở đẩy trạng thái mới, token đóng lấy phần tử cuối (stack rỗng thì bỏ qua), phần chữ giữa hai token thành một run mang trạng thái hiện tại; **gộp** hai run liền kề cùng trạng thái và bỏ run rỗng. Token không bao giờ lọt vào `text` của run, nên đầu ra không thể chứa `⟦`.
- `agrees_with`: `is_balanced(translation) and counts(translation) == counts(source)`.

- [ ] **Step 4: Chạy test để chắc nó qua**

Run: `python -m pytest tests/test_emphasis.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/emphasis.py tests/test_emphasis.py
git commit -m "feat(emphasis): add the sentinel grammar for bold and italic spans"
```

---

### Task 2: Ingest phát sentinel quanh span đậm/nghiêng

**Files:**
- Modify: `src/polyglot_lc/ingest/pdf_layout.py`
- Test: `tests/test_pdf_layout.py`

**Interfaces:**
- Consumes: `emphasis.strip`, `emphasis.BOLD_OPEN`, `emphasis.BOLD_CLOSE`, `emphasis.ITALIC_OPEN`, `emphasis.ITALIC_CLOSE` (Task 1).
- Produces:
  - `_RawLine.spans: list[emphasis.Run]` — run theo **một** dòng, rỗng với dòng `table`/`code`.
  - `_join_runs(parts: list[list[emphasis.Run]]) -> list[emphasis.Run]` — ghép run của các dòng nguồn trong một đoạn.
  - `_mark_emphasis(runs: list[emphasis.Run]) -> str` — chuỗi đã bọc token, chỉ khi có từ hai trạng thái trở lên.
  - `_Unit.text` của đoạn trộn kiểu **có** token; đoạn một kiểu **không** đổi gì.

- [ ] **Step 1: Thêm helper và test thất bại**

Thêm vào `tests/test_pdf_layout.py` (giữ nguyên `_line`, `_chars`, BODY/MONO/BOLD hiện có):

```python
#: Một mặt chữ mà `_font_style` đọc ra là đậm/nghiêng (nó chỉ tìm chuỗi con).
BOLD_FACE = "STUVWX+Verdana-Bold"
ITALIC_FACE = "YZABCD+Verdana-Italic"

def _styled_line(
    parts: list[tuple[list[str], str]], top: float, x0: float = 100.0, size: float = 12.0
) -> dict:
    """Một dòng nguồn có nhiều kiểu: mỗi phần là (các từ, mặt chữ)."""
    chars: list[dict] = []
    x = x0
    for tokens, font in parts:
        piece = _chars(tokens, x, size, font)
        chars.extend(piece)
        x = max(ch["x1"] for ch in piece) + 3.0  # khoảng cách từ, giữa hai kiểu
    return {"chars": chars, "top": top}
```

```python
def test_a_mixed_style_line_marks_only_the_bold_span():
    page = _StubPage(
        [_styled_line([(["Alpha"], BODY), (["Beta"], BOLD_FACE), (["Gamma"], BODY)], top=150.0)],
        height=595.0,
    )
    units = _build_units([page])
    assert units[0].text == "Alpha ⟦BOLD⟧Beta⟦/BOLD⟧ Gamma"

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
    assert geo["font"] == BODY and geo["bold"] is False
    assert set(geo) == {"rects", "size", "kind", "font", "bold", "italic", "color", "markers"}
```

- [ ] **Step 2: Chạy test để chắc nó thất bại**

Run: `python -m pytest tests/test_pdf_layout.py -q -k "style or tokens or wrapped or geometry"`
Expected: FAIL — `AttributeError: 'dict' object has no attribute 'text'`… với `_styled_line` thì lỗi thật là `assert 'Alpha Beta Gamma' == 'Alpha ⟦BOLD⟧Beta⟦/BOLD⟧ Gamma'`.

- [ ] **Step 3: Sửa `pdf_layout.py`**

1. `_RawLine.__slots__`/`__init__` thêm `spans: list[tuple[str, bool, bool]]` (mặc định `[]`).
2. Hàm mới `_spans(chars) -> list[tuple[str, bool, bool]]`: gom **các ký tự liền kề cùng cặp `(bold, italic)`** thành một span, dùng `_font_style(_font_of(ch))`; dùng `line_text` để dựng chữ của span (giữ nguyên luật chèn dấu cách theo khoảng glyph). Chỉ gọi cho dòng `body`/`bullet`/`numbered`/`heading`; dòng `table`/`code` để rỗng.
3. `_classify` truyền `spans=_spans(chars)` vào `_RawLine`.
4. `_join_runs(parts)` — bản sao luật của `_join_wrapped` ở mức run: nối bằng một dấu cách; nếu chữ tích luỹ kết thúc bằng `-` và run sau bắt đầu bằng chữ thường thì **bỏ** gạch nối (nối từ gãy dòng); sau khi nối, **gộp hai run cùng trạng thái nằm hai bên mối nối** (đây là chỗ biến hai cặp token thành một).
5. `_mark_emphasis(runs)`: nếu số trạng thái **khác nhau** trong `runs` < 2 thì trả `"".join(text)`; ngược lại bọc mỗi run bằng cặp token của trạng thái nó (chỉ mở/đóng khi trạng thái **đổi** giữa hai run liền kề).
6. `flush()` dùng `_join_runs([line.spans for line, _, _ in pending])` thay cho `_join_wrapped([rendered …])`, rồi `_mark_emphasis`.
7. **Mục danh sách**: hiện `pending.append((ln, "- " + item, marker))` đã ghép dấu đầu dòng vào chuỗi `rendered`. Với spans phải làm tương đương **trên run đầu tiên**: gọi `_bullet_text` (hoặc `_number_marker` đối chiếu số) lên chữ của run đầu để **gỡ** glyph bullet mà `ln.text` vẫn còn, rồi ghép `"- "` vào trước run đầu **sau khi** đã gỡ. Bỏ bước gỡ này thì bullet bị in hai lần (một lần từ chữ nguồn, một lần từ `pending`). Test hiện có về danh sách trong `tests/test_pdf_layout.py` là chỗ bắt lỗi này.

- [ ] **Step 4: Chạy test để chắc nó qua**

Run: `python -m pytest tests/test_pdf_layout.py -q`
Expected: PASS, **kể cả toàn bộ test cũ** (tài liệu một kiểu không sinh token nên chuỗi Markdown không đổi).

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/ingest/pdf_layout.py tests/test_pdf_layout.py
git commit -m "feat(ingest): mark bold and italic spans with sentinels"
```

---

### Task 3: Prompt của Translator và Verifier

**Files:**
- Modify: `src/polyglot_lc/agents/translator.py:63-79` (khối `Rules`), `src/polyglot_lc/agents/verifier.py:82`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: tên token của Task 1.
- Produces: không có API mới — chỉ chuỗi prompt.

- [ ] **Step 1: Viết test thất bại**

```python
def test_the_translator_prompt_describes_the_emphasis_tokens(mock_env):
    from polyglot_lc.config import get_agent_config

    t = TranslatorAgent(get_agent_config("TRANSLATOR"))
    joined = "\n".join(m["content"] for m in t.build_prompt("A ⟦BOLD⟧B⟦/BOLD⟧ C.", "en", "vi", "literal"))
    assert "⟦BOLD⟧" in joined and "⟦/BOLD⟧" in joined
    assert "never translate or rename the tokens" in joined


def test_the_verifier_prompt_names_the_emphasis_tokens_as_structure(mock_env):
    from polyglot_lc.config import get_agent_config

    v = VerifierAgent(get_agent_config("VERIFIER"))
    prompt = _capture_verifier_prompt(v)
    assert "⟦BOLD⟧" in prompt
```

- [ ] **Step 2: Chạy test để chắc nó thất bại**

Run: `python -m pytest tests/test_agents.py -q -k emphasis`
Expected: FAIL trên cả hai assertion.

- [ ] **Step 3: Sửa hai prompt**

`translator.py`, trong khối `Rules` (ngay sau gạch đầu dòng nói về `⟦CODE_1⟧`), thêm **một** gạch đầu dòng:

```
"- Keep the emphasis pairs ⟦BOLD⟧...⟦/BOLD⟧ and ⟦ITALIC⟧...⟦/ITALIC⟧ around the "
"emphasised words; translate the text inside them, never translate or rename "
"the tokens, and never add a pair of your own.\n"
```

`verifier.py:82`, mục `4. Structure`, thêm vào danh sách cấu trúc bắt buộc còn nguyên:

```
", the emphasis pairs (⟦BOLD⟧...⟦/BOLD⟧, ⟦ITALIC⟧...⟦/ITALIC⟧) "
```

Không sửa danh sách check, không thêm mục mới, không đụng `hard_checks` (spec §2.1).

- [ ] **Step 4: Chạy test để chắc nó qua**

Run: `python -m pytest tests/test_agents.py -q`
Expected: PASS, kể cả `test_mock_dispatcher_still_recognises_the_live_prompts` (mock khoá theo chuỗi con `"QA verifier"`, không đổi).

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/agents/translator.py src/polyglot_lc/agents/verifier.py tests/test_agents.py
git commit -m "feat(agents): tell the translator and verifier about the emphasis tokens"
```

---

### Task 4: Dựng lại — gỡ token và quyết định block nào có run

**Files:**
- Modify: `src/polyglot_lc/pipeline/pdf_output.py` (`plain_text:352`, `_Placement:~355`, `placements:633`, `_table_placements:~500`)
- Test: `tests/test_pdf_output.py`

**Interfaces:**
- Consumes: `emphasis.strip`, `emphasis.agrees_with`, `emphasis.to_runs`.
- Produces:
  - `plain_text(markdown_text: str, kind: str) -> str` — nay **luôn** gỡ token (chữ ký không đổi).
  - `_Placement.runs: list[tuple[str, bool, bool]] | None` — `None` = đường `insert_textbox` như cũ.
  - `_Placement.face: str = ""` — **tên họ mặt chữ** của block (`geo["font"]`), không phải đường dẫn tệp font: Task 5 cần nó để `match_font(face, bold, italic)` tìm mặt đậm/nghiêng cùng họ.
  - `placements(...)` gán `runs` và `face`; **không** đổi chữ ký hay số giá trị trả về.

- [ ] **Step 1: Viết test thất bại**

```python
def test_plain_text_drops_emphasis_tokens():
    assert plain_text("a ⟦BOLD⟧b⟦/BOLD⟧ c", "body") == "a b c"
    assert plain_text("⟦BOLD⟧a⟦/BOLD⟧b", "heading") == "ab"


def _emphasis_doc(source_text: str) -> Document:
    return Document(
        blocks=[
            Block(
                kind="paragraph",
                text=source_text,
                meta={"pdf": {"rects": [(0, 72.0, 100.0, 300.0, 112.0)], "size": 10.0, "kind": "body"}},
            )
        ]
    )


def test_a_block_with_balanced_tokens_gets_runs():
    doc = _emphasis_doc("A ⟦BOLD⟧B⟦/BOLD⟧ C")
    items, _ = placements(doc, {0: "A ⟦BOLD⟧B⟦/BOLD⟧ C"}, match_style=False)
    assert items[0].runs == [("A ", False, False), ("B", True, False), (" C", False, False)]


def test_a_block_whose_translation_lost_a_token_gets_no_runs():
    """Lệch token thì in thường: guard ở đây là phòng ngừa, và im lặng (spec §2.1)."""
    doc = _emphasis_doc("A ⟦BOLD⟧B⟦/BOLD⟧ C")
    items, _ = placements(doc, {0: "A B ⟦BOLD⟧C"}, match_style=False)
    assert items[0].runs is None
    assert "⟦" not in items[0].text


def test_a_block_whose_source_had_no_tokens_gets_no_runs():
    """Model tự thêm cặp token: nguồn không nhấn mạnh thì không được in nhấn mạnh."""
    doc = _emphasis_doc("A B C")
    items, _ = placements(doc, {0: "A ⟦BOLD⟧B⟦/BOLD⟧ C"}, match_style=False)
    assert items[0].runs is None


def test_tokens_inside_a_cell_are_stripped_before_the_cell_is_split():
    doc = _table_doc()  # helper sẵn có: block bảng 2x2 có `cells`
    table = "| ⟦BOLD⟧a⟦/BOLD⟧ | b |\n| --- | --- |\n| c | d |"
    items, _ = placements(doc, {0: table}, match_style=False)
    assert [item.text for item in items] == ["a", "b", "c", "d"]
    assert all("⟦" not in item.text for item in items)
```

- [ ] **Step 2: Chạy test để chắc nó thất bại**

Run: `python -m pytest tests/test_pdf_output.py -q -k "plain_text or runs or cell"`
Expected: FAIL — `AttributeError: 'Placement' object has no attribute 'runs'` và token còn trong text.

- [ ] **Step 3: Sửa `pdf_output.py`**

1. `plain_text`: bọc kết quả trả về bằng `emphasis.strip(...)` (giữ nguyên phần gỡ dấu heading/list).
2. `_Placement` thêm hai trường: `runs: list[tuple[str, bool, bool]] | None = None` và `face: str = ""`. Trong `placements`, nhánh block thường gán `item.face = str(geo.get("font") or "")` (cùng chỗ đang gán `item.font_file`). `_table_placements` **không** cần `face`: v1 không in run trong ô.
3. Trong `placements()`, tại nhánh block thường, sau khi đã có `raw` và `text`:

```python
if _has_tokens(block.text) and emphasis.agrees_with(block.text, raw) and len(set(
    (b, i) for _, b, i in emphasis.to_runs(text)
)) >= 2:
    item.runs = emphasis.to_runs(text)
```

(`_has_tokens(t)` là `any(tok in t for tok in emphasis.TOKENS)` — định nghĩa đúng một chỗ trong `pdf_output.py`; **không** đặt điều kiện này trong `emphasis.py` vì nó là luật của tầng dựng, không phải của ngữ pháp.)

4. Trong `_table_placements`, gỡ token khỏi chữ của **từng ô** ngay sau khi tách ma trận: `row[column] = emphasis.strip(row[column])` trước khi tạo `_Placement`. `_markdown_cells` **không** đổi (token không chứa `|`).
5. `_marked_text` và `_toc_line` **không** đổi: chúng chạy trên `item.text` đã gỡ token.

- [ ] **Step 4: Chạy test để chắc nó qua**

Run: `python -m pytest tests/test_pdf_output.py -q`
Expected: PASS, kể cả toàn bộ test cũ (mọi `assert item.runs is None` gián tiếp đều đúng vì tài liệu cũ không có token).

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/pipeline/pdf_output.py tests/test_pdf_output.py
git commit -m "feat(pdf): decide which blocks carry emphasis runs"
```

---

### Task 5: Dựng lại — in từng run bằng mặt chữ của nó

**Files:**
- Modify: `src/polyglot_lc/pipeline/pdf_output.py` (`_insert:814`, hàm mới `_insert_runs`)
- Modify: `tests/pdfdoc.py` (helper mới)
- Test: `tests/test_pdf_output.py`

**Interfaces:**
- Consumes: `_Placement.runs` (Task 4); `match_font(name, bold, italic)` (đã có); `_resource_name(font_file)` (đã có); `_align_of` (đã có).
- Produces: `_insert_runs(page, item, pymupdf, default_font, source_font: str) -> str | None` — trả `"block"`/`None` như `_insert`.

- [ ] **Step 1: Thêm helper `tests/pdfdoc.py` và test thất bại**

```python
def write_styled_pdf(path, before: str, bold: str, after: str, *, size: float = 10.0) -> Path:
    """Một dòng thân bài có cụm giữa in đậm, rồi trở lại thường.

    Ba run vẽ trên **cùng một baseline** để pdfplumber gom chúng thành một dòng
    nguồn trộn kiểu. Bề rộng lấy từ `pymupdf.get_text_length` để các run khít nhau.
    """
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=_WIDTH, height=_HEIGHT)
    x = _LEFT
    for text, font in ((before, _BODY_FONT), (bold, "hebo"), (after, _BODY_FONT)):
        page.insert_text((x, _TOP), text, fontname=font, fontsize=size)
        x += pymupdf.get_text_length(text, fontname=font, fontsize=size)
    page.insert_text((_LEFT, _TOP + _ADVANCE), "A plain following line.", fontname=_BODY_FONT, fontsize=size)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    doc.close()
    return out
```

```python
def _spans(pdf_path) -> list[tuple[str, int, str]]:
    """``(text, flags, font)`` của mọi span trên trang đầu.

    Bit 16 là đậm và bit 2 là nghiêng. Cờ này sống sót khi nhúng bằng ``fontfile``
    **chỉ khi mỗi mặt có resource name riêng** — trùng tên thì PyMuPDF in bằng mặt
    đăng ký trước đó, đó là lý do `_resource_name` băm theo đường dẫn tệp font.
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

    Nếu chúng trùng tên, trang thật có hai mặt cùng tên sẽ âm thầm in bằng mặt
    thứ nhất và nhấn mạnh biến mất mà không có lỗi nào.
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


def test_a_long_translation_uses_the_room_below_before_giving_up(tmp_path, unicode_font):
    """`rebuild_pdf` xoá chữ nguồn **trước** khi in, nên một block không in được là
    mất sạch chữ chứ không để lại chữ nguồn. Đường run phải dùng hết chỗ tới lề dưới
    ở cỡ sàn trước khi chịu thua."""
    import pymupdf

    from pdfdoc import write_styled_pdf

    src = write_styled_pdf(tmp_path / "in.pdf", "Alpha ", "Beta", " Gamma")
    doc = load_document(src)
    long_text = "Alpha ⟦BOLD⟧" + "rất dài " * 12 + "⟦/BOLD⟧ Gamma"
    stats = rebuild_pdf(src, doc, {0: long_text}, tmp_path / "out.pdf", font_path=unicode_font)

    assert stats.failures == [], "chữ dài vẫn phải in được, chỉ là tràn xuống dưới"
    with pymupdf.open(tmp_path / "out.pdf") as rebuilt:
        words = rebuilt[0].get_text("words")
        height = rebuilt[0].rect.height
    assert words, "trang phải còn chữ"
    assert all(word[1] >= 0 and word[3] <= height for word in words), "chữ không được rời khỏi trang"
    assert not any("⟦" in word[4] for word in words)
```

- [ ] **Step 2: Chạy test để chắc nó thất bại**

Run: `python -m pytest tests/test_pdf_output.py -q -k "bold_span or second_face or missing_local or room_below"`
Expected: FAIL — không span nào mang bit 16 (cả block in bằng một mặt chữ thường).

- [ ] **Step 3: Thêm `_insert_runs` và rẽ nhánh trong `_insert`**

`_insert`, ngay sau khi xử lý `item.kind == "cell"`, thêm:

```python
if item.runs is not None:
    return _insert_runs(page, item, pymupdf, default_font)
```

`_insert_runs` — chữ ký `(page, item, pymupdf, default_font) -> str | None`, đọc `item.runs` và `item.face`:

1. Chọn mặt chữ cho từng cặp `(bold, italic)` xuất hiện trong `item.runs`: `match_font(item.face, bold, italic)`; trả `None` thì dùng `default_font`. Khi `font_path`/`PDF_FONT_PATH` được chỉ định, mọi run dùng mặt đã ghim (một mặt duy nhất, đúng như `placements` đang làm với `match_style=False`).
2. Ngắt dòng tham lam: với mỗi run, tách theo từ; đo bằng `pymupdf.Font(fontfile=...).text_length(piece, fontsize=size)`; xuống dòng khi bề rộng vượt `box[2] - box[0]`.
3. Vừa hay chưa: số dòng × `1.15 * size` ≤ `box[3] - box[1]`. Hạ `size` theo đúng bước `max(item.size * 0.05, 0.25)` tới sàn `max(item.size * _MIN_SCALE, 4.0)`.
4. In từng run bằng `page.insert_text((x, baseline), run_text, fontname=_resource_name(font), fontfile=font, fontsize=size, color=item.color)`; `x` cộng dồn theo bề rộng đo được; canh giữa (chỉ `kind == "heading"`, luật `source_box` hiện có) thì lùi `x` đầu dòng theo `(box_width - line_width) / 2`.
5. **Không vừa ở cỡ sàn** → dùng lại đúng nhánh cuối của `_insert`: in ở cỡ sàn vào khung mở rộng xuống `max(page.rect.height - _BOTTOM_MARGIN, box[3])`, rồi trả `"block"`. Lý do là bắt buộc: `rebuild_pdf` đã `apply_redactions` **trước** khi gọi `_insert`, nên trả `None` ở đây là **mất sạch chữ** của block đó (spec §7.4).
6. Trả `None` **chỉ khi** cả khung mở rộng cũng không in được — khi đó rơi vào `stats.failures`.
7. Ghi `logger.debug` khi một run phải dùng `default_font` vì không có mặt đậm/nghiêng cục bộ.

- [ ] **Step 4: Chạy test để chắc nó qua**

Run: `python -m pytest tests/test_pdf_output.py -q`
Expected: PASS. Nếu `test_a_bold_span_is_printed_bold_and_the_rest_plain` vẫn không thấy `flags & 16`, **đừng** hạ assertion xuống thành so tên font rồi bỏ qua: trước hết kiểm `_resource_name` có khác nhau giữa hai mặt không (probe đã xác nhận cơ chế này hoạt động), vì đó gần như luôn là nguyên nhân.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/pipeline/pdf_output.py tests/pdfdoc.py tests/test_pdf_output.py
git commit -m "feat(pdf): print each emphasis run in its own face"
```

---

### Task 6: Ghi lỗi khi sentinel lệch (tầng pipeline)

**Files:**
- Modify: `src/polyglot_lc/orchestrator/graph.py` (module-level `settle_emphasis`, dùng trong `collect_node`), `src/polyglot_lc/ui/trace.py:59` (tập `CAUSES`)
- Modify test: `tests/test_trace.py:50` (`== 11` → `== 12`)
- Test: `tests/test_concurrency_budget.py`

**Interfaces:**
- Consumes: `emphasis.agrees_with`, `emphasis.strip` (Task 1); `write_styled_pdf` (Task 5) để có nguồn thật mang token.
- Produces: `settle_emphasis(chunk_text: str, translation: str) -> tuple[str, bool]` — trả `(chữ đã xử lý, đã gỡ token hay chưa)`. Đặt ở **mức module** trong `orchestrator/graph.py` (không phải closure) chính vì để test được mà không cần dựng graph.

- [ ] **Step 1: Viết test thất bại**

Hai test đơn vị cho hàm thuần (đặt ở `tests/test_concurrency_budget.py`, cùng chỗ với các test graph khác):

```python
def test_settle_emphasis_drops_a_lopsided_pair():
    from polyglot_lc.orchestrator.graph import settle_emphasis

    text, dropped = settle_emphasis("A ⟦BOLD⟧B⟦/BOLD⟧ C", "A B ⟦BOLD⟧C")
    assert text == "A B C"
    assert dropped is True


def test_settle_emphasis_keeps_a_matching_pair():
    from polyglot_lc.orchestrator.graph import settle_emphasis

    text, dropped = settle_emphasis("A ⟦BOLD⟧B⟦/BOLD⟧ C", "A ⟦BOLD⟧B⟦/BOLD⟧ C")
    assert text == "A ⟦BOLD⟧B⟦/BOLD⟧ C"
    assert dropped is False
```

Và một test đi hết graph, theo đúng khuôn của `test_chunk_survives_rb_upsert_failure` (cùng file):

```python
def test_a_lopsided_emphasis_pair_is_recorded_as_an_emphasis_error(mock_env, tmp_path, monkeypatch):
    """Không tốn lượt dịch lại vì kiểu chữ (spec §2.1), nhưng phải thấy được trong trace."""
    from pdfdoc import write_styled_pdf

    import json

    from polyglot_lc.llm.client import LLMResponse
    from polyglot_lc.orchestrator.graph import make_chunk_graph
    from polyglot_lc.ui.events import EventSink

    path = write_styled_pdf(tmp_path / "styled.pdf", "Alpha ", "Beta", " Gamma")
    chunks, _ = _expected_chunks(path)
    assert "⟦BOLD⟧" in chunks[0].text, "nguồn thật phải mang token"
    sink = EventSink(tmp_path, enabled=True)
    ctx = _ctx(mock_env, _BoomRB())
    ctx.sink = sink
    # Bản dịch mock làm mất token đóng → lệch → phải bị gỡ và ghi lại.
    monkeypatch.setattr(
        ctx.translator, "chat", lambda *a, **k: LLMResponse(text="Alpha ⟦BOLD⟧Beta Gamma", model="m")
    )
    out = make_chunk_graph(ctx).invoke(
        {
            "chunk": chunks[0],
            "src_lang": "en",
            "tgt_lang": "vi",
            "domain": "general",
            "strategy": "literal",
            "few_shot": [],
            "pitfalls": [],
            "glossary_md": "",
        },
        config={"recursion_limit": 100},
    )

    assert "⟦" not in out["translated"]
    records = [json.loads(line) for line in sink.path.read_text(encoding="utf-8").splitlines()]
    assert any(r.get("cause") == "emphasis" for r in records), records
```

- [ ] **Step 2: Chạy test để chắc nó thất bại**

Run: `python -m pytest tests/test_concurrency_budget.py -q -k emphasis`
Expected: FAIL — `ImportError: cannot import name 'settle_emphasis'`.

- [ ] **Step 3: Thêm `settle_emphasis`, nối vào `collect_node`, và mở rộng `CAUSES`**

`orchestrator/graph.py`, ở mức module:

```python
def settle_emphasis(chunk_text: str, translation: str) -> tuple[str, bool]:
    """Drop the chunk's emphasis tokens unless they match the source exactly.

    The tokens mark spans, not content, so a mismatch is not worth a
    re-translation; but leaving a lopsided pair in would put a ⟦ on the page.
    """
    if emphasis.agrees_with(chunk_text, translation):
        return translation, False
    return emphasis.strip(translation), True
```

Trong `collect_node`, ngay **sau** khối `repair_code_placeholders` hiện có, cùng phong cách với nó:

```python
translated, dropped_emphasis = settle_emphasis(state["chunk"].text, translated)
if dropped_emphasis:
    logger.warning(
        "chunk={} emphasis tokens did not match the source; printing this chunk plain",
        state["chunk"].id,
    )
    ctx.record_error(
        step="collect",
        chunk_id=state["chunk"].id,
        cause="emphasis",
        message="emphasis tokens dropped: counts do not match the source",
    )
```

`ui/trace.py:59`: thêm `"emphasis"` vào `frozenset` `CAUSES`. **Bắt buộc**: `record_error` đổi mọi cause ngoài tập này thành `"unknown"` rồi mới ghi. `tests/test_trace.py:50` đổi `assert len(CAUSES) == 11` thành `== 12`, kèm một dòng nói rõ đây là tập đóng có thêm một phần tử thật.

Mức **chunk** là mức đúng ở đây: `collect_node` chỉ thấy chuỗi của cả chunk, không thấy ranh giới block. Guard theo block nằm ở Task 4 và im lặng.

- [ ] **Step 4: Chạy test để chắc nó qua**

Run: `python -m pytest tests/test_concurrency_budget.py tests/test_trace.py tests/test_pipeline.py -q`
Expected: PASS, kể cả `test_trace.py` với con số mới.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/orchestrator/graph.py src/polyglot_lc/ui/trace.py tests/test_concurrency_budget.py tests/test_trace.py
git commit -m "feat(pipeline): drop and record emphasis tokens that do not match the source"
```

---

### Task 7: Markdown đích mang nhấn mạnh chuẩn

**Files:**
- Modify: `src/polyglot_lc/pipeline/assembler.py` (`assemble:178`, `assemble_partial:259`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `emphasis.to_markdown` (Task 1).
- Produces: chuỗi trả về của `assemble`/`assemble_partial` không còn token, và có `**`/`*`.

- [ ] **Step 1: Viết test thất bại**

```python
def test_the_markdown_output_has_markdown_emphasis():
    from polyglot_lc.pipeline.assembler import assemble

    chunks = [_FakeChunk("c0", [0, 1])]           # helper sẵn có ở đầu tệp
    out = assemble(chunks, {"c0": "A ⟦BOLD⟧B⟦/BOLD⟧ C"}, {})
    assert "**B**" in out
    assert "⟦" not in out
```

- [ ] **Step 2: Chạy test để chắc nó thất bại**

Run: `python -m pytest tests/test_pipeline.py -q -k markdown_emphasis`
Expected: FAIL — `'⟦' in out`.

- [ ] **Step 3: Sửa `assembler.py`**

Bọc giá trị trả về của **cả** `assemble` và `assemble_partial` bằng `emphasis.to_markdown(...)`, ở cuối hàm, sau khi code đã được `restore_code` khôi phục — `to_markdown` chỉ đụng bốn token của mình nên không thể làm hỏng code đã khôi phục. Không đổi chữ ký.

- [ ] **Step 4: Chạy test để chắc nó qua**

Run: `python -m pytest tests/test_pipeline.py -q`
Expected: PASS, kể cả các test assembly hiện có (không có token thì `to_markdown` không đổi gì).

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/pipeline/assembler.py tests/test_pipeline.py
git commit -m "feat(pipeline): render emphasis tokens as Markdown in the .md output"
```

---

### Task 8: Chạy toàn bộ và rà lại tài liệu

**Files:**
- Modify: `src/polyglot_lc/pipeline/pdf_output.py` (docstring module), `src/polyglot_lc/ingest/pdf_layout.py` (docstring module)

**Interfaces:**
- Consumes: tất cả task trên.
- Produces: không code mới.

- [ ] **Step 1: Cập nhật docstring module**

`pdf_output.py`: thêm một đoạn ngắn nói rằng một block có nhấn mạnh được in bằng layout tự viết (ngắt dòng tham lam theo run, mỗi run một mặt chữ) và rằng token **không bao giờ** lộ lên trang, cạnh đoạn đã có về `⟦CODE_n⟧`.
`pdf_layout.py`: thêm vào danh sách đặc điểm rằng **span đậm/nghiêng trong một dòng** được đánh dấu bằng sentinel, và rằng đoạn một kiểu không sinh token nào.

- [ ] **Step 2: Chạy toàn bộ suite và ruff**

Run:
```bash
set -o pipefail
python -m pytest -q
ruff check src tests webui
ruff format --check src tests webui
```
Expected: `pytest` báo số test **lớn hơn 492** và không có `failed`; cả hai lệnh `ruff` in ra kết quả sạch. Ghi lại con số test mới vào phần báo cáo.

- [ ] **Step 3: Kiểm tay trên tài liệu thật (không cần LLM)**

Run:
```bash
python - <<'PY'
import glob
from polyglot_lc import emphasis
from polyglot_lc.ingest import load_document

paths = sorted(glob.glob('input/*.pdf'))
if not paths:
    print('input/ rỗng: bỏ qua phép đếm')
else:
    doc = load_document(paths[0])
    marked = [b for b in doc.blocks if any(tok in b.text for tok in emphasis.TOKENS)]
    print(f'{paths[0]}: {len(marked)}/{len(doc.blocks)} block mang token')
    for b in marked[:5]:
        print('  ', b.text[:90])
PY
```
Expected: in ra đường dẫn tài liệu và tỷ lệ block mang token (hoặc thông báo bỏ qua nếu `input/` rỗng). Con số này là **dữ liệu** để biết tính năng chạm bao nhiêu phần tài liệu thật — ghi vào báo cáo, **không** đặt làm điều kiện pass/fail.

- [ ] **Step 4: Commit**

```bash
git add src/polyglot_lc/pipeline/pdf_output.py src/polyglot_lc/ingest/pdf_layout.py
git commit -m "docs: describe emphasis sentinels in the ingest and rebuild modules"
```
