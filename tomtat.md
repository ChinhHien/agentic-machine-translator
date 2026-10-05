# polyglot-lc — Tóm tắt toàn diện

> Tài liệu này giải thích **toàn bộ** dự án: thuật ngữ, luồng xử lý dữ liệu, các agent,
> cấu hình, và cách dự án tiêu thụ token. Mọi khẳng định kỹ thuật đều trích dẫn từ mã nguồn.
>
> Bổ sung cho [`README.md`](README.md) (hướng dẫn cài đặt/cách dùng) và
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (sơ đồ kiến trúc chi tiết).

---

## Mục lục

1. [Dự án này làm gì](#1-dự-án-này-làm-gì)
2. [Bảng thuật ngữ](#2-bảng-thuật-ngữ)
3. [Kiến trúc tổng quan](#3-kiến-trúc-tổng-quan)
4. [Luồng xử lý dữ liệu chi tiết](#4-luồng-xử-lý-dữ-liệu-chi-tiết)
5. [Năm agent và tác dụng của từng agent](#5-năm-agent-và-tác-dụng-của-từng-agent)
6. [Ba phân hệ con: Reasoning Bank, SkillOpt, Translation Memory](#6-ba-phân-hệ-con)
7. [Prompt caching — kỹ thuật quan trọng nhất của dự án](#7-prompt-caching-kỹ-thuật-quan-trọng-nhất-của-dự-án)
8. [Định tuyến model/provider (kiến trúc Hybrid)](#8-định-tuyến-modelprovider-kiến-trúc-hybrid)
9. [Chịu lỗi: retry, backup, concurrency](#9-chịu-lỗi-retry-backup-concurrency)
10. [Cấu hình đầy đủ](#10-cấu-hình-đầy-đủ)
11. [Chi phí token đo được thực tế](#11-chi-phí-token-đo-được-thực-tế)
12. [Câu hỏi thường gặp](#12-câu-hỏi-thường-gặp)
13. [Bản đồ mã nguồn](#13-bản-đồ-mã-nguồn)

---

## 1. Dự án này làm gì

**polyglot-lc** là công cụ dịch máy **văn bản dài** (sách, báo cáo, tài liệu kỹ thuật) qua
nhiều ngôn ngữ, dùng kiến trúc **đa agent**.

Điểm khác biệt so với gọi API dịch thông thường:

| Đặc điểm | polyglot-lc |
| --- | --- |
| Chất lượng | Nhiều agent kiểm tra chéo nhau, không phải một lần dịch duy nhất |
| Văn bản dài | Cắt đoạn thông minh theo ngữ nghĩa, có chồng lấn để không mất ngữ cảnh |
| Tự cải thiện | Ghi nhớ kinh nghiệm (Reasoning Bank) và tự tối ưu prompt (SkillOpt) |
| Linh hoạt về nhà cung cấp | Mỗi agent dùng provider/model **khác nhau** được |
| Tiết kiệm tiền | Chia sẻ cache giữa các agent đọc cùng một nguồn |

**Yêu cầu:** Python 3.11 trở lên. Không cần GPU nếu dùng API cloud.

---

## 2. Bảng thuật ngữ

Phần này giải thích mọi thuật ngữ chuyên môn xuất hiện trong dự án.

### Thuật ngữ AI/LLM

| Thuật ngữ | Nghĩa tiếng Việt |
| --- | --- |
| **LLM** (Large Language Model) | Mô hình ngôn ngữ lớn — mô hình AI được huấn luyện trên lượng lớn văn bản, có khả năng hiểu và sinh ra ngôn ngữ tự nhiên |
| **Token** | Đơn vị nhỏ nhất mà mô hình đọc. 1 token ≈ 0,75 từ tiếng Anh, ≈ 1,5–2 từ tiếng Việt (tiếng Việt dùng nhiều token hơn) |
| **Prompt** | Câu lệnh/câu hỏi gửi cho LLM. Prompt càng dài, LLM càng tốn kém |
| **Context window** | Cửa sổ ngữ cảnh — số token tối đa LLM đọc được trong một lần gọi |
| **Prompt caching** | Provider lưu lại prompt đã dùng; gọi lại với prompt giống nhau thì tính tiền rẻ hơn |
| **Stable prefix** (tiền tố ổn định) | Phần đầu của prompt **không thay đổi** giữa các lần gọi — điều kiện tiên quyết để cache hoạt động |
| **Embedding** (vector đại diện) | Biến một đoạn văn bản thành dãy số để máy tính đo độ tương đồng ngữ nghĩa |
| **Temperature** | Mức "sáng tạo" (0 = chọn an toàn nhất, 1+ = linh hoạt hơn nhưng dễ sai) |
| **Few-shot** | Đưa kèm vài ví dụ mẫu trong prompt để LLM học theo khuôn mẫu |
| **JSON mode** | Chế độ bắt LLM trả lời đúng định dạng JSON để chương trình đọc được |
| **Context / prefix caching hit** | Provider tái sử dụng phần prompt đã cache thay vì đọc lại từ đầu |

### Thuật ngữ dịch thuật

| Thuật ngữ | Nghĩa tiếng Việt |
| --- | --- |
| **Machine Translation (MT)** | Dịch máy — dịch bằng máy chứ không phải con người |
| **NMT** (Neural Machine Translation) | Dịch máy thần kinh — dịch máy dùng mạng nơ-ron (kiểu mô hình LLM hiện đại) |
| **Glossary** | Bảng thuật ngữ — danh sách thuật khoá/thuật ngữ chuyên ngành và cách dịch chuẩn của chúng |
| **Terminology consistency** | Tính nhất quán thuật ngữ — cùng một thuật ngữ phải luôn được dịch giống nhau trong cả tài liệu |
| **Fidelity** | Tính trung thành — bản dịch có truyền đủ ý gốc không, không thêm bớt |
| **Fluency** | Tính trôi chảy — bản dịch có tự nhiên, đọc không gượng gạo không |
| **Literal translation** | Dịch sát/lit — dịch từng từ, gần nghĩa gốc nhưng đôi khi không tự nhiên |
| **Calque** | Dịch nghĩa — dịch cấu trúc câu của ngôn ngữ nguồn, dùng khi cấu trúc đó không hợp với tiếng Việt |
| **Transliteration** | Phiên âm — viết tên riêng theo âm gốc thay vì dịch nghĩa (VD: "Shakespeare" → "Shakespeare" chứ không phải "Sê-pha-kiê") |
| **Code shielding** | Bảo vệ mã nguồn — thay thế khối code bằng placeholder trước khi dịch, rồi khôi phục sau, để LLM không "dịch nhầm" code |
| **Code placeholder** | Ký hiệu tạm thay cho khối code, dạng `⟦CODE_1⟧` |
| **Pitfall** (lỗi thường gặp) | Danh sách lỗi dịch thường gặp cần tránh trong ngôn ngữ đích |

### Thuật ngữ kiến trúc phần mềm

| Thuật ngữ | Nghĩa tiếng Việt |
| --- | --- |
| **Pipeline** | Chuỗi quy trình xử lý: dữ liệu đi qua các bước tuần tự rồi ra kết quả |
| **Agent** (tác nhân) | Một thành phần AI có một vai trò riêng, có prompt riêng, model riêng. Giống như nhân viên mỗi người một nhiệm vụ |
| **Orchestrator** (điều phối) | Thành phần quyết định **agent nào chạy tiếp theo** dựa trên kết quả trước đó |
| **State** (trạng thái) | Dữ liệu đang được xử lý, truyền giữa các bước. VD: bản nháp hiện tại, điểm đã chấm |
| **Graph** (đồ thị) | Cách biểu diễn luồng điều phối dạng nút + cạnh. LangGraph dùng cơ chế này |
| **Conditional edge** (cạnh có điều kiện) | Cạnh đồ thị chỉ được đi khi điều kiện thỏa mãn — VD: "chấm < 7 thì refine, ngược lại verify" |
| **Concurrency** (song song) | Xử lý nhiều việc cùng lúc thay vì tuần tự |
| **Thread-safe** (an toàn luồng) | Dữ liệu dùng chung an toàn khi nhiều luồng truy cập cùng lúc |
| **Best-effort** (cố gắng hết sức) | Thành phần phụ; nếu lỗi thì chỉ ghi cảnh báo, **không** làm hỏng kết quả chính |
| **Idempotent-safe** | Chạy lại nhiều lần cho kết quả không khác |
| **Atomic write** (ghi nguyên tử) | Ghi file qua file tạm rồi đổi tên — tránh file bị hỏng giữa chừng |
| **Registry pattern** | Một bảng ánh xạ định dạng file → hàm xử lý, cho phép thêm định dạng mới không sửa code cũ |

---

## 3. Kiến trúc tổng quan

```
                     ┌────────────────────────────────────────────────┐
                     │                  polyglot-lc                  │
 .md/.txt/.pdf ──Ingest──▶ NFC + code-shield ──▶ Chunker (tiktoken) │
                     │        │                    │                │
                     │        ▼                    ▼                │
                     │  Summary Tree         Glossary                │
                     │        │                    │                │
                     │        ▼                    ▼                │
                     │  ┌───────────── LangGraph pipeline ────────┐  │
   Reasoning Bank ◀───┼──┤ Planner → Translator → Critic          │  │
   (ChromaDB) ────────┼──┤      ▲           │ (score < ngưỡng)  │  │
                     │  │      │           ▼                    │  │
   SkillOpt ─────────┼──┤      └──── Refiner (tối đa 2 vòng)   │  │
   (skills/*.json)   │  │                  │                   │  │
                     │  │                  ▼                   │  │
                     │  │             Verifier ──fail──▶ Dịch lại│  │
                     │  │                  │ pass                │  │
                     │  │                  ▼                     │  │
                     │  │        save_trace → chunk tiếp theo    │  │
                     │  └──────────────────┼──────────────────────┘  │
                     │                     ▼                        │
                     │              Assemble → SkillOpt             │
                     └─────────────────────────────────────────────┘
```

### Cấu trúc thư mục

```
src/polyglot_lc/
├── cli.py                 # Giao diện dòng lệnh (typer) + báo cáo token
├── config.py              # Cấu hình per-agent, nhóm model, cache, retry
├── pipeline.py            # Điều phối end-to-end + báo cáo kết quả
├── ingest/                # Nạp file: registry, md, txt, pdf (MarkItDown)
├── segment/               # Cắt đoạn: chunker, summary_tree, glossary
├── agents/                # base, context, planner, translator, critic,
│                          #   refiner, verifier
├── orchestrator/          # state, router, graph (LangGraph)
├── reasoning_bank/        # store (ChromaDB), embedder, schema
├── skillopt/              # optimizer, skill_store
├── memory/tm.py           # Translation Memory (bộ nhớ dịch)
├── llm/                   # client.py (OpenAI-compatible), resilience.py
└── ui/                    # simple.py (in từng bước), events.py (log JSONL)
```

---

## 4. Luồng xử lý dữ liệu chi tiết

### Bước 0 — Nạp cấu hình

Trước khi làm bất cứ việc gì, [cli.py](src/polyglot_lc/cli.py) đọc cấu hình từ `.env`:
kiểm tra đủ 5 agent + endpoint embedding. Thiếu biến bắt buộc → in lỗi rõ ràng và
thoát với mã 2.

> **Cơ chế quan trọng:** [config.py:25](src/polyglot_lc/config.py#L25) gọi `load_dotenv()`
> **không** có `override=True`. python-dotenv mặc định *không* ghi đè biến đã có trong
> môi trường — nghĩa là **biến truyền từ shell luôn thắng file `.env`**. Đây là cách an toàn
> để test với config khác mà không sửa `.env`:
>
> ```bash
> SOURCE_GROUP_MODEL=gemini-2.5-flash polyglot translate --file cuonsach.md
> ```

### Bước 1 — Ingest (nạp & chuẩn hóa)

[pipeline.py:170](src/polyglot_lc/pipeline.py#L170) → [ingest/registry.py](src/polyglot_lc/ingest/registry.py)

`registry.py` dùng **Registry pattern**: bảng ánh xạ đuôi file → hàm xử lý.

| Đuôi file | Loader | Cách xử lý |
| --- | --- | --- |
| `.md`, `.markdown` | `ingest/md.py` | Parse trực tiếp bằng `markdown-it-py` |
| `.txt`, `.text` | `ingest/txt.py` | Bao thành một khối văn bản |
| `.pdf` | `ingest/pdf.py` | Chuyển PDF → Markdown bằng **MarkItDown**, rồi dùng chung parser `.md` |

Trong `md.py` có 4 việc quan trọng:

1. **NFC normalization** — chuẩn hoá Unicode. File nguồn có thể dùng ký tự tổ hợp
   (VD: `e` viết bằng 2 byte) khác với dạng dựng sẵn; NFC gộp về một dạng để so khớp
   ổn định. Nếu không làm bước này, các phép so khớp chuỗi về sau sẽ hỏng.
2. **Code shielding** — thay mọi khối code (cả fenced ```` ``` ```` lẫn thụt lề 4 space)
   bằng placeholder `⟦CODE_1⟧`, lưu ánh xạ trong `code_map`. Lý do: nếu đưa code thật
   cho LLM, nó sẽ "dịch" cả comment trong code và làm hỏng chương trình.
3. **Tách Block cấu trúc** — mỗi đơn vị là một `Block`:
   ```python
   kind  # "heading" | "paragraph" | "list" | "code" | "quote"
   text  # nội dung
   level # cấp heading (1..6)
   meta  # thông tin phụ (VD: các item của list)
   ```
4. **Bỏ dấu nhấn inline** — `*đậm*` → `đậm` (giữ nội dung, bỏ ký hiệu), vì dấu `*`
   gây nhiễu khi LLM dịch.

> **Code shielding là best-effort, không phải đảm bảo tuyệt đối.** Regex ở
> [md.py:49](src/polyglot_lc/ingest/md.py#L49) không bắt được mọi trường hợp (ví dụ code
> lồng trong blockquote phức tạp). Dự án cũng kiểm tra ở [verifier.py:38](src/polyglot_lc/agents/verifier.py#L38):
> nếu số `⟦CODE_n⟧` ở bản dịch không khớp số ở bản gốc → bản dịch bị đánh hỏng.

### Bước 2 — Phát hiện ngôn ngữ

Nếu `--src auto`, [pipeline.py:64](src/polyglot_lc/pipeline.py#L64) dùng **langdetect** trên
2.000 ký tự đầu. Lỗi → mặc định `en`.

### Bước 3 — Chunker (cắt đoạn)

[segment/chunker.py](src/polyglot_lc/segment/chunker.py) — **tim trọng tâm của dự án**.

| Khái niệm | Giải thích |
| --- | --- |
| **Chunk** | Một đơn vị dịch. Mỗi chunk có `id`, `text`, `token_count`, `section_path` |
| **Token budget** | Ngân sách token mỗi chunk, lấy từ `CHUNK_MAX_TOKENS` (mặc định 700) |
| **Overlap** | Phần cuối chunk trước được lặp lại ở đầu chunk sau (`CHUNK_OVERLAP_TOKENS`, mặc định 80) |

Nguyên tắc hoạt động:

1. **Đo token bằng `tiktoken`** theo model của Translator (`get_encoding` thử
   `encoding_for_model`, lỗi thì lùi về `cl100k_base`). Tức là ngân sách token khớp với
   đúng tokenizer mà provider sẽ dùng.
2. **Không cắt giữa Block** — chỉ cắt ở ranh giới đoạn văn/heading, tránh cắt giữa câu.
3. **Hard guard** ([chunker.py:92](src/polyglot_lc/segment/chunker.py#L92)) — nếu một Block
   đơn lẻ đã vượt ngân sách, vẫn giữ nguyên thành một chunk. Không cắt nát vì cắt nát
   câu còn tệ hơn.
4. **Overlap** — sau khi đóng một chunk, lấy phần đuôi (≤ `overlap_tokens`) làm đầu
   chunk kế tiếp.

> **Vì sao cần overlap?** Câu nối hai câu thường nằm ở ranh giới chunk. Nếu không overlap,
> "Tôi nghĩ rằng đó là sai. **Vì vậy** chúng ta cần xem lại." sẽ bị cắt đôi và chunk sau
> mất ngữ cảnh "vì sao". Overlap đảm bảo không câu nào bị mất liên kết.

> **Chunk size ảnh hưởng chi phí mạnh nhất.** Đo thực tế trên sách ~505 trang:
> `700` → 550 chunk; `1500` → 220 chunk. Chunk lớn hơn = ít lần gọi API hơn = tổng token
> thấp hơn (xem [mục 11](#11-chi-phí-token-đo-được-thực-tế)). Lý do: phần "overhead" lặp
> lại ở mỗi chunk (strategy + glossary + chỉ dẫn) được trả nhiều lần hơn khi chunk nhỏ.

### Bước 4 — Glossary (bảng thuật ngữ)

[segment/glossary.py](src/polyglot_lc/segment/glossary.py) trích tự động các thuật ngữ
ứng viên theo 3 heuristic:

- Cụm danh từ riêng (viết hoa không ở đầu câu)
- Từ viết tắt (ALL-CAPS, ≥ 2 ký tự)
- Từ lặp ≥ `min_count` lần

Cộng thêm glossary thủ công từ file CSV (`--glossary`), giới hạn **200 mục**
(`MAX_USER_GLOSSARY_TERMS`) để bảo đảm context không phình vô hạn.

Kết quả `glossary_md` được nhét vào **mọi** prompt của nhóm read-source → bảo đảm
tính nhất quán thuật ngữ toàn tài liệu.

### Bước 5 — Summary Tree (cây tóm tắt)

[segment/summary_tree.py](src/polyglot_lc/segment/summary_tree.py) xây cây 3 tầng:
**doc → section → chunk**.

Điểm tối ưu đáng chú ý: **chỉ tóm tắt ở tầng section và doc**, bỏ qua tầng chunk vì
những tóm tắt đó không được dùng ở đâu. Biến `SUMMARY_TREE_CHUNK_SUMMARIES=0`
(mặc định) tiết kiệm **1 request mỗi chunk**.

Số request Planner = `số section + 1` (doc).

### Bước 6 — Plan cấp tài liệu

[graph.py:69](src/polyglot_lc/orchestrator/graph.py#L69) `plan_document()` chạy **đúng một
lần**, tuần tự, không song song:

1. Truy vấn Reasoning Bank với `f"{domain} {doc_summary[:200]}"`, lấy `RB_TOP_K` trace giống nhất
2. Gọi Planner sinh JSON:
   ```json
   {"strategy": "...", "few_shot": ["src -> tgt", ...], "pitfalls": ["..."], "notes": "..."}
   ```

Kết quả này được đưa vào `base_state` — **mọi chunk đều dùng chung** một chiến lược dịch.

### Bước 7 — Xử lý song song từng chunk

[pipeline.py:120](src/polyglot_lc/pipeline.py#L120) `_run_chunks()` dùng `ThreadPoolExecutor`
với `MAX_CONCURRENCY` worker (mặc định 2).

Rất quan trọng — **hai mức song song tách biệt**:

| Mức | Song song? | Lý do |
| --- | --- | --- |
| Giữa các chunk | ✅ Có (`MAX_CONCURRENCY`) | Chunk độc lập nhau, không phụ thuộc |
| Trong một chunk (translate→critic→refine→verify) | ❌ Không | Critic cần bản nháp của Translator; Refiner cần critique của Critic |

Kết quả được **giữ đúng thứ tự chunk gốc** bất kể chunk nào hoàn thành trước.

### Bước 8 — Đồ thị LangGraph mỗi chunk

[orchestrator/graph.py:108](src/polyglot_lc/orchestrator/graph.py#L108) dựng 7 node:

```
translator → critic → ┬─(điểm thấp)─→ refiner → critic (vòng lại)
                      └─(điểm cao)─→ verifier → ┬─(fail)─→ retranslate → verifier
                                                   └─(pass)→ collect → save_trace → END
```

Điều kiện rẽ nhánh nằm ở [orchestrator/router.py](src/polyglot_lc/orchestrator/router.py):

```python
after_critic(state, CRITIC_THRESHOLD, REFINE_ROUNDS, REFINE_MIN_SCORE):
    nếu đã đủ số vòng refine  → "verify"
    nếu điểm >= REFINE_MIN_SCORE → "verify"     # cổng chặn thích nghi (adaptive)
    nếu điểm < CRITIC_THRESHOLD → "refine"
    ngược lại → "verify"
```

### Bước 9 — Assemble (ghép file)

[pipeline.py:272](src/polyglot_lc/pipeline.py#L272) nối các chunk **theo thứ tự gốc** (sắp
xếp theo `chunk.id`), gọi `restore_code()` để thay placeholder bằng code thật, rồi ghi file
đầu ra (mặc định `<tên>.<ngôn_ngữ_đích><đuôi>`).

### Bước 10 — SkillOpt

Cuối run, [skillopt/optimizer.py](src/polyglot_lc/skillopt/optimizer.py) gom
`(prompt, output, điểm_critic)` của mọi chunk, nhờ Planner đề xuất **textual gradient**
mới, rồi chỉ **promote** nếu trung bình điểm 20 chunk gần nhất **cao hơn** skill hiện tại.

### Bước 11 — Báo cáo

[cli.py:46](src/polyglot_lc/cli.py#L46) in: số chunk, điểm critic trung bình, số retry,
số trace đã lưu, skill có được promote hay không, đường dẫn file ra, và **báo cáo token**
chi tiết theo từng provider/model.

---

## 5. Năm agent và tác dụng của từng agent

### Bảng tổng quan

| Agent | Chạy khi nào | Model env | Nhiệm vụ | Tần suất |
| --- | --- | --- | --- | --- |
| **Planner** | 1 lần/run | `PLANNER_*` | Chọn chiến lược dịch cho cả tài liệu | Rất ít |
| **Translator** | Mỗi chunk | `SOURCE_GROUP_*` | Dịch 1 chunk | Nhiều nhất |
| **Critic** | Sau mỗi bản nháp | `SOURCE_GROUP_*` | Chấm điểm 0–10 + góp ý | Nhiều |
| **Refiner** | Khi điểm thấp | `SOURCE_GROUP_*` | Viết lại theo góp ý | Trung bình |
| **Verifier** | Cuối mỗi chunk | `VERIFIER_*` | Kiểm tra chất lượng cuối | Nhiều |

### 5.1 Planner — "Trưởng nhóm dịch"

[agents/planner.py](src/polyglot_lc/agents/planner.py). Chạy **đúng một lần** cho cả tài liệu.

Nhiệm vụ: nhìn tóm tắt toàn bộ tài liệu + glossary + các trace cũ, rồi ra **một quyết định
chiến lược** áp dụng cho mọi chunk:

- **`strategy`** — chiến lược dịch, VD "Dịch sát, giữ thuật ngữ nhất quán, ưu tiên
  tiếng Việt tự nhiên hơn dịch từng từ"
- **`few_shot`** — tối đa 6 ví dụ mẫu `nguồn → đích`
- **`pitfalls`** — tối đa 6 lỗi thường gặp cần tránh

Nhiệt độ `0.2` (thấp) vì cần JSON ổn định, không cần sáng tạo.

Ngoài ra Planner còn phục vụ `summarize_with_planner()` (summary tree) và
`SkillOptimizer.propose_gradient()`.

### 5.2 Translator — "Người dịch"

[agents/translator.py](src/polyglot_lc/agents/translator.py). **Agent chạy nhiều nhất.**

Prompt gồm 2 message:
1. **System message** = stable prefix dùng chung (xem [mục 7](#7-prompt-caching-kỹ-thuật-quan-trọng-nhất-của-dự-án))
2. **User message** = chỉ dẫn + few-shot + khớp Translation Memory + luật chơi

Sau khi dịch, kết quả được lưu vào Translation Memory để chunk sau có thể tái dùng.

### 5.3 Critic — "Người phê bình"

[agents/critic.py](src/polyglot_lc/agents/critic.py). Đọc **cả bản gốc lẫn bản nháp**,
trả về JSON:

```json
{"score": 8.5, "issues": ["..."], "suggestions": ["..."]}
```

`feedback_text` ([critic.py:21](src/polyglot_lc/agents/critic.py#L21)) rút gọn thành khối văn
bản gọn để đưa cho Refiner. Nhiệt độ thấp (`0.1`–`0.3`) vì chấm điểm cần nhất quán.

> **Phòng vệ chống bản dịch liệt.** Nếu model trả JSON hỏng, `review()` bắt exception và
> trả `score=7.0` ("critic reply unparseable") — chấm là đạt để pipeline không bị kẹt.

### 5.4 Refiner — "Biên tập viên"

[agents/refiner.py](src/polyglot_lc/agents/refiner.py). Nhận (bản nháp + critique) và viết
lại. **Chỉ chạy khi `CRITIC_THRESHOLD`/`REFINE_MIN_SCORE` cho thấy cần thiết**, tối đa
`REFINE_ROUNDS` vòng (mặc định 2). Nhiệt độ cao hơn (`0.4`) vì đây là viết sáng tạo có kiểm soát.

### 5.5 Verifier — "Kiểm tra viên"

[agents/verifier.py](src/polyglot_lc/agents/verifier.py). Chạy 2 tầng:

**Tầng 1 — hard checks (tất định, không tốn token)** ([verifier.py:33](src/polyglot_lc/agents/verifier.py#L33)):

| Kiểm tra | Cách làm | Vì sao |
| --- | --- | --- |
| Code placeholder | So số `⟦CODE_n⟧` giữa gốc và dịch | LLM hay làm mất placeholder → code hỏng |
| Cấu trúc heading | So chuỗi cấp `#` | LLM hay bỏ hoặc thêm heading |

Nếu hỏng → **fail ngay, không gọi LLM** → tiết kiệm token.

**Tầng 2 — LLM verdict**: gửi `source[:2000]` + `translation[:2000]`, hỏi
`{"pass": true|false, "reasons": [...]}`.

Nhiệt độ `0.0` — tuyệt đối tất định.

> **Phòng vệ chống chặn.** Nếu LLM lỗi, `verify()` trả `ok=True` với lý do "verifier
> unavailable; skipped" — hệ thống không bao giờ để một chunk hỏng chặn cả bản dịch.

### Vì sao chia nhóm Translator/Critic/Refiner?

Đây là quyết định kiến trúc quan trọng nhất ([config.py:46](src/polyglot_lc/config.py#L46)).

**Cả ba đều đọc lại cùng một đoạn gốc.** Nếu dùng ba model khác nhau, đoạn gốc bị tính tiền
3 lần. Đặt cả ba trên **một** provider/model thì prefix giống nhau được cache → chỉ trả tiền
1 lần, hai lần còn lại rẻ hơn nhiều.

Planner (chạy 1 lần, đọc tóm tắt) và Verifier (đọc bản dịch cuối) không chia sẻ prefix
nên tách riêng — có thể dùng model rẻ hơn.

---

## 6. Ba phân hệ con

### 6.1 Reasoning Bank — "Sổ ghi kinh nghiệm"

[reasoning_bank/](src/polyglot_lc/reasoning_bank/). Lưu trữ kinh nghiệm dịch bằng
**ChromaDB** (cơ sở dữ liệu vector).

Schema trace ([schema.py:14](src/polyglot_lc/reasoning_bank/schema.py#L14)):

```python
domain, src_lang, tgt_lang, strategy, few_shot[], pitfalls[], score, ts
```

- **`score`** chuẩn hoá `0..1` = `điểm_critic / 10`
- **Cổng lưu**: chỉ ghi khi `score >= MIN_UPSERT_SCORE` (0.75) — không lưu cảnh dịch tệ
- **Truy vấn**: mỗi lần chạy, plan cầu `RB_TOP_K` trace gần nhất về chủ đề

> **Best-effort.** Cả truy vấn lẫn ghi đều bọc `try/except` ([pipeline.py:215](src/polyglot_lc/pipeline.py#L215)).
> Reasoning Bank hỏng thì pipeline vẫn chạy, chỉ mất tính "tích lũy kinh nghiệm".

### 6.2 SkillOpt — "Tự tối ưu prompt"

[skillopt/](src/polyglot_lc/skillopt/). Khác Reasoning Bank ở chỗ: SkillOpt **tự sửa
chỉ thần dịch**.

Mỗi chunk lưu `(prompt, output, điểm_critic)`. Cuối run, Planner đọc 5 mẫu gần nhất và đề
xuất `instruction` mới + tối đa 3 `few_shot`.

**Quy tắc promote** ([optimizer.py:118](src/polyglot_lc/skillopt/optimizer.py#L118)):

```
chỉ promote nếu  trung bình 20 chunk gần nhất  >  trung bình skill hiện tại
```

Skill lưu ở `skills/<src>-<tgt>/<domain>.json`, ghi **nguyên tử** (file tạm → đổi tên).

### 6.3 Translation Memory — "Ghi nhớ trong cuộn dịch"

[memory/tm.py](src/polyglot_lc/memory/tm.py). Chỉ sống trong **một lần chạy**, không lưu
đĩa. Mỗi cặp (nguồn → đích) được lưu; khi dịch chunk sau, `lookup()` tìm khớp:

1. Khớp **chính xác** trước
2. Khớp **mờ** (fuzzy) bằng `difflib.SequenceMatcher`, ngưỡng `0.85`, trên 400 ký tự đầu

Mọi truy cập đều khóa (`thread-safe`) vì chunk chạy song song.

---

## 7. Prompt caching — kỹ thuật quan trọng nhất của dự án

### 7.1 Nguyên lý

Hầu hết provider tính tiền **input token theo lần gọi**. Nếu cùng một đoạn văn gốc được gửi
3 lần (cho Translator, Critic, Refiner) thì trả tiền 3 lần — trừ khi provider nhận ra
phần **tiền tố ổn định** (stable prefix) và cache lại.

### 7.2 Cách dự án đảm bảo prefix thật sự giống nhau

[agents/context.py](src/polyglot_lc/agents/context.py) dựng stable prefix theo thứ tự
**cố định và xác định**:

```
role + ngôn ngữ → strategy → pitfalls → glossary → văn bản gốc
```

Ba agent đều đặt prefix này làm **`messages[0]`** (system message), và phần động (bản
nháp, critique, chỉ dẫn) nằm **sau**. Nhờ vậy byte đầu tiên của request trùng nhau.

### 7.3 Ba chế độ `PROMPT_CACHE`

| Chế độ | Hành vi | Dùng khi |
| --- | --- | --- |
| `auto` (mặc định) | Không gửi marker; chỉ sắp xếp prompt ổn định và **đọc metric** cache trả về | OpenAI, DeepSeek, Gemini, Groq |
| `explicit` | Thêm `cache_control` breakpoint kiểu Anthropic | Proxy OpenAI-compatible tới Anthropic |
| `off` | Không gửi gì | Ollama, vLLM (không có cache) |

Cú pháp alias: `on`/`markers` → `explicit`, `auto` → `auto`, `off` → `off`.

### 7.4 Báo cáo HIT / MISS / UNKNOWN

[client.py](src/polyglot_lc/llm/client.py) đọc metric cache từ **3 hãng khác nhau**:

| Hãng | Trường metric |
| --- | --- |
| OpenAI | `usage.prompt_tokens_details.cached_tokens` |
| DeepSeek | `usage.prompt_cache_hit_tokens` |
| Anthropic | `usage.cache_read_input_tokens` |

Nguyên tắc thiết kế quan trọng: **thiếu metric ≠ MISS**. Provider không báo cáo thì đánh
dấu `UNKNOWN`, không đoán. Nếu không, báo cáo sẽ phủ định sự thật.

### 7.5 Fingerprint (chỉ để debug)

`stable_prefix_fingerprint()` ([context.py:96](src/polyglot_lc/agents/context.py#L96)) trả
về SHA1 16 ký tự của (version + model + prefix). Giá trị này **không bao giờ chứa văn bản
gốc** — chỉ là con số, nên an toàn khi log.

### 7.6 Tác động thực tế

Đo trên sách ~505 trang: stable prefix ≈ **917 token/chunk**. Nếu prefix luôn HIT, input
của nhóm read-source rơi từ **~2.31M → ~504k token** (tiết kiệm ~1.8M).

> **Quan trọng:** chỉ provider **có** prompt cache mới hưởng được. Ollama/vLLM không có,
> nên đừng hy vọng con số này khi test local.

---

## 8. Định tuyến model/provider (kiến trúc Hybrid)

### 8.1 Cơ chế kế thừa

[config.py:150](src/polyglot_lc/config.py#L150) đọc env theo thứ tự:

```
<NAME>_BASE_URL/MODEL/API_KEY/TEMPERATURE     (ưu tiên cao nhất)
        ↓ nếu trống
SOURCE_GROUP_BASE_URL/MODEL/API_KEY/TEMPERATURE   (chỉ TRANSLATOR/CRITIC/REFINER)
        ↓ nếu trống
demo defaults (chỉ khi LLM_MOCK=1)
```

### 8.2 Ba nhóm

| Nhóm | Agent | Prefix env | Ghi chú |
| --- | --- | --- | --- |
| planner | PLANNER | `PLANNER_*` | Độc lập |
| source | TRANSLATOR, CRITIC, REFINER | `SOURCE_GROUP_*` | Chia sẻ cache |
| verifier | VERIFIER | `VERIFIER_*` | Độc lập |

### 8.3 Ba lỗi cấu hình dễ mắc

> **1. Quên bỏ `SOURCE_GROUP_*`.** Nếu set `TRANSLATOR_MODEL` nhưng quên `CRITIC_MODEL`,
> Critic rơi về group — có thể là model trả phí ngoài ý muốn.
>
> **2. Quên `TEMPERATURE`.** Nó cũng kế thừa từ group. Set `VERIFIER_BASE_URL/MODEL/API_KEY`
> mà quên `VERIFIER_TEMPERATURE=0.0` → verifier chạy ở 0.3, chấm điểm nhiễu hơn.
>
> **3. Tưởng tách được quota.** Rate limit tính **theo model**, không theo key. Dùng
> `gemini-2.5-flash` cho cả Planner lẫn Translator vẫn là **một** hạn mức chung. Muốn tách
> hạn mức phải dùng **model khác**.

### 8.4 Backup endpoint

Mỗi agent có thể có `<NAME>_BACKUP_*`. Chỉ chuyển sang backup sau khi retry cạn
([client.py:236](src/polyglot_lc/llm/client.py#L236)), và **chạy đúng một lần** trên backup.
Backup không lồng nhau. Bắt buộc có **cả hai** `BACKUP_BASE_URL` và `BACKUP_MODEL`.

---

## 9. Chịu lỗi: retry, backup, concurrency

### 9.1 Phân loại lỗi

[llm/resilience.py](src/polyglot_lc/llm/resilience.py) chuẩn hoá mọi exception thành 3 loại:

| Loại | HTTP | Xử lý |
| --- | --- | --- |
| `RateLimitError` | 429 | Chờ theo gợi ý provider, rồi thử lại |
| `TransientError` | 408, 409, 425, 500, 502, 503, 504, timeout, lỗi mạng | Backoff luỹ thừa, thử lại |
| `PermanentError` | 400, 401, 403, 404, 422 | **Không** thử lại, **không** fallback — đổi endpoint cũng vô ích |

### 9.2 Đọc gợi ý chờ của provider

`parse_retry_seconds()` hiểu cả:

- `Please try again in 47.4s` → 47.4 giây
- `try again in 900ms` → 0.9 giây
- `Retry-After: 12` → 12 giây
- `try again in 2 minutes` → 120 giây

Rất nhiều provider free plan **không** trả header mà nhét số vào message — nên phải parse
text. Sau đó cộng thêm `LLM_RETRY_BUFFER_SECONDS`.

### 9.3 Backoff + jitter

```
chờ = min(max_delay, base_delay × 2^attempt)     (2s, 4s, 8s, 16s…)
chờ = chờ × (1 − jitter × random())               (giảm ngẫu nhiên, không bao giờ vượt)
```

**Jitter** (ngẫu nhiên hóa) chống "hiệu ứng đàn bầy": nếu 100 tiến trình cùng gặp 429 và cùng
chờ đúng 2.0s, chúng sẽ lại gặp 429 đúng lúc 2.0s. Biến thiên giữa các client phá vỡ đồng
bộ đó. `LLM_RETRY_JITTER=0` → không ngẫu nhiên (dùng cho test).

### 9.4 Tách hai loại "thử lại"

Đây là thiết kế đáng chú ý:

| Loại | Nguyên nhân | Cách xử lý |
| --- | --- | --- |
| **Hạ tầng** | 429, timeout, mạng | Retry `LLM_MAX_RETRIES` lần + fallback backup |
| **Ngữ nghĩa** | Verifier trả FAIL | `retranslate` tối đa **1** lần, kèm critique |

Hai đường này **không trộn lẫn** — tránh việc một lỗi chất lượng bị retry hạ tầng vô nghĩa.

### 9.5 Cạm bẫy dễ gặp

[llm/client.py:226](src/polyglot_lc/llm/client.py#L226) truyền `max_retries=0` cho OpenAI SDK,
vì lớp resilience tự xử lý. Nếu để SDK retry, ta sẽ có **hai tầng retry chồng nhau** → thời
gian chờ nhân bậy.

### 9.6 Các thành phần best-effort

Không được làm hỏng bản dịch, dù lỗi:

| Thành phần | Hành vi khi lỗi |
| --- | --- |
| Reasoning Bank truy vấn/ghi | Warning, chạy tiếp không có trace |
| Embedding endpoint | Chuyển sang hash vector ([embedder.py:37](src/polyglot_lc/reasoning_bank/embedder.py#L37)) |
| SkillOpt | Warning, bỏ qua tối ưu |
| Verifier LLM | Trả `pass=True`, ghi rõ lý do |

---

## 10. Cấu hình đầy đủ

### 10.1 Nhóm cấu hình

```bash
# === AGENT (5 agent) ===
PLANNER_BASE_URL, PLANNER_MODEL, PLANNER_API_KEY, PLANNER_TEMPERATURE
SOURCE_GROUP_BASE_URL, SOURCE_GROUP_MODEL, SOURCE_GROUP_API_KEY, SOURCE_GROUP_TEMPERATURE
# ghi đè từng agent:
TRANSLATOR_*, CRITIC_*, REFINER_*, VERIFIER_*
# backup (bắt buộc CẢ HAI base_url + model):
TRANSLATOR_BACKUP_BASE_URL, TRANSLATOR_BACKUP_MODEL, TRANSLATOR_BACKUP_API_KEY

# === EMBEDDING (bắt buộc) ===
EMBED_BASE_URL, EMBED_MODEL, EMBED_API_KEY

# === ĐƯỜNG DẪN ===
REASONING_BANK_PATH=./data/reasoning_bank
SKILLS_DIR=./skills
LOGS_DIR=./logs
LOG_JSONL=1

# === CHẤT LƯỢNG ===
CHUNK_MAX_TOKENS=700
CHUNK_OVERLAP_TOKENS=80
CRITIC_THRESHOLD=7.0
REFINE_ROUNDS=2
REFINE_MIN_SCORE=7.0
RB_TOP_K=4

# === HIỆU NĂNG ===
MAX_CONCURRENCY=2

# === CHỊU LỖI ===
LLM_MAX_RETRIES=3
LLM_RETRY_BASE_DELAY=2
LLM_RETRY_MAX_DELAY=60
LLM_RETRY_BUFFER_SECONDS=2
LLM_RETRY_JITTER=0.2

# === CACHE & GIAI PHÁP ===
PROMPT_CACHE=auto
SUMMARY_TREE_CHUNK_SUMMARIES=0

# === DEMO KHÔNG CẦN KEY ===
LLM_MOCK=0
```

### 10.2 Chế độ demo không cần key

```bash
LLM_MOCK=1 polyglot translate --file sample.md --src en --tgt vi
```

Toàn bộ pipeline chạy với mock trong bộ nhớ. Thiếu `BASE_URL`/`MODEL` cũng không lỗi vì
[config.py:209](src/polyglot_lc/config.py#L209) dựng giá trị demo.

### 10.3 Bảng tra nhanh tác động của từng biến

| Biến | Tăng lên | Hậu quả |
| --- | --- | --- |
| `CHUNK_MAX_TOKENS` | ↑ | Ít chunk, ít request, **tổng token giảm** |
| `MAX_CONCURRENCY` | ↑ | Nhanh hơn, nhưng **đốt RPM nhanh hơn** |
| `REFINE_ROUNDS` | ↑ | Chất lượng có thể ↑, token ↑↑ |
| `REFINE_MIN_SCORE` | ↑ | Bỏ qua refiner nhiều hơn, token ↓ |
| `CRITIC_THRESHOLD` | ↑ | Ít kích hoạt refiner |
| `RB_TOP_K` | ↑ | Prompt dài hơn, truy vấn chính xác hơn |
| `LLM_MAX_RETRIES` | ↑ | Chịu lỗi tốt hơn, chạy lâu hơn khi hỏng |
| `PROMPT_CACHE=explicit` | — | Bắt buộc nếu qua proxy Anthropic |

---

## 11. Chi phí token đo được thực tế

Đo bằng chính `chunk_document()` và các prompt builder của dự án trên một cuốn sách mô
phỏng **~505 trang** (242k từ tiếng Anh).

### 11.1 Với `CHUNK_MAX_TOKENS=700`

**550 chunk**, tổng **364,962 token** nguồn.

| Agent | Input/chunk |
| --- | --- |
| Translator | 1,014 |
| Critic | 1,638 |
| Refiner | 1,549 |
| Verifier | 866 |

Bản dịch tiếng Việt ≈ **432,548 token** (~786/chunk — tiếng Việt tốn nhiều token hơn tiếng Anh).

| Kịch bản | Request | Input | Output | **Tổng** |
| --- | ---: | ---: | ---: | ---: |
| **A. Lý tưởng** — mọi chunk đạt `REFINE_MIN_SCORE` | **1,741** | 2,072,771 | 499,208 | **~2.57M** |
| **B. Thực tế** — 60% chunk refine 2 vòng, 15% verify fail | **3,193** | 4,243,288 | 1,073,705 | **~5.32M** |

> Hai con số hơi khác nhau giữa bảng này và bảng 11.2 (`5.32M` vs `5.34M`) là do chạy
> hai script đo độc lập lấy mẫu ở số chunk khác nhau (40 chunk vs 30 chunk). Sai số
> ~0.4%, không ảnh hưởng kết luận.

### 11.2 Ảnh hưởng của `CHUNK_MAX_TOKENS`

| `CHUNK_MAX_TOKENS` | Chunk | Request (A) | Request (B) | Tổng token (A) | Tổng token (B) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 700 | 550 | 1,741 | 3,193 | 2.53M | 5.34M |
| 1000 | 358 | 1,153 | 2,099 | 1.94M | 4.09M |
| 1500 | 220 | 731 | 1,312 | 1.55M | 3.25M |
| 2000 | 138 | 480 | 845 | 1.31M | 2.74M |

**Tăng lên 1500: −58% request, −39% token tổng.**

### 11.3 Nơi token bị đốt

| Nhóm | Input | Output | Ghi chú |
| --- | ---: | ---: | --- |
| SOURCE_GROUP | ~1,475,000 | ~478,000 | Đa số |
| VERIFIER | ~490,000 | ~23,000 | **Input gấp 21× output** |
| PLANNER | ~121,000 | ~5,000 | Rẻ nhất |

> **Điểm đáng chú ý:** Verifier tốn 490k input token chỉ để sinh 23k output, vì prompt
> gửi cả `source[:2000]` **và** `translation[:2000]` ở mỗi chunk. Đây là nơi tiết kiệm lớn
> nhất còn lại.

### 11.4 Cấu hình đề xuất cho sách dài

```bash
CHUNK_MAX_TOKENS=1500
REFINE_MIN_SCORE=7.0
SUMMARY_TREE_CHUNK_SUMMARIES=0
PROMPT_CACHE=auto
```

### 11.5 Cảnh báo khi dùng free plan

Kịch bản A cần 1,741 request. Với `MAX_CONCURRENCY=2` → khoảng 20–40 request/phút.
Free tier 15 RPM sẽ **429 liên tục**. Muốn chạy bằng free plan:
`MAX_CONCURRENCY=1` + `CHUNK_MAX_TOKENS=1500` và chấp nhận chạy vài giờ — hoặc dùng
backup endpoint.

### 11.6 Quy đổi tiền

```
tiền = input_token/1e6 × giá_input  +  output_token/1e6 × giá_output
```

Ví dụ với giá Flash-tier ($0.30/M input, $2.50/M output): A ≈ **$1.87**, B ≈ **$3.96**.

---

## 12. Câu hỏi thường gặp

### Làm sao test config khác mà không sửa `.env`?

`load_dotenv()` không `override`, nên biến shell thắng `.env`:

```bash
SOURCE_GROUP_MODEL=gemini-2.5-flash MAX_CONCURRENCY=1 polyglot translate --file cuonsach.md
```

Chạy `polyglot translate ...` trần → tự động về config production.

### Dùng model free plan giới hạn RPM thì sao?

Tổ hợp: `MAX_CONCURRENCY=1` + `CHUNK_MAX_TOKENS=1500` + `LLM_RETRY_BASE_DELAY=10`.
Thiết lập backup endpoint để lúc hết quota vẫn chạy được. Chi tiết: [mục 9](#9-chịu-lỗi-retry-backup-concurrency).

### Token có bị đếm thừa không?

Không nếu provider hỗ trợ cache — stable prefix chỉ trả tiền một lần. Nhưng nếu dùng
Ollama/vLLM thì **không có** cache này.

### Prompt cache có hoạt động không?

Chạy và xem dòng `Cache (per request): hits/misses/unknown`. Nếu `unknown` chiếm đa số thì
provider không báo metric (không phải lỗi). Nếu `misses` cao → prefix không đủ giống nhau.

### Refiner chạy quá nhiều, token tăng vọt?

Hạ `REFINE_MIN_SCORE`. Kiểm tra `CRITIC_THRESHOLD` — model chấm điểm thấp sẽ kích hoạt
refine liên tục.

### Có dùng GPU không?

Không, nếu dùng API cloud. Nếu muốn chạy local hoàn toàn thì cần GPU/RAM cho Ollama hoặc
vLLM.

### Kết quả có tái lập được không?

Không hoàn toàn — LLM không tất định (temperature > 0) và chạy song song. Muốn tối đa
tính tất định: `TEMPERATURE=0`, `MAX_CONCURRENCY=1`, `LLM_RETRY_JITTER=0`.

### Chạy sách nhiều chương có khác nhau không?

Reasoning Bank lưu theo `domain`, SkillOpt lưu theo `skills/<src>-<tgt>/<domain>.json`.
Muốn tách, dùng `--domain` khác nhau.

### PDF có giữ được cấu trúc không?

Phụ thuộc độ phức tạp của PDF. PDF là định dạng trình bày, không có cấu trúc ngữ nghĩa —
MarkItDown phải suy đoán. Bảng phức tạp và sơ đồ có thể hỏng.

---

## 13. Bản đồ mã nguồn

| Muốn hiểu | Đọc file |
| --- | --- |
| Luồng tổng thể | [pipeline.py](src/polyglot_lc/pipeline.py) |
| Cấu hình env | [config.py](src/polyglot_lc/config.py) |
| Giao diện CLI | [cli.py](src/polyglot_lc/cli.py) |
| Nạp & chuẩn hoá file | [ingest/md.py](src/polyglot_lc/ingest/md.py), [ingest/pdf.py](src/polyglot_lc/ingest/pdf.py) |
| Cắt đoạn | [segment/chunker.py](src/polyglot_lc/segment/chunker.py) |
| Bảng thuật ngữ | [segment/glossary.py](src/polyglot_lc/segment/glossary.py) |
| Cây tóm tắt | [segment/summary_tree.py](src/polyglot_lc/segment/summary_tree.py) |
| **Prompt caching** | [agents/context.py](src/polyglot_lc/agents/context.py) |
| Từng agent | [agents/](src/polyglot_lc/agents/) |
| Điều phối LangGraph | [orchestrator/graph.py](src/polyglot_lc/orchestrator/graph.py), [router.py](src/polyglot_lc/orchestrator/router.py) |
| Gọi API + metric | [llm/client.py](src/polyglot_lc/llm/client.py) |
| **Retry & phân loại lỗi** | [llm/resilience.py](src/polyglot_lc/llm/resilience.py) |
| Kinh nghiệm tích lũy | [reasoning_bank/](src/polyglot_lc/reasoning_bank/) |
| Tự tối ưu prompt | [skillopt/](src/polyglot_lc/skillopt/) |
| Bộ nhớ dịch | [memory/tm.py](src/polyglot_lc/memory/tm.py) |

---

## Phụ lục — Lịch sử tài liệu

| Tài liệu | Dành cho |
| --- | --- |
| **tomtat.md** (file này) | Hiểu toàn bộ dự án từ con số 0 |
| [README.md](README.md) | Cài đặt, tham khảo nhanh CLI |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Sơ đồ chi tiết, quyết định thiết kế |