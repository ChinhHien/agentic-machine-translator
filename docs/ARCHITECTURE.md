# polyglot-lc — Kiến trúc hệ thống

Tài liệu này giải thích: một lượt dịch chạy qua pipeline như thế nào, mỗi bước
do agent/model nào đảm nhiệm, vì sao chọn công cụ đó, và các cơ chế quan trọng
(caching, concurrency, retry, token accounting, các thành phần best-effort)
hoạt động ra sao.

---

## 1. Sơ đồ tổng quan

```
                          File gốc (.md / .txt / .pdf)
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │  Ingest (registry)        │  PDF → Markdown (MarkItDown)
                        │  md / txt / pdf          │  → chuẩn hoá NFC + che code
                        │                          │  placeholder ⟦CODE_n⟧
                        └────────────┬─────────────┘
                                     │  Document gồm các Block có cấu trúc
                                     ▼
                        ┌──────────────────────────┐
                        │  Segment (chunker)        │  chia theo tiktoken
                        │  tiktoken                │  tại ranh giới heading/đoạn
                        └────────────┬─────────────┘  + overlap, section_path
                                     │  danh sách Chunk
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
             ┌──────────────┐                ┌─────────────────┐
             │ Glossary     │                │ Summary tree    │  LLM (Planner)
             │ (regex/heur.)│                │ section + doc   │  KHÔNG gọi theo
             └──────┬───────┘                └────────┬────────┘  từng chunk
                    │                                 │
                    └───────────────┬─────────────────┘
                                    ▼
                          ┌───────────────────┐
                          │ Planner (1 lần)   │  provider/model PLANNER_*
                          │ strategy/few-shot │  + tra Reasoning Bank
                          └─────────┬─────────┘
                                    │  plan: strategy, few_shot, pitfalls
                                    ▼
   ╔═══════════════════════════════════════════════════════════════════════╗
   ║  NHÓM ĐỌC NGUỒN (READ-SOURCE) — dùng chung 1 provider/model (SOURCE_GROUP_*)
   ║                                                                        ║
   ║      ┌───────────────────────────────────────────────┐                ║
   ║      │  PREFIX ỔN ĐỊNH — cacheable (byte giống hệt)   │                ║
   ║      │  vai trò + cặp ngôn ngữ + strategy +           │                ║
   ║      │  pitfalls + glossary + CHUNK GỐC               │                ║
   ║      └───────────────┬───────────────────────────────┘                ║
   ║          ┌───────────┼───────────┐                                     ║
   ║          ▼           ▼           ▼                                     ║
   ║     Translator    Critic      Refiner                                  ║
   ║     (+động:       (+động:     (+động:                                  ║
   ║      few-shot/TM)  bản nháp)   bản nháp + nhận xét)                    ║
   ║          └───────────┼───────────┘                                     ║
   ╚══════════════════════╪═════════════════════════════════════════════════╝
                          │  bản nháp cuối của chunk
                          ▼
                 ┌────────────────────┐
                 │ Verifier           │  provider/model VERIFIER_*
                 │ kiểm tra cứng trước│  (chỉ đọc chunk + bản dịch, không đọc cả file)
                 │ rồi mới gọi LLM    │
                 └─────────┬──────────┘
                           │
                fail ──────┤ (tối đa 1 lần dịch lại kèm lý do)
                           │ pass
                           ▼
                 save_trace → Reasoning Bank (ChromaDB)
                           ▼
                 chunk kế ↺ / assemble → restore_code → ghi file kết quả
                           ▼
                 cập nhật SkillOpt (skills/<src>-<tgt>/<domain>.json)
```

Điểm mấu chốt: các chunk **độc lập** với nhau, nên phần thân
`Translator → Critic → Refiner → Verifier` của **từng chunk** có thể chạy song
song. Planner, cây tóm tắt, assemble và SkillOpt luôn chạy tuần tự vì phụ thuộc toàn tài liệu.

---

## 2. Từng bước xử lý (kèm lý do chọn công cụ)

| # | Bước | Module | Việc xảy ra | Gọi LLM? | Vì sao dùng công cụ này |
|---|------|--------|-------------|----------|--------------------------|
| 1 | Ingest | `ingest/` | PDF được **MarkItDown** chuyển thành Markdown, rồi parse bằng **đúng đường đi** của `.md`/`.txt`: chuẩn hoá NFC, che code thành `⟦CODE_n⟧`, tạo `Document` gồm các `Block` có kiểu | không | MarkItDown cho Markdown có cấu trúc (heading/list) nên các bước sau không phụ thuộc định dạng gốc; `markdown-it-py` parse; vẫn giữ bộ đọc pypdf→pdfplumber làm dự phòng |
| 2 | Nhận diện ngôn ngữ | `pipeline._detect_lang` | `langdetect` khi `--src auto` | không | rẻ, ổn định |
| 3 | Chia đoạn | `segment/chunker.py` | Cắt thành các chunk ≤ `CHUNK_MAX_TOKENS`, tại ranh giới đoạn/heading, có overlap | không | tiktoken khớp tokenizer của model Translator nên ngân sách là token thật, không phải đoán theo ký tự |
| 4 | Glossary | `segment/glossary.py` | Heuristic regex: từ viết tắt, cụm viết hoa, thuật ngữ lặp ≥2 lần; cộng CSV người dùng | không | miễn phí, tất định; giúp thuật ngữ dịch nhất quán |
| 5 | Tóm tắt (section + doc) | `segment/summary_tree.py` | 1 lần gọi Planner cho mỗi **section** + 1 lần cho **toàn tài liệu** | Planner | cho Planner cái nhìn tổng quan gọn. Tóm tắt **từng chunk bị bỏ mặc định** (không dùng ở hạ nguồn) để tiết kiệm 1 request/chunk |
| 6 | Lập kế hoạch | `agents/planner.py` | 1 strategy + few-shot + pitfalls cho cả tài liệu | Planner | gọi 1 lần ở cấp tài liệu; đồng thời tra Reasoning Bank để tham chiếu các lượt dịch cũ tương tự |
| 7 | Dịch | `agents/translator.py` | Mỗi chunk, theo plan + glossary + TM | nhóm đọc nguồn | dịch trung thành, giữ code/định dạng |
| 8 | Phê bình | `agents/critic.py` | Cho điểm bản nháp 0–10 kèm issues/suggestions | nhóm đọc nguồn | cổng chất lượng có cấu trúc; quyết định có refine hay không |
| 9 | Tinh chỉnh | `agents/refiner.py` | Viết lại bản nháp theo nhận xét (≤ `REFINE_ROUNDS`) | nhóm đọc nguồn | thu hẹp khoảng cách chất lượng trước khi kiểm tra |
| 10 | Kiểm tra | `agents/verifier.py` | **Kiểm tra cứng tất định trước** (placeholder, cấp heading), rồi mới hỏi LLM | Verifier | kiểm tra cứng rẻ, bắt lỗi cấu trúc *trước khi* tốn 1 lần gọi LLM |
| 11 | Dịch lại | `orchestrator/graph.py` | Khi kiểm tra fail, dịch lại 1 lần kèm lý do | nhóm đọc nguồn | tự sửa có giới hạn |
| 12 | Lưu trace | `reasoning_bank/` | Lưu trace thành công (điểm ≥ 0.75) vào ChromaDB | không (chỉ embed) | tái dùng chiến lược tốt cho tài liệu sau |
| 13 | Ghép kết quả | `pipeline.py` | `restore_code`, nối chunk theo đúng thứ tự, ghi file | không | — |
| 14 | SkillOpt | `skillopt/optimizer.py` | Thu thập (prompt, output, score); đề xuất "textual gradient"; chỉ promote nếu điểm trung bình gần đây tốt hơn | Planner | cải thiện chỉ dẫn cho Translator qua nhiều lần chạy |

---

## 3. Định tuyến model/provider (kiến trúc Hybrid)

| Nhóm | Agent | Cấu hình | Có đọc nguồn gốc? |
|------|-------|----------|-------------------|
| Planner | `PLANNER` | `PLANNER_*` | tóm tắt doc + section (không đọc raw) |
| **Đọc nguồn** | `TRANSLATOR`, `CRITIC`, `REFINER` | `SOURCE_GROUP_*` (giá trị riêng `*_*` sẽ ghi đè) | có — cùng một chunk gốc, qua prefix cache dùng chung |
| Verifier | `VERIFIER` | `VERIFIER_*` | chỉ chunk đang kiểm tra + bản dịch (đã cắt ngắn), không đọc cả file |
| Embedding | Reasoning Bank | `EMBED_*` | không áp dụng |

**Cơ chế:** ba agent nhóm đọc nguồn liên tục đối chiếu
`chunk gốc → bản nháp → nhận xét`. Vì vậy chúng được gom về **một** provider/model
để provider có thể cache phần prefix chung. Planner và Verifier tách
riêng vì Planner chạy 1 lần, Verifier chạy ở cuối — dùng chung cache không đem
lại lợi ích mà lại làm ràng buộc cấu hình.

Nếu một agent có biến riêng (`TRANSLATOR_*`…) thì giá trị riêng thắng; nếu không
thì lấy từ `SOURCE_GROUP_*`.

---

## 4. Concurrency, retry và adaptive refiner

### 4.1 Concurrency có giới hạn (`MAX_CONCURRENCY`, mặc định 2)

- Mỗi chunk được xử lý bằng **một đồ thị riêng** (`make_chunk_graph`), giữ
  nguyên thứ tự nội bộ `Translator → Critic → Refiner → Verifier`.
- Chỉ **các chunk khác nhau** mới chạy song song, thông qua
  `ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)`.
- **Đảm bảo thứ tự output:** kết quả được ghi vào mảng theo **chỉ số chunk gốc**,
  nên hoàn thành sớm/muộn không đổi thứ tự. Bước `assemble` vẫn sắp theo `chunk.id`.
- Planner, cây tóm tắt, assemble, SkillOpt luôn **tuần tự**.
- `MAX_CONCURRENCY=1` ⇒ chạy tuần tự hoàn toàn.
- Nếu một chunk ném lỗi: lỗi được **lan truyền** (giống hành vi tuần tự cũ) và
  **không ghi file output dở dang**.

> Vì sao dùng thread: LLM SDK hiện tại là **đồng bộ** (blocking I/O), nên thread
> pool là primitive đủ dùng, không cần async/framework mới. `TranslationMemory`,
> `RetryStats`, `TokenUsage`, `EventSink` và bộ đếm bước CLI đều đã được bọc lock.

### 4.2 Retry: tách rõ lỗi hạ tầng vs lỗi ngữ nghĩa

**Lỗi hạ tầng** (transport/provider) — được retry có giới hạn:
`429`, `408`, timeout, connection error, `500/502/503/504`.
Cơ chế trong `llm/resilience.py`:
1. Nếu provider trả `Retry-After` / "try again in 47.4s" → **ưu tiên dùng hint**
   (cộng thêm `LLM_RETRY_BUFFER_SECONDS`, chặn trần bởi `LLM_RETRY_MAX_DELAY`).
2. Nếu không có hint → exponential backoff `base * 2^n`, **có jitter** giới hạn
   (`LLM_RETRY_JITTER`, mặc định 0.2) để tránh dồn dập; jitter chỉ nhân hệ số
   trong `[1-jitter, 1]` nên **không bao giờ vượt quá** giá trị exponential.
3. Số lần retry bị chặn bởi `LLM_MAX_RETRIES` (đây chính là `MAX_INFRA_RETRIES`);
   hết retry thì chuyển sang endpoint backup (`<NAME>_BACKUP_*`) nếu có, ngược lại ném lỗi.
4. Lỗi vĩnh viễn (`400/401/403/404/422`...) **không retry, không fallback**.

**Lỗi ngữ nghĩa** (Verifier FAIL) — hoàn toàn tách biệt: đi theo đường
`retranslate → verify` **tối đa 1 lần** (`after_verifier(..., max_retries=1)`).
Hai loại retry không trộn vào nhau.

> Cơ chế đếm: retry không tạo thêm "request" trong thống kê. `TokenUsage.add`
> chỉ chạy **một lần trên mỗi lần gọi `chat` thành công**; `RetryStats` đếm riêng
> số lần retry/fallback.

### 4.3 Adaptive refiner (`REFINE_MIN_SCORE`)

Router `after_critic` quyết định đi tiếp:

```
nếu refine_round >= REFINE_ROUNDS        → verify
nếu score >= REFINE_MIN_SCORE            → verify   (bỏ qua Refiner)
nếu score <  CRITIC_THRESHOLD            → refine
ngược lại                                 → verify
```

- Mặc định `REFINE_MIN_SCORE = CRITIC_THRESHOLD = 7.0` ⇒ **hành vi không đổi**.
- Nâng `REFINE_MIN_SCORE` để bỏ qua Refiner với các bản nháp "đã đủ tốt"
  (ví dụ `CRITIC_THRESHOLD=8.5`, `REFINE_MIN_SCORE=7.0` ⇒ điểm 8.0 sẽ bỏ qua refine).
- Critic và Verifier **luôn chạy** — không bỏ qua cổng chất lượng.

### 4.4 Các thành phần best-effort (không được làm hỏng bản dịch)

- **Reasoning Bank**: `retrieve`/`upsert` được bọc try/except + lock. Lỗi truy
  vấn → tiếp tục **không có trace**; lỗi ghi → `stored=False`. Không có queue/service mới.
- **SkillOpt**: bọc try/except; lỗi tối ưu chỉ ghi cảnh báo, `skill_promoted=False`,
  bản dịch vẫn hoàn tất.
- **Verifier**: kiểm tra cứng chạy **trước** mọi lần gọi LLM. Nếu cấu trúc đã sai
  ⇒ trả về ngay, **0 lần gọi LLM Verifier**.

---

## 5. Prompt/context caching

### 5.1 Ranh giới cache

Ba agent nhóm đọc nguồn nhận **prefix ổn định giống hệt nhau** (cùng chunk ⇒ byte
giống hệt). Phần động đặt **sau** prefix:

```
[ SYSTEM — ỔN ĐỊNH, CACHEABLE ]
  You are a member of an expert {src}->{tgt} translation team.
  Global translation strategy: ...
  Pitfalls to avoid: ...
  Glossary (use consistently): ...
  --- SOURCE TEXT START ---
  <chunk gốc>
  --- SOURCE TEXT END ---

[ USER — ĐỘNG, KHÔNG CACHE ]
  chỉ dẫn riêng của agent + few-shot / bản nháp / nhận xét / TM matches
```

Vì sao đặt thứ tự này: provider cache theo **prefix chung dài nhất**. Chỉ cần
phần đầu (system) giống hệt giữa Translator/Critic/Refiner thì phần lớn token
(nguồn gốc + glossary + strategy) được phục vụ từ cache; bản nháp/nhận xét thay
đổi nên phải nằm ngoài.

### 5.2 Chế độ `PROMPT_CACHE`

- `auto` (mặc định): không gửi marker; sắp xếp request theo hướng "ổn định trước"
  để provider tự cache prefix (OpenAI, DeepSeek, …).
- `explicit`: gắn thêm breakpoint kiểu Anthropic
  `cache_control: {"type": "ephemeral"}` lên message system ổn định, cho gateway
  yêu cầu marker tường minh.
- `off`: không gửi marker (vẫn đọc metric nếu provider trả về).

### 5.3 Trạng thái cache HIT / MISS / UNKNOWN

`HIT`/`MISS` chỉ được suy ra khi provider **thực sự trả metadata cache**:

| Trường hợp | Kết luận |
|------------|----------|
| Provider trả metadata và `cached > 0` | `HIT` |
| Provider trả metadata và `cached == 0` | `MISS` |
| Provider **không** trả trường cache | `UNKNOWN` |

> Quy tắc quan trọng: **không** được coi `cached == 0` là MISS nếu provider đơn
> giản là không cung cấp metadata — đó là `UNKNOWN`. Các định dạng được nhận
> diện: OpenAI `prompt_tokens_details.cached_tokens`, DeepSeek
> `prompt_cache_hit_tokens`, Anthropic `cache_read_input_tokens` /
> `cache_creation_input_tokens`.

### 5.4 Fingerprint prefix (chỉ để debug)

`stable_prefix_fingerprint` băm `phiên bản + model + nội dung message system ổn
định` (tức đã bao hàm cặp ngôn ngữ, strategy, pitfalls, glossary, chunk gốc).
Chỉ **log ở mức debug** dạng `chunk=... cache_fingerprint=... model=...`;
**không bao giờ log văn bản nguồn thô**, chỉ log digest.

---

## 6. Token accounting

`TokenUsage` cộng dồn **thread-safe**, theo toàn cục, theo agent và theo
provider/model:

```
Input:           123.456
Cached input:     98.000
Uncached input:   25.456
Output:           12.345
Total:           135.801
Requests:              8
Cache (per request):
  hits:   3
  misses: 2
  unknown: 3
```

Quy tắc:

- `Cached input` là **tập con** của `Input`; `Uncached input = Input − Cached`.
- Chỉ số theo provider/model chỉ in khi dùng **nhiều hơn một** model.
- Khi provider **không trả usage**, token được **ước lượng** (~ký tự/4) và số
  request ước lượng được báo rõ (`estimated usage for N request(s)`) — **không tự
  bịa số liệu** và không gán nhãn chúng như số thật.
- Retry không làm tăng số request (xem 4.2).

---

## 7. Giới hạn & điều không được kỳ vọng

- Hiệu quả cache **phụ thuộc provider/runtime** và chưa được xác minh nếu gateway
  không trả metadata (khi đó mọi request là `UNKNOWN`).
- PDF scan/ảnh không có văn bản vẫn không trích được (giới hạn cố hữu).
- Không có batching, queue, DB, dashboard, worker nền hay router model động —
  đúng theo phạm vi dự án.
