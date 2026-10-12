# Spec: trace cấu trúc cho một lượt dịch

- Ngày: 2026-10-12
- Trạng thái: đã tự soát (từ vựng, ràng buộc, ví dụ, tiêu chí); **chờ duyệt spec**
- Phạm vi: `src/polyglot_lc/` (ui, orchestrator, pipeline, cli) và một dòng nhãn nhỏ trong `webui/`
- Sub-project: **F** (observability) trong bảy việc đã tách; danh sách bảy việc ở mục 1

## 1. Mục tiêu

Một lượt dịch phải **tự giải thích được sau khi đã chạy xong**: nó đã được bảo làm gì, mỗi khối đi
đường nào và vì sao, bước nào tốn bao lâu bao nhiêu token, và khi hỏng thì hỏng vì loại nguyên nhân
gì. Hiện nay không có câu trả lời nào trong số đó nằm trên đĩa.

Cụ thể, người vận hành phải trả lời được, chỉ từ file của chính lượt chạy đó:

1. **Vì sao khối này tốn bốn request?** — quyết định `fast_path`, `refine_gate`, `retranslate` đã
   được tính trong code nhưng bị vứt đi ngay sau khi dùng (mục 3).
2. **Lượt chạy này đã được bảo làm gì?** — model, provider, ngôn ngữ, lĩnh vực, cờ, và **hash của
   tài liệu**, để lần sau chạy lại cùng một tài liệu là so sánh được.
3. **Hỏng vì cái gì?** — hết quota, timeout, sai key, provider trả rác, hay tài liệu không đọc
   được: bốn thứ này cần bốn cách xử lý khác nhau, mà hiện tại tất cả chỉ là một chuỗi `error`.
4. **Chunk nào là chunk lỗi, và tốn bao nhiêu?** — số liệu theo từng khối, không chỉ tổng cả lượt.

Đây là sub-project **F** (observability) trong bảy việc đã tách; nó đứng trước vì mọi việc sau
(cost theo chunk, cảnh báo PDF, điểm eval của SkillOpt, tỷ lệ dùng lại của TM) đều phải đọc số liệu
từ đây. Danh sách bảy việc: verifier toàn khối (A), PDF reconstruction (B), TM bằng embedding (C),
SkillOpt an toàn (D), cost/latency (E), **trace (F)**, giới hạn an toàn (G).

## 2. Ba quyết định đã chốt với người dùng

| Quyết định | Chốt |
|---|---|
| Bản ghi đi đâu và ai đọc | **Một file JSONL mỗi lượt, giàu hơn file hiện tại.** Đọc bởi lệnh `polyglot trace`, báo cáo cuối lượt, và web UI. Không backend, không OTLP, **không thêm dependency** |
| Bản ghi chứa gì | **Chỉ metadata.** Id, hash, độ dài, số token, thời lượng, quyết định + lý do, nguyên nhân lỗi. **Không bao giờ chứa văn bản tài liệu**, kể cả bản dịch |
| Tái hiện lỗi tới đâu | **Ghi đủ để so sánh, và một lệnh giải thích.** Chưa làm replay (chạy lại từ trace), vì pipeline chưa có bảo đảm tất định: sampling của LLM, Reasoning Bank và checkpoint đều làm hai lượt khác nhau |

Điều ba quyết định này **loại trừ**, và không được lén thêm vào khi code: pricing theo tiền tệ (E),
xuất span OTLP, replay, bản ghi cho TM/Reasoning Bank (C), và mọi thay đổi giao diện ngoài một nhãn
nguyên nhân ở khối "Vấn đề" của web UI.

## 3. Ràng buộc đã xác minh trong repo

| # | Ràng buộc | Bằng chứng |
|---|---|---|
| T1 | `EventSink` mở `<LOGS_DIR>/run_<ts>.jsonl`, có lock, mỗi bản ghi một dòng JSON | [ui/events.py](../../../src/polyglot_lc/ui/events.py) |
| T2 | `Event` hiện có đúng `{ts, step, chunk_id, status, latency, error}`; `SimpleStep` phát `start` lúc vào và `ok`/`fail` lúc ra, kèm `latency` ms | [ui/simple.py](../../../src/polyglot_lc/ui/simple.py) |
| T3 | Quyết định routing **đã được tính kèm lý do** rồi bị vứt: `draft_signals` (score + issues) → `should_take_fast_path`; `after_critic`; `after_verifier` | [graph.py](../../../src/polyglot_lc/orchestrator/graph.py), [router.py](../../../src/polyglot_lc/orchestrator/router.py) |
| T4 | Lỗi provider **đã được phân loại**: `RateLimitError`, `TransientError`, `PermanentError`, `is_concurrency_limit`, `normalize_provider_error`, `parse_retry_seconds` | [llm/resilience.py](../../../src/polyglot_lc/llm/resilience.py) |
| T5 | `TokenUsage` là bộ đếm **cấp class** (toàn cục), có `snapshot()` / `snapshot_by_agent()` / `snapshot_by_model()`, gồm `calls`, `sent`, `received`, `cached`, `cache_hits`, `cache_misses`, `estimated_calls` | [llm/client.py](../../../src/polyglot_lc/llm/client.py) |
| T6 | `RetryStats.snapshot()` là nguồn duy nhất biết số lần retry/fallback thật sự xảy ra | `llm/resilience.py` |
| T7 | `PipelineResult` đã có chunks, routes (fast/standard/deep), retries, errors, output_path, pdf_path, pdf_blocks, log_path, elapsed_s | [pipeline/coordinator.py](../../../src/polyglot_lc/pipeline/coordinator.py) |
| T8 | `TranslationPipeline.__init__` dựng `EventSink(self.app.logs_dir, enabled=app.log_jsonl)`; `run()` có `finally` chạy cho cả đường thành công lẫn đường ném lỗi | `pipeline/coordinator.py` |
| T9 | **`webui/*.py` không được `import polyglot_lc`** (nếu không, `load_dotenv` của server sẽ đầu độc môi trường tiến trình con) | spec webui, quy tắc cứng 1 |
| T10 | Web UI tìm log bằng glob `run_*.jsonl` trong `logs/webui/<run_id>/`; lọc dòng JSONL hỏng khi đọc | [webui/store.py](../../../webui/store.py) |
| T11 | Web UI tự sinh `run_id` dạng `YYYYMMDD_HHMMSS_<6 hex>` và truyền `LOGS_DIR` cho con | [webui/runs.py](../../../webui/runs.py) |
| T12 | CLI là typer app, một lệnh `translate`; báo cáo cuối lượt in ra ở `translate` và đã in tổng route | [cli.py](../../../src/polyglot_lc/cli.py) |
| T13 | `logs_dir` mặc định là `./logs`, còn tiến trình con do web UI chạy nhận `LOGS_DIR=logs/webui/<run_id>` — tức trace của lượt từ web UI nằm **sâu một cấp** dưới `logs/` | [config.py:313](../../../src/polyglot_lc/config.py#L313), [webui/runs.py:202](../../../webui/runs.py#L202) |
| T14 | `Chunk` có `id`, `text`, `token_count`, `index` và phương thức `hash()` trả sha1 10 hex của `text` — đủ để định nghĩa `plan_hash` mà không cần thêm trường | [segment/chunker.py:37](../../../src/polyglot_lc/segment/chunker.py#L37) |
| T15 | Cổng kiểm tra của repo: `pytest`, `ruff check .`, `ruff format --check .` (ruff 0.16 cũng định dạng khối python trong `.md`) | `README.md`, plan 2026-10-11 |

## 4. Kiến trúc

Một nguồn ghi (tiến trình con của pipeline), một file append-only mỗi lượt, ba đầu đọc. Không dựng
module writer mới: `EventSink` đã giữ file và lock, nên nó nhận thêm `record()`; phần *từ vựng* và
phần *đọc* nằm ở một module mới `polyglot_lc/ui/trace.py`.

```
coordinator ── run_start ─┐
  chunk graph ── decision ─┤
    SimpleStep ── step_*   ├─→ <LOGS_DIR>/run_<ts>.jsonl ─→ polyglot trace <run>
  verifier/resilience ── error ─┘                        ─→ web UI (tiến độ + Vấn đề)
coordinator ── run_end ────┘                             ─→ báo cáo cuối lượt
```

Ba lý do của cách chia này:

- **Một file, không hai.** Bản ghi mới và bản ghi tiến độ cũ cùng nằm trong file mà web UI đã poll,
  nên không có hai nguồn sự thật để lệch nhau, và không phải đổi cách web UI tìm log (T10).
- **Tên file giữ nguyên `run_<ts>.jsonl`.** Đổi tên sẽ làm mất lịch sử của mọi lượt đã chạy trên đĩa
  (T10), trong khi `run_id` trong từng bản ghi đã đủ để tách hai lượt (mục 5).
- **Từ vựng ở `ui/trace.py`, không ở `ui/events.py`.** `events.py` là bản ghi tiến độ mà UI đang đọc;
  `trace.py` sở hữu ba thứ chỉ trace cần: hàm dựng bản ghi, **từ vựng nguyên nhân lỗi**, và đầu đọc
  dùng cho `polyglot trace`. `webui/` giữ đầu đọc riêng của nó, vì T9 cấm nó import package.

## 5. Từ vựng bản ghi

Sáu loại, bảy tên (vì `step_start`/`step_end` là một cặp). Mọi bản ghi **giữ nguyên các khoá cũ**
(`ts, step, chunk_id, status, latency, error`) và thêm `run_id`; các trường mới đều tuỳ chọn, nên
`Event` cũ, web UI và test hiện tại chạy y nguyên.

| Loại | Trường | Ghi ở đâu |
|---|---|---|
| `run_start` | `file`, `sha256`, `bytes`, `pages`, `chunks`, `plan_hash`, `src`, `tgt`, `domain`, `agents` (danh sách `{name, model, host, temperature}`), `max_tokens`, `flags` (`resume`, `format`, `partial_output`), `version` | coordinator, ngay sau khi đọc tài liệu và chốt kế hoạch chia khối |
| `step_start` / `step_end` | `name`, `chunk_id`, `duration_ms`, `tokens` = `{sent, received, cached}` của riêng bước, `status` | `SimpleStep.__enter__` / `__exit__` |
| `decision` | `chunk_id`, `kind` ∈ `fast_path` \| `refine_gate` \| `retranslate` \| `verify_mode`, `chosen`, `reasons[]`, `numbers{}` | translator node, hai cạnh router, verifier node (T3) |
| `error` | `step`, `chunk_id`, `cause` (từ vựng đóng ở mục 6), `message` | nơi lỗi được bắt: `SimpleStep.__exit__`, `VerifierAgent`, `call_with_resilience` |
| `run_end` | `status`, `exit_code`, `duration_ms`, `tokens` tổng, `calls`, `cache_hits`/`cache_misses`, `retries`, `fallbacks`, `chunks_done`, `chunks_failed`, `artifacts` (`md`/`pdf` + size), `rebuild` (`placed`, `kept`, `failures`), `problems` | coordinator, trong `finally` (T8) |
| `run_note` | `text` | cho ghi chú không thuộc loại nào ở trên (ví dụ "Reasoning Bank không dùng được"), thay vì nhét vào `error` |

Ví dụ ba dòng đầu và một dòng cuối của một lượt thật (đã rút gọn, không có văn bản tài liệu):

```json
{"ts": 1791744000.12, "run_id": "20261012_015104_dad6b2", "type": "run_start", "file": "silk-dg.pdf", "sha256": "9f2c…", "bytes": 7189364, "pages": 412, "chunks": 27, "plan_hash": "3ab1…", "src": "en", "tgt": "vi", "domain": "general", "agents": [{"name": "translator", "model": "gpt-4o", "host": "api.example.com", "temperature": 0.2}, {"name": "verifier", "model": "gpt-4o-mini", "host": "api.example.com", "temperature": 0.0}], "max_tokens": 700, "flags": {"resume": false, "format": "md"}, "version": "0.1.0"}
{"ts": 1791744001.50, "run_id": "20261012_015104_dad6b2", "type": "step_end", "step": "translate", "chunk_id": "chunk-003", "status": "ok", "latency": 1840.2, "duration_ms": 1840, "tokens": {"sent": 1180, "received": 640, "cached": 0}}
{"ts": 1791744001.51, "run_id": "20261012_015104_dad6b2", "type": "decision", "kind": "fast_path", "chunk_id": "chunk-003", "chosen": false, "reasons": ["complexity 0.62 > 0.35", "draft issues: length ratio 0.18"], "numbers": {"complexity": 0.62, "max_complexity": 0.35, "draft_score": 0.41, "min_score": 0.75}}
{"ts": 1791744120.77, "run_id": "20261012_015104_dad6b2", "type": "run_end", "status": "done", "exit_code": 0, "duration_ms": 120650, "tokens": {"sent": 28400, "received": 15120, "cached": 18200}, "calls": 61, "cache_hits": 14, "cache_misses": 47, "retries": 2, "fallbacks": 0, "chunks_done": 27, "chunks_failed": 0, "artifacts": [{"kind": "md", "path": "...", "bytes": 113000}], "rebuild": {"placed": 0, "kept": 0, "failures": []}, "problems": 0}
```

Năm chi tiết của từ vựng, để không phải quyết lại lúc code:

- **`plan_hash` là hash của kế hoạch chia khối, không phải của tài liệu.** `sha256` của chuỗi nối
  `"{chunk.id}:{chunk.hash()}"` theo đúng thứ tự khối, lấy 12 hex đầu — dùng lại `Chunk.hash()` đã
  có (sha1 của `chunk.text`, 10 hex), nên không phát minh thêm gì. `sha256` trả lời "cùng tài liệu
  chưa", `plan_hash` trả lời "cùng cách chia chưa": đổi `CHUNK_MAX_TOKENS` cho ra `plan_hash` khác
  dù `sha256` y nguyên, và không có nó thì hai lượt khác cách chia trông y hệt nhau vì `chunk-NNN`
  vẫn đánh số từ đầu.
- **`pages` là số trang khi nguồn là PDF, `null` với md/txt.** Ghi `null` chứ không bỏ khoá, để đầu
  đọc không phải phân biệt "không có trang" với "chưa hỏi".
- **`agents` chỉ chứa `host`, không chứa URL đầy đủ hay key.** `host` lấy từ
  `urlparse(base_url).hostname` của `AgentConfig` ([config.py:98](../../../src/polyglot_lc/config.py#L98)),
  nên phần đường dẫn và mọi thứ trước `@` bị bỏ lại. `temperature` là trường thật của `AgentConfig`,
  không phải trường mới.

- **`reasons` là chữ người đọc được, sinh ra từ số liệu thật** — "complexity 0.62 > 0.35", không phải
  "fast path bị từ chối". `numbers` đi kèm để máy lọc được mà không phải parse chữ.
- **`decision` không thay `step_*`.** Một quyết định là *lý do* đằng sau một bước, không phải một
  bước; chúng nằm cạnh nhau, không lồng nhau.
- **`tokens` của một bước là xấp xỉ khi chạy song song.** `TokenUsage` là bộ đếm toàn cục (T5), nên
  khi `MAX_CONCURRENCY > 1` hai thread có thể xen vào giữa hai lần `snapshot()`. Điều này được ghi
  **ngay trong docstring của hàm dựng bản ghi** và trong tài liệu lệnh `trace`, để không ai đọc con
  số đó như số chính xác. Số chính xác tuyệt đối theo bước cần thread-local attribution — việc của
  sub-project E, không phải ở đây.

## 6. Từ vựng nguyên nhân lỗi

`cause` là một tập đóng, ánh xạ từ những gì T4 đã phân loại sẵn — không đoán lại từ chuỗi:

| `cause` | Nguồn |
|---|---|
| `rate_limit` | `RateLimitError` |
| `concurrency` | `TransientError`/`RateLimitError` khi `is_concurrency_limit()` đúng |
| `timeout` | `TransientError` (timeout / lỗi kết nối / 408/409/425/5xx) |
| `auth` | `PermanentError` với status 401/403 |
| `provider` | `PermanentError` còn lại (400/404/422, phản hồi không đọc được) |
| `parse` | `chat_json` không parse được JSON theo yêu cầu |
| `verify` | verifier trả `ok=false` sau khi đã hết lượt retranslate |
| `placeholder` | marker `⟦CODE_n⟧` bị mất và được vá lại |
| `config` | `ConfigError` (thiếu key, thiếu model, giá trị sai) |
| `input` | tài liệu không đọc được (PDF không có lớp chữ, file biến mất) |
| `unknown` | còn lại — và luôn kèm `message` nguyên văn để không mất thông tin |

`unknown` tồn tại là có chủ ý: một nguyên nhân bịa ra còn tệ hơn một nguyên nhân thừa nhận là chưa
biết. Test khoá lại rằng mọi `cause` xuất hiện trong log đều thuộc tập này.

## 7. Danh tính lượt chạy

`run_id` trở thành một thứ pipeline tự biết, thay vì chỉ web UI biết:

- `polyglot translate --run-id <id>`; mặc định tự sinh theo đúng dạng web UI đang dùng
  (`YYYYMMDD_HHMMSS_<6 hex>`), để hai bên không sinh ra hai id khác nhau cho cùng một lượt.
- Web UI truyền `RUN_ID` cho tiến trình con, cạnh `LOGS_DIR` nó đã truyền (T11). `run.json` của web
  UI và `run_id` trong trace vì thế là **một giá trị**.
- Mọi bản ghi đều mang `run_id`, nên hai lượt dùng chung một `LOGS_DIR` vẫn tách được khi đọc — điều
  mà tên file `run_<ts>.jsonl` (giây) không bảo đảm.

## 8. Ba đầu đọc

**`polyglot trace <run-id|path> [--json]`** — lệnh mới. Không có đối số thì **quét đệ quy
`logs/`** (cha của `LOGS_DIR`) và lấy `run_*.jsonl` mới nhất, vì lượt từ web UI nằm sâu một cấp ở
`logs/webui/<run_id>/` (T11) và chỉ quét `LOGS_DIR` sẽ không thấy chúng. Truyền `run-id` thì khớp
**tên thư mục** trước, rồi mới khớp tên file; `--log-dir` để chỉ định gốc khác.

Lệnh in ra: phần đầu (cài đặt đã chạy + hash tài liệu), dòng thời gian theo từng khối (đường
đã đi, lý do, thời lượng từng bước, token từng bước), các vấn đề kèm `cause`, và tổng kết. `--json`
in đúng cấu trúc đó ở dạng máy đọc. Đầu đọc nằm ở `ui/trace.py` và **bỏ qua dòng JSONL hỏng** theo
đúng cách web UI đang làm (T10).

**Báo cáo cuối lượt** — báo cáo hiện tại ở `cli.py` đã in tổng route; nó thêm phần *lý do*: ví dụ
bao nhiêu khối bị đẩy khỏi fast path và vì lý do gì, để lượt chạy tự giải thích mà không cần mở file.

**Web UI** — khối "Vấn đề" đang hiển thị step + chunk + message; nó thêm nhãn `cause` (một từ, đã
dịch), và thẻ tiến độ thêm dòng "route: fast/standard/deep". Không thêm trang, không đổi bố cục.
Server chỉ chuyển tiếp `cause` trong payload `problems` — nó đọc JSONL bằng đầu đọc riêng của mình
(T9), nên thay đổi ở đây là ba dòng.

## 9. Xử lý lỗi và biên

| Tình huống | Hành vi |
|---|---|
| Lượt chết trước bản ghi đầu tiên | Không có `run_start`: `trace` nói rõ điều đó và in những gì có, thay vì dựng một bảng rỗng |
| Lượt chết giữa lúc ghi | Dòng cuối có thể cụt: đầu đọc bỏ qua dòng hỏng, giữ phần còn lại, và nói ra số dòng bị bỏ |
| Hai lượt chung `LOGS_DIR` | Tách bằng `run_id`; không dựa vào tên file |
| `LOG_JSONL=0` | Không có file trace; `trace` báo không có file, `run_end` không được ghi (không có chỗ ghi) — cùng cách web UI đã xử lý |
| File trace quá lớn | Metadata-only: ước tính dưới 1 MB cho một cuốn vài trăm khối (mỗi khối ~8 bản ghi × ~180 B). Không xoay vòng file, không nén |
| Bí mật lọt vào trace | `run_start` ghi **host** của provider, không ghi URL đầy đủ, không ghi giá trị biến môi trường, không ghi header. Một test canary đặt giá trị giả vào môi trường và khẳng định nó không xuất hiện trong file |
| Trace ghi ra ngoài `logs/` | Không: file luôn nằm trong `LOGS_DIR` mà pipeline đã đọc từ cấu hình |

## 10. Kiểm thử

Tất cả ở `tests/`, không cần API key (mock mode), theo C5 của spec webui.

**`tests/test_trace.py`** (mới)
- một lượt mock sinh đủ từ vựng: `run_start` → `step_*` → `decision` → `run_end`, và **mỗi dòng parse
  được thành JSON**;
- `run_start` mang đúng `sha256` của file đầu vào (tính lại trong test) và số khối của kế hoạch;
- khối đi fast path ghi `decision kind=fast_path chosen=true` kèm `numbers.complexity` và
  `numbers.draft_score`; khối bị đẩy sang đường đầy đủ ghi `chosen=false` **kèm lý do đọc được**;
- `refine_gate` ghi đúng số đã quyết định (`critic_score`, `threshold`, `refine_round`);
- mọi `cause` trong file thuộc tập đóng ở mục 6;
- **canary**: giá trị của một biến môi trường bí mật đặt trong test không xuất hiện trong file;
- **không có văn bản tài liệu**: với một tài liệu có câu đánh dấu riêng, câu đó không xuất hiện
  trong trace (khoá lại quyết định "metadata only");
- dòng JSONL cụt ở cuối file không làm đầu đọc nổ và không làm mất các bản ghi trước đó.

**`tests/test_trace_cli.py`** (mới)
- `polyglot trace <path>` in ra cài đặt, quyết định, vấn đề kèm `cause`, và tổng kết;
- trên một lượt chết trước bản ghi đầu, nó nói rõ và thoát 0 (đọc được một lượt hỏng là việc bình
  thường, không phải lỗi của lệnh);
- `--json` in ra JSON parse được và **ổn định**: chạy hai lần trên cùng file cho cùng kết quả.

**`tests/test_ui_events.py`** (mở rộng, không phá test cũ)
- `step_end` có `duration_ms` và `tokens`, và các khoá cũ vẫn còn nguyên;
- `SimpleStep` phát `cause` khi bước ném `RateLimitError`/`PermanentError`.

**`tests/test_agents.py`**, **`tests/test_pipeline.py`** — không đổi hành vi: verifier, routing và
pipeline vẫn xanh; `decision` chỉ **thêm** bản ghi, không thay đổi quyết định nào.

**`tests/test_webui_server.py`** — payload `problems` có thêm `cause`; phần còn lại của hợp đồng web
UI không đổi (đây là chỗ dễ vỡ nhất, vì web UI đọc chính file đó).

**Cổng kiểm tra:** `python -m pytest`, `python -m ruff check .`, `python -m ruff format --check .`.

## 11. Tiêu chí nghiệm thu

1. Chạy một lượt mock rồi `polyglot trace` được: đọc ra cài đặt đã chạy (kèm hash tài liệu), từng
   khối đi đường nào **và vì sao**, vấn đề kèm nguyên nhân, tổng token/request/cache/retry.
2. Cùng lệnh đó trả lời được "khối này tốn mấy request" mà không cần mở file log bằng tay.
3. Một lượt bị rate limit ghi `cause=rate_limit`, không phải một chuỗi tự do; một khối bị verifier
   từ chối ghi `cause=verify` kèm lý do của verifier.
4. Grep file trace của một lượt thật **không** tìm thấy câu văn nào của tài liệu, và không tìm thấy
   giá trị biến môi trường nào (có test).
5. Web UI vẫn hiển thị đúng lượt đang chạy và lịch sử y như trước, với nhãn nguyên nhân mới trong
   khối "Vấn đề".
6. `ruff check`, `ruff format --check` và toàn bộ suite xanh.

## 12. Rủi ro đã biết

- **Token theo bước là xấp xỉ khi chạy song song** (mục 5). Nói thẳng ra ở cả docstring lẫn tài liệu
  lệnh, thay vì để người đọc tin đó là số chính xác.
- **File trace là bản ghi thứ hai về lượt chạy**, cạnh `run.json` của web UI. `run_id` chung giữ hai
  bên khớp nhau, nhưng hai nơi cùng mô tả một lượt là một chỗ có thể lệch — chấp nhận, vì gộp chúng
  lại là việc của sub-project khác.
- **`decision` thêm một bản ghi mỗi khối mỗi lần rẽ nhánh**, nên lượt nhiều khối sẽ có file dài hơn
  khoảng 30%. Vẫn dưới 1 MB; nếu vượt, việc cắt bớt thuộc về người đọc, không phải người ghi.
- **Lệnh `trace` chỉ giải thích, không chạy lại** — người dùng vẫn phải tự chạy lại để so sánh, và
  kết quả có thể khác vì sampling. Đây là giới hạn đã biết của quyết định ở mục 2, không phải thiếu
  sót của bản này.

## 13. Việc để sau (không thuộc bản này)

Pricing theo tiền tệ và cost theo provider (E); bản ghi cho TM/Reasoning Bank, tỷ lệ dùng lại (C);
điểm eval và so sánh trước khi promote (D); xuất OTLP; replay có diff (đã bàn, cố tình hoãn).
