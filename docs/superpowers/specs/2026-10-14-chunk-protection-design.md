# Spec: bảo vệ chunk tới khi dịch xong — không bỏ chunk, không để note

- Ngày: 2026-10-14
- Trạng thái: đã thực hiện trong cùng phiên (người dùng đã chốt trước khi code); tự soát lại theo
  checklist spec
- Phạm vi: `src/polyglot_lc/pipeline/coordinator.py`, `src/polyglot_lc/orchestrator/graph.py`,
  `src/polyglot_lc/orchestrator/state.py`, `src/polyglot_lc/ui/simple.py`,
  `src/polyglot_lc/config.py`, `src/polyglot_lc/llm/resilience.py`, `src/polyglot_lc/cli.py`,
  `webui/static/app.js`, `tests/`
- Sub-project: **D** (chất lượng/độ bền của lượt dịch) — hành vi khi nhà cung cấp lỗi

## 1. Mục tiêu

Một lượt dịch phải **bảo vệ bản dịch tới khi hoàn thiện**: không chunk nào bị bỏ qua, và file kết quả
của một lượt đã xong **không** chứa dấu `<!-- chunk-NNN: untranslated … -->`. Hiện nay lỗi nhà cung
cấp ở một chunk (429, timeout…) làm chunk đó bị bỏ rơi và file cuối mang marker "untranslated" — một
phần tài liệu mất chữ nhưng lượt chạy vẫn kết thúc "Completed".

Sau khi xong, ba câu phải trả lời được **chỉ từ kết quả chạy**:

1. **Chunk có được dịch không?** — file `.md` có đủ bản dịch của mọi chunk, không marker nào.
2. **Đã trả tiền cho cùng một câu mấy lần?** — số lần `Translator.translate` được gọi cho một chunk
   là **một**, kể cả khi lần chạy đầu hỏng ở Critic/Verifier.
3. **Đang chờ hay treo?** — dòng in ra, sự kiện `retry_wait` trong log, và caption dưới thanh tiến độ
   nói đang chờ chunk nào / hỏng ở đâu / còn bao lâu.

## 2. Quyết định đã chốt với người dùng

| Quyết định | Chốt |
|---|---|
| Trần số lần thử lại | **Không trần**: chunk được chạy lại cho tới khi có bản dịch |
| Khi nào lượt chạy kết thúc | Chỉ khi **mọi chunk đã xong**; lối ra là Ctrl-C (CLI) hoặc nút Dừng (web UI) |
| Bản nháp đã sinh ra | **Được giữ lại**, không vứt đi khi lỗi xảy ra ở bước sau |
| Note trong file kết quả | **Không còn** ở lượt đã xong; marker chỉ còn là lưới an toàn cho file viết dở |

## 3. Sự thật đã kiểm trong repo (trước khi sửa)

| Điểm | Hành vi cũ | Nơi |
|---|---|---|
| Lỗi nhà cung cấp ở một chunk | Bị khối `except LLMError`, chunk bị bỏ rơi, `run_one` trả dict rỗng | `coordinator.py::_run_chunks.run_one` |
| Bản dịch đã có mà chỉ Critic/Verifier hỏng | Bị vứt cùng cả lần chạy của chunk | cùng chỗ (chỉ `out` sống sót khi `invoke` **không** ném) |
| Dịch rỗng | 2 lần thử thêm rồi bỏ, **không** ghi checkpoint | `coordinator.py::translate_chunk` |
| Marker trong file | `<!-- chunk-NNN: untranslated (…) -->` theo từng block mất | `assembler.py:204` |
| Trạng thái cuối của lượt | `failed`/`exit_code=1` khi `result.errors` khác rỗng — kể cả lỗi đã được xử lý xong | `coordinator.py::_record_run_end` |
| Trạng thái từng bước | `SimpleStep.__exit__` gọi `add_error` cho **mọi** bước hỏng, kể cả bước được chạy lại thành công | `ui/simple.py` |

Hai hệ quả của hàng cuối + hàng áp chót: một lỗi tạm thời rồi qua vẫn làm lượt chạy bị báo `failed`,
nên "bảo vệ chunk" phải đi kèm việc **trạng thái nói về file**, nếu không thì file đúng mà lượt chạy
vẫn bị đọc là hỏng.

## 4. Thiết kế

### 4.1 Vòng chạy lại cấp chunk (không giới hạn)

`_run_chunks` (`coordinator.py:415`) bọc mỗi chunk trong vòng lặp:

- `chunk_graph.invoke` ném `LLMError` → ghi `ui.add_error`, in/ném sự kiện chờ, **chờ rồi chạy lại
  chunk đó**; không trả về kết quả rỗng, không bỏ chunk.
- Bản dịch rỗng cũng là "chưa xong": chờ rồi hỏi lại, thay cho `max_empty_retries = 2`.
- Nhịp chờ: `jittered_delay` của tầng resilience (đã có test), lũy thừa từ
  `CHUNK_RETRY_BASE_DELAY` (5s) tới trần `CHUNK_RETRY_MAX_DELAY` (120s), jitter 0.2 — các chunk cùng
  hỏng trong một đợt outage không quay lại cùng một giây.
- `RetryStats.add_chunk_retry()` đếm số lần chạy lại, `run_end` ghi `chunk_retries`.
- **Cắt được**: chờ theo từng lát `_RETRY_TICK = 0.5s` và kiểm `self._stop`; `_run_chunks` bật cờ này
  khi bất kỳ ngoại lệ nào thoát ra khỏi vòng `as_completed`, để `ThreadPoolExecutor` join được ngay
  thay vì đợi một vòng lặp không trần.
- Lỗi **không phải** `LLMError` (bug, tài liệu hỏng) vẫn dừng lượt chạy như trước: chỉ lỗi nhà cung
  cấp mới đáng để chờ. Test `test_a_crashed_run_resumes_the_chunks_it_finished` giữ nguyên hành vi này.

### 4.2 Bản nháp đã sinh ra được giữ lại

- `GraphContext` (`graph.py:114-127`) giữ `drafts[chunk_id]` (bản nháp mới nhất) và
  `current_steps[chunk_id]` (bước đang chạy), dưới `resume_lock`.
- `translator_node` nhận `kept_draft` và **không gọi model** khi đã có bản nháp; `refiner_node` và
  `retranslate_node` cũng ghi lại bản nháp mới nhất, nên lần chạy sau vào thẳng bước hỏng.
- `kept_draft` **phải** khai trong `ChunkState` (`state.py:43`): LangGraph bỏ mọi khoá không có trong
  state — đây là lỗi đã gặp khi chạy test đầu tiên (Translator bị gọi lại lần hai).
- Khi dùng lại bản nháp, `translator_node` **không** ghi lại `decision` fast-path: cùng một bản nháp
  thì cổng đã chấm rồi, ghi lại là đếm một lần từ chối thành hai trong báo cáo.

### 4.3 Nhìn thấy đang chờ

- `SimpleUI.retry(step, chunk_id, message)` (`simple.py:144`) in ra và phát một sự kiện
  `status="retry"`, `type="retry_wait"` mang bước đã hỏng (`ctx.last_step`) và lý do.
- Web UI: `chunkStates` đã hiểu `retry` là "đang chạy" nên chunk không nằm đỏ trong lúc chờ;
  `progressWord` (`app.js:1235`) thêm nhánh caption `Đang chờ thử lại: <bước> · <chunk> — <lý do>`;
  `logEntry` in dòng log như mọi sự kiện khác (CSS `data-status="retry"` đã có sẵn).
- Sự kiện `retry` cũng vào `problem_count` và danh sách problems của server: **cái giá** của lượt chạy
  nằm trong log, còn **kết quả** nằm ở trạng thái cuối.

### 4.4 File kết quả và trạng thái lượt

- `assemble` giữ nguyên nhánh marker (`assembler.py:204`) như **lưới an toàn** cho người gọi trực tiếp
  và cho file viết dở (`PARTIAL_OUTPUT=1` có thể chứa `pending`/`untranslated` trong lúc chạy).
  Một lượt đã xong không còn đi qua nhánh đó: mọi chunk đều có bản dịch, nên marker biến mất khi file
  được ghi lần cuối.
- `_record_run_end` (`coordinator.py:275`): `failed` khi **còn chunk trống hoặc lượt chết**
  (`result.fatal_error`), không phải khi *có* lỗi. `detail`/CLI nói cùng một điều: `Failed` cho lượt
  không giao đủ, `Completed with errors` khi mọi chunk đã dịch nhưng đường đi có lỗi đã xử lý,
  `Completed` khi sạch.
- Checkpoint: chỉ ghi khi chunk đã có bản dịch (nhánh `break` ở trên), nên `--resume` vẫn là lưới an
  toàn chứ không phải đường chính.

### 4.5 Cấu hình

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `CHUNK_RETRY_BASE_DELAY` | `5` | Chờ lần đầu trước khi chạy lại chunk; lũy thừa từ đây |
| `CHUNK_RETRY_MAX_DELAY` | `120` | Trần chờ giữa hai lần chạy lại |

Có trong `.env.example`, `README.md` (bảng biến), và test `tests/test_config.py`.

## 5. Kiểm thử

| Test | Khẳng định |
|---|---|
| `test_a_rate_limited_chunk_is_waited_out_not_left_behind` | 5 lần 429 rồi thành công → chunk được chạy lại **6** lần, file không có `untranslated`, `run_end.status == "done"`, `exit_code == 0`, `chunk_retries == 5`, 5 sự kiện `retry_wait` mang đúng chunk/bước/lý do |
| `test_a_chunk_whose_critic_failed_keeps_the_draft_it_paid_for` | Critic hỏng phát đầu → Translator vẫn chỉ được gọi **đúng một lần** cho mỗi chunk |
| `test_a_waiting_chunk_is_visible_on_both_surfaces` | `SimpleUI.retry` in ra và phát event `retry`/`retry_wait`, `problem_count` tăng |
| `test_a_run_waiting_out_a_provider_says_so_under_the_bar` | app.js có nhánh caption theo `status === "retry"` và CSS cho dòng log `retry` |
| `test_chunk_retry_wait_has_a_default_and_is_env_overridable`, `test_a_negative_chunk_retry_wait_is_rejected` | Hai biến mới có mặc định, đọc được từ env, giá trị `0` bị từ chối |
| `test_an_interrupted_run_is_not_recorded_as_a_finished_one` | Ctrl-C (không có chunk nào hỏng) vẫn ghi `run_end` là `failed`/`exit_code=1` |
| `tests/test_cli.py`: `test_a_clean_run_says_completed`, `test_a_provider_failure_the_run_waited_out_is_not_a_failed_translation`, `test_the_command_says_failed_when_a_chunk_has_no_translation` | Ba câu kết luận của CLI khớp với quy tắc ở §4.4 — đọc từ output thật của `polyglot translate`, không phải từ `PipelineResult` |

Bộ test hiện tại: **568 passed** (`pytest tests/ -q`), `ruff check` + `ruff format --check` sạch,
`node --check webui/static/app.js` sạch.

## 6. Ngoài phạm vi

- **Dừng giữa lượt**: API chưa có "dừng mềm" cho CLI ngoài Ctrl-C; web UI vẫn dừng bằng tín hiệu tiến
  trình (đã có trước spec này).
- **Tiếp tục đúng bước đã hỏng trong cùng lần chạy** (giữ cả `critique`, `verify_result`): bản nháp là
  thứ đắt nhất và đã được giữ; Critic/Verifier sẽ chạy lại cho chunk đó — chấp nhận, và ghi lại ở §7.
- **`resumed`** trong `_from_checkpoint` vẫn không phải khoá của `ChunkState` (không ai đọc), không sửa
  trong spec này.

## 7. Rủi ro đã biết

- **Chờ không trần**: một lượt CLI không người trông nom có thể chờ rất lâu khi khoá sai (401 bị hiểu
  là lỗi tạm). Đây là lựa chọn của người dùng; lối ra là Ctrl-C / nút Dừng, và cờ `_stop` được kiểm
  mỗi 0.5s nên Ctrl-C không phải đợi hết nhịp chờ.
- **Chạy lại cả chunk**: bản nháp sống sót nhưng `refine_round`/`retranslate_count` đặt lại từ 0, nên
  một chunk chạy lại có thể tốn thêm một lượt Critic/Refine/Verifier. Bounded bởi cấu hình hiện có.
- **`ui.errors` dài** khi outage kéo dài (một dòng mỗi lần hỏng). CLI chỉ in 5 dòng cuối, log giữ tất
  cả; nếu về sau thấy ồn, nên gộp theo chunk thay vì theo lần thử.
- **`problem_count` tăng theo số lần chờ** — trung thực nhưng làm "số vấn đề" của một lượt outage dài
  trông lớn.
