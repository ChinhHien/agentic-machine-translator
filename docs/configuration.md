# Cấu hình polyglot-lc

Mọi thứ điều chỉnh được đều là biến môi trường. Sao chép [`.env.example`](../.env.example)
thành `.env` rồi điền **bốn nhóm endpoint**; các biến còn lại trong bản mẫu đã ở dạng
comment, bỏ comment dòng nào là đổi đúng biến đó.

Thứ tự ưu tiên: biến môi trường của tiến trình → `.env` → mặc định trong mã.
`load_dotenv()` không ghi đè biến đã có, nên `LLM_MOCK=1 polyglot translate ...`
luôn thắng `.env`.

## 1. Bắt buộc: bốn nhóm endpoint

- **Planner** — `PLANNER_BASE_URL`, `PLANNER_MODEL`, `PLANNER_API_KEY`: chạy một lần ở đầu pipeline (chiến lược, thuật ngữ).
- **Nhóm đọc-nguồn** — `SOURCE_GROUP_BASE_URL`, `SOURCE_GROUP_MODEL`, `SOURCE_GROUP_API_KEY`: Translator · Critic · Refiner dùng chung một provider/model để bộ nhớ đệm tiền tố phục vụ cả ba. Đặt riêng `TRANSLATOR_*`/`CRITIC_*`/`REFINER_*` vẫn thắng.
- **Verifier** — `VERIFIER_BASE_URL`, `VERIFIER_MODEL`, `VERIFIER_API_KEY`: độc lập, chỉ đọc bản dịch cuối.
- **Embedding** — `EMBED_BASE_URL`, `EMBED_MODEL`, `EMBED_API_KEY`: Reasoning Bank.

Thiếu `BASE_URL`/`MODEL` (khi không đặt `LLM_MOCK=1`) thì pipeline dừng ngay và nêu
đúng tên biến còn thiếu.

**Dự phòng**: `LLM_BACKUP_BASE_URL` + `LLM_BACKUP_MODEL` (+ `LLM_BACKUP_API_KEY`) dùng
chung cho mọi agent, chỉ sau khi endpoint chính cạn lượt thử lại. Cần cả hai biến;
trùng đúng cặp `base_url` + model với endpoint chính thì bị coi như **không có** dự
phòng. Không có dự phòng thì lượt chạy dừng với lý do rõ ràng, không quay vòng vô hạn.

## 2. Toàn bộ biến

Định dạng bảng: `| tên | mặc định trong mã | ý nghĩa |`.

### 2.1 Theo từng agent

`<NAME>` là `PLANNER`, `TRANSLATOR`, `CRITIC`, `REFINER` hoặc `VERIFIER`; bảng dưới
viết cho Planner, bốn agent còn lại có đúng bộ biến đó.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PLANNER_BASE_URL` | (bắt buộc) | Endpoint OpenAI-compatible. |
| `PLANNER_MODEL` | (bắt buộc) | Planner chạy một lần nên model nhỏ là đủ. |
| `PLANNER_API_KEY` | (trống) | Ollama/vLLM nhận giá trị bất kỳ. |
| `PLANNER_TEMPERATURE` | `0.3` | Độ ngẫu nhiên. |
| `PLANNER_TIMEOUT` | `300` | Trần thời gian một request (giây), thay cho `LLM_TIMEOUT`. |
| `TRANSLATOR_BASE_URL` | (nhóm đọc-nguồn) | Ghi đè nhóm cho riêng Translator. |
| `TRANSLATOR_MODEL` | (nhóm đọc-nguồn) | Model dịch; là một phần danh tính checkpoint. |
| `TRANSLATOR_API_KEY` | (nhóm đọc-nguồn) | Ghi đè khoá cho riêng Translator. |
| `TRANSLATOR_TEMPERATURE` | `0.3` | Hạ thấp cho tài liệu kỹ thuật. |
| `TRANSLATOR_TIMEOUT` | `300` | Trần thời gian một request dịch (giây). |
| `CRITIC_BASE_URL` | (nhóm đọc-nguồn) | Ghi đè nhóm cho riêng Critic. |
| `CRITIC_MODEL` | (nhóm đọc-nguồn) | Model chấm điểm draft. |
| `CRITIC_API_KEY` | (nhóm đọc-nguồn) | Ghi đè khoá cho riêng Critic. |
| `CRITIC_TEMPERATURE` | `0.3` | Độ ngẫu nhiên khi chấm điểm. |
| `CRITIC_TIMEOUT` | `300` | Trần thời gian một request chấm điểm (giây). |
| `REFINER_BASE_URL` | (nhóm đọc-nguồn) | Ghi đè nhóm cho riêng Refiner. |
| `REFINER_MODEL` | (nhóm đọc-nguồn) | Model sửa draft theo critique. |
| `REFINER_API_KEY` | (nhóm đọc-nguồn) | Ghi đè khoá cho riêng Refiner. |
| `REFINER_TEMPERATURE` | `0.3` | Độ ngẫu nhiên khi sửa. |
| `REFINER_TIMEOUT` | `300` | Trần thời gian một request sửa (giây). |
| `VERIFIER_BASE_URL` | (bắt buộc) | Endpoint Verifier, độc lập với nhóm đọc-nguồn. |
| `VERIFIER_MODEL` | (bắt buộc) | Model kiểm tra ngữ nghĩa cuối. |
| `VERIFIER_API_KEY` | (trống) | Khoá của Verifier. |
| `VERIFIER_TEMPERATURE` | `0.0` | Áp dụng cả khi kiểm định bản vá thuật ngữ. |
| `VERIFIER_TIMEOUT` | `300` | Trần thời gian một request kiểm tra (giây). |

### 2.2 Nhóm, embedding, dự phòng, timeout

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `SOURCE_GROUP_BASE_URL` | (bắt buộc) | Endpoint dùng chung cho Translator/Critic/Refiner. |
| `SOURCE_GROUP_MODEL` | (bắt buộc) | Model dùng chung cho ba agent đó. |
| `SOURCE_GROUP_API_KEY` | (trống) | Khoá dùng chung cho ba agent đó. |
| `SOURCE_GROUP_TEMPERATURE` | `0.3` | Đặt riêng từng agent vẫn thắng. |
| `EMBED_BASE_URL` | (bắt buộc) | Endpoint embedding của Reasoning Bank. |
| `EMBED_MODEL` | (bắt buộc) | Đổi model làm bank tự reset collection. |
| `EMBED_API_KEY` | (trống) | Khoá cho endpoint embedding. |
| `EMBED_TIMEOUT` | `300` | Trần thời gian một request embedding (giây). |
| `LLM_BACKUP_BASE_URL` | (trống) | Endpoint dự phòng chung; phải đi cùng `LLM_BACKUP_MODEL`. |
| `LLM_BACKUP_MODEL` | (trống) | Cùng gateway nhưng khác model vẫn là backup hợp lệ. |
| `LLM_BACKUP_API_KEY` | (trống) | Khoá của endpoint dự phòng. |
| `LLM_TIMEOUT` | `300` | Trần một request cho mọi agent/embedding không đặt `_TIMEOUT` riêng. Call không stream nên đây là hạn chót cho cả quá trình nghĩ + trả lời. |
| `LLM_MAX_RETRIES` | `3` | Số lần thử lại trước khi chuyển sang dự phòng. |
| `LLM_RETRY_BASE_DELAY` | `2` | Chờ ban đầu giữa các lần thử (giây). |
| `LLM_RETRY_MAX_DELAY` | `60` | Trần của khoảng chờ đó (giây). |
| `LLM_RETRY_BUFFER_SECONDS` | `2` | Cộng thêm khi provider có gợi ý thời gian chờ. |
| `LLM_RETRY_JITTER` | `0.2` | Tỉ lệ ngẫu nhiên để nhiều worker không thử lại cùng lúc. |
| `LLM_RETRY_CONCURRENCY_WAIT` | `10` | Chờ tối thiểu khi bị 429 "request khác đang giữ slot". |
| `LLM_RETRY_SLOT_REBUILD_AFTER` | `3` | Số lần 429-slot liên tiếp trước khi dựng lại connection pool. |
| `LLM_PROVIDER_CONCURRENCY` | `1` | Số request tối đa đang chạy trên mỗi endpoint. |
| `LLM_PROVIDER_CONCURRENCY_MAX` | `0` | Trần cho concurrency tự nới; `0` là giữ cố định. |

### 2.3 Ngân sách output

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `TRANSLATOR_MAX_TOKENS` | `16384` | Ngân sách cho **lần hỏi lại** sau khi reply bị cắt; lần đầu không đặt trần. Thang leo gấp đôi tới `32768`. |
| `TRANSLATOR_REASONING` | `auto` | `off` gửi hint `thinking.type=disabled` trên mọi lần hỏi; gateway không hiểu sẽ bỏ qua. |
| `VERIFIER_MAX_TOKENS` | `4096` | Ngân sách một verdict; retry xin gấp đôi (tối đa `32768`). Model reasoning tính cả token suy nghĩ vào trần này. |
| `VERIFIER_REASONING` | `auto` | Như `TRANSLATOR_REASONING`, cho Verifier. |

### 2.4 Chia đoạn, song song, chất lượng

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `CHUNK_MAX_TOKENS` | `700` | Ngân sách token mỗi chunk. |
| `CHUNK_OVERLAP_TOKENS` | `80` | Token chồng lấn giữa hai chunk. |
| `MAX_CONCURRENCY` | `1` | Số chunk dịch song song; đi cùng `LLM_PROVIDER_CONCURRENCY`. |
| `CRITIC_THRESHOLD` | `7` | Điểm tối thiểu để draft không bị trả lại. |
| `REFINE_MIN_SCORE` | `7` | Draft đạt mức này thì bỏ qua Refiner. |
| `REFINE_ROUNDS` | `2` | Số vòng sửa tối đa. |
| `RB_TOP_K` | `4` | Số trace Reasoning Bank đưa vào prompt. |
| `ADAPTIVE_QUALITY` | `1` | `0` là tắt đường tắt thích ứng. |
| `FAST_PATH_MAX_COMPLEXITY` | `0.65` | Chunk đơn giản hơn mức này mới đi đường tắt. |
| `FAST_PATH_MIN_SCORE` | `9` | Điểm draft tối thiểu cho đường tắt. |
| `FAST_PATH_SKIP_LLM_VERIFY` | `0` | `1` là bỏ cả Verifier trên chunk đường tắt. |
| `CONSISTENCY_CHECK` | `1` | Bật lượt vá thuật ngữ xuyên chunk. |
| `CONSISTENCY_MAX_PATCHES` | `5` | Trần số lần vá mỗi tài liệu; `0` là tắt. |

### 2.5 Chờ và chạy lại một chunk

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `CHUNK_RETRY_BASE_DELAY` | `5` | Chờ ban đầu trước khi chạy lại chunk lỗi (giây). |
| `CHUNK_RETRY_MAX_DELAY` | `120` | Trần khoảng chờ đó (giây). |
| `CHUNK_MAX_ATTEMPTS` | `6` | Số lần thử của một chunk trước khi bỏ cho lượt này. |
| `CHUNK_STOP_AFTER_FAILURES` | `3` | Số chunk liên tiếp hết endpoint dự phòng thì coi là sự cố và dừng lượt; dưới ngưỡng đó chunk chỉ thành lỗ hổng (exit 2). |
| `PREFLIGHT` | `true` | Gọi thử một câu qua Planner/Translator/Verifier trước khi dịch, để sai khoá hay sai model lộ ra trong vài giây thay vì ở chunk cuối. |
| `VERIFY_UNAVAILABLE_RETRIES` | `2` | Số lần hỏi lại Verifier trên chính draft khi verdict không tới. |
| `VERIFY_RETRY_DELAY` | `5` | Chờ giữa các lần đó (giây). |
| `COMPLETION_ROUNDS` | `2` | Số vòng quét các chunk còn thiếu checkpoint. |
| `COMPLETION_RETRY_DELAY` | `20` | Chờ trước mỗi vòng quét (giây). |
| `COMPLETION_DEADLINE_SECONDS` | `900` | Hạn chót cho toàn bộ các vòng quét; `0` là tắt. |

### 2.6 Thư mục, đầu ra, log

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `REASONING_BANK_PATH` | `./data/reasoning_bank` | Nơi lưu ChromaDB của Reasoning Bank. |
| `SKILLS_DIR` | `./skills` | Nơi lưu `skill.md` và `history.jsonl`. |
| `SKILL_APPLY` | `1` | `0` là lượt đối chứng: không bơm skill, không promote. |
| `SKILL_MIN_SAMPLES` | `20` | Số chunk tối thiểu trước khi được promote. |
| `SKILL_MIN_GAIN` | `0.2` | Mức vượt tối thiểu so với skill đang dùng. |
| `LOGS_DIR` | `./logs` | Nơi ghi log JSONL. |
| `LOG_JSONL` | `1` | `0` là không ghi trace JSONL. |
| `CHECKPOINT_DIR` | `./data/checkpoints` | Checkpoint theo chunk; `off` là tắt hẳn. Danh tính checkpoint gồm file + nội dung, cặp ngôn ngữ, model và base URL của Translator, hash chunk, glossary, cờ fast-path và skill — đổi endpoint hay skill sẽ không phát lại bản dịch cũ. |
| `RESUME` | `0` | `1` là dùng lại checkpoint của lượt bị gián đoạn. |
| `OUTPUT_FORMAT` | `md` | `md`, `pdf` hoặc `both`. |
| `PDF_FONT_PATH` | (trống) | Font Unicode để in bản dịch vào PDF; trống là tự dò. |
| `PARTIAL_OUTPUT` | `1` | Ghi file ngay khi từng chunk xong. |
| `INGEST_AI_REVIEW` | `0` | Sửa block trích xuất nghi lỗi bằng AI; chỉ gửi từng block. |
| `INGEST_AI_REVIEW_MAX_BLOCKS` | `20` | Trần số block được gửi duyệt mỗi tài liệu. |
| `INGEST_AI_REVIEW_MAX_CHARS` | `2000` | Block dài hơn bị bỏ qua, không cắt. |
| `PROMPT_CACHE` | `auto` | `auto`, `explicit` (gửi marker kiểu Anthropic) hoặc `off`. |
| `LLM_MOCK` | `0` | `1` chạy toàn bộ pipeline bằng mock, không cần key. |
| `LLM_VERBOSE` | `0` | `1` log chi tiết từng request. |
| `LLM_VERBOSE_RESILIENCE` | `0` | `1` log chi tiết quyết định thử lại/fallback. |

### 2.7 Chỉ dành cho web UI

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PYTHON` | (tự dò) | Interpreter chạy tiến trình con của web UI. |
| `RUN_ID` | (tự sinh) | Mã lượt chạy mà tiến trình con ghi vào trace. |

Web UI còn tự đặt cho tiến trình con: `LOGS_DIR`, `LOG_JSONL=1`,
`PYTHONIOENCODING=utf-8` (bắt buộc trên Windows) và `LLM_MOCK=1` cho lượt mock.

## 3. Công thức sẵn dùng

- **Chạy thử không cần khoá**: `LLM_MOCK=1 polyglot translate --file input/handbook.pdf`.
- **Endpoint chỉ cho một request cùng lúc**: `LLM_PROVIDER_CONCURRENCY=1` và `MAX_CONCURRENCY=1`. Hai tầng độc lập: một cái là hàng đợi provider, một cái là số chunk song song.
- **Model reasoning**: token suy nghĩ tính vào cùng ngân sách output và cả timeout. Đặt `TRANSLATOR_REASONING=off`/`VERIFIER_REASONING=off` nếu gateway hiểu hint, hoặc nâng trần và `LLM_TIMEOUT` cho đủ.
- **Tiết kiệm token**: giữ `ADAPTIVE_QUALITY=1`; nâng `FAST_PATH_MIN_SCORE` để ít chunk đi đường tắt hơn.
- **Dọn `.env` cũ đang đóng băng mặc định**:

  ```bash
  grep -oE '^[A-Z][A-Z0-9_]*' .env | sort > /tmp/co.txt   # tên biến đang đặt
  ```

  Xoá hoặc comment những dòng trùng giá trị mặc định trong bảng trên — chúng không
  đổi hành vi, chỉ che mặc định tương lai. Xoá cả dòng `[TEMPLATE]` nếu có.
