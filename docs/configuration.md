# Cấu hình polyglot-lc

Mọi thứ điều chỉnh được đều nằm trong biến môi trường. Bản mẫu đầy đủ là
[`.env.example`](../.env.example): sao chép thành `.env` rồi điền.

**Thứ tự ưu tiên**: biến môi trường của tiến trình → `.env` → mặc định trong mã.
`load_dotenv()` chạy lúc import và **không ghi đè** biến đã có, nên
`LLM_MOCK=1 polyglot translate ...` luôn thắng `.env`. Vì vậy bản mẫu để mọi biến
tinh chỉnh ở dạng **comment**: không bỏ comment thì mặc định trong mã được dùng,
kể cả khi mặc định đó thay đổi về sau.

## 1. Bốn nhóm endpoint (phần duy nhất bắt buộc)

| Nhóm | Biến | Dùng cho |
|---|---|---|
| Planner | `PLANNER_BASE_URL`, `PLANNER_MODEL`, `PLANNER_API_KEY` | Một lần ở đầu pipeline: chiến lược, thuật ngữ, ghi chú từng phần |
| Nhóm đọc-nguồn | `SOURCE_GROUP_BASE_URL`, `SOURCE_GROUP_MODEL`, `SOURCE_GROUP_API_KEY`, `SOURCE_GROUP_TEMPERATURE` | Translator · Critic · Refiner dùng **chung** một provider/model |
| Verifier | `VERIFIER_BASE_URL`, `VERIFIER_MODEL`, `VERIFIER_API_KEY` | Chỉ đọc bản dịch cuối + tiêu chí |
| Embedding | `EMBED_BASE_URL`, `EMBED_MODEL`, `EMBED_API_KEY` | Reasoning Bank (truy hồi trace cũ) |

`<NAME>_BASE_URL` và `<NAME>_MODEL` là **bắt buộc** cho cả bốn nhóm (trừ khi
`LLM_MOCK=1`). Thiếu thì pipeline dừng ngay với thông báo nêu đúng tên biến còn
thiếu — và với Translator/Critic/Refiner còn gợi ý luôn biến `SOURCE_GROUP_*`
tương ứng.

### Vì sao Translator/Critic/Refiner dùng chung một nhóm

Cả ba đều **đọc lại cùng một chunk nguồn** và đối chiếu nó với plan/draft. Giữ
chúng trên cùng một provider/model giúp bộ nhớ đệm tiền tố của provider phục vụ
cùng một tiền tố ổn định (văn bản gốc + glossary + strategy) thay vì tính tiền
lại cho từng agent. Đặt nhóm một lần, và ghi đè riêng vẫn thắng: giá trị nào của
`TRANSLATOR_*`/`CRITIC_*`/`REFINER_*` được đặt thì thắng `SOURCE_GROUP_*` tương ứng.
Planner và Verifier **độc lập** với nhóm này.

### Endpoint dự phòng

`LLM_BACKUP_BASE_URL` + `LLM_BACKUP_MODEL` (+ `LLM_BACKUP_API_KEY`) chỉ dùng
sau khi endpoint chính đã cạn lượt thử lại. **Cần cả hai** biến base_url và
model — đặt lẻ một trong hai là lỗi cấu hình.

Dự phòng phủ **mọi agent của pipeline** — Planner, Translator, Critic, Refiner,
Verifier — và **trừ embedding**: Reasoning Bank là phần best-effort, thiếu dự
phòng ở đó mất một trace chứ không mất bản dịch. Một endpoint dự phòng, đặt
một lần cho cả pipeline: endpoint nào chết thì nhóm agent đang dùng nó rơi
sang endpoint này.

**Không có endpoint dự phòng thì lượt chạy dừng, không quay vòng vô hạn.** Khi
một agent đã cạn lượt thử lại mà không có nơi để chuyển sang, chunk đó được
hỏi **một lần** rồi lượt chạy kết thúc với lý do rõ ràng (mã thoát 1) — thay vì
chờ mãi trên một endpoint đã từ chối trả lời. Có endpoint dự phòng thì vòng
lặp cấp chunk vẫn chạy: mỗi vòng thử lại chính rồi dự phòng.

## 2. Toàn bộ biến

Định dạng bảng: `| tên | mặc định của mã | ý nghĩa và khi nào nên đổi |`.
Mặc định là giá trị **trong mã**; đó cũng là giá trị trong comment của bản mẫu.

### 2.1 Theo từng agent

`<NAME>` là một trong `PLANNER`, `TRANSLATOR`, `CRITIC`, `REFINER`, `VERIFIER`.
Bảng dưới viết cho Planner; bốn agent còn lại có đúng bộ biến đó với tiền tố của mình.

| Biến | Mặc định | Ý nghĩa và khi nào nên đổi |
|---|---|---|
| `PLANNER_BASE_URL` | (bắt buộc) | Endpoint OpenAI-compatible. Đổi khi muốn Planner chạy trên model rẻ/khác. |
| `PLANNER_MODEL` | (bắt buộc) | Model của Planner. Planner chạy một lần nên thường dùng model nhỏ là đủ. |
| `PLANNER_API_KEY` | (trống) | Khoá cho endpoint trên. Ollama/vLLM thường nhận giá trị bất kỳ. |
| `PLANNER_TEMPERATURE` | `0.3` | Độ ngẫu nhiên. Hạ xuống cho công việc cần ổn định; `0.0` cho Verifier là hợp lý. |
| `PLANNER_TIMEOUT` | `300` | Trần thời gian một request (giây), thay cho `LLM_TIMEOUT` cho riêng agent này. |
| `TRANSLATOR_BASE_URL` | (nhóm đọc-nguồn) | Ghi đè nhóm cho riêng Translator. |
| `TRANSLATOR_MODEL` | (nhóm đọc-nguồn) | Model dịch. Cũng là một phần danh tính checkpoint của lượt chạy. |
| `TRANSLATOR_API_KEY` | (nhóm đọc-nguồn) | Ghi đè khoá cho riêng Translator. |
| `TRANSLATOR_TEMPERATURE` | `0.3` | Độ ngẫu nhiên khi dịch. Thấp hơn cho tài liệu kỹ thuật. |
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
| `VERIFIER_BASE_URL` | (bắt buộc) | Endpoint của Verifier; nhóm riêng, không chung với nhóm đọc-nguồn. |
| `VERIFIER_MODEL` | (bắt buộc) | Model kiểm tra ngữ nghĩa cuối cùng. |
| `VERIFIER_API_KEY` | (trống) | Khoá của Verifier. |
| `VERIFIER_TEMPERATURE` | `0.0` | Độ ngẫu nhiên riêng của Verifier, áp dụng cả khi kiểm định ngữ nghĩa bản vá thuật ngữ. |
| `VERIFIER_TIMEOUT` | `300` | Trần thời gian một request kiểm tra (giây). |

### 2.2 Nhóm, embedding, và trần chung

| Biến | Mặc định | Ý nghĩa và khi nào nên đổi |
|---|---|---|
| `SOURCE_GROUP_BASE_URL` | (bắt buộc cho nhóm) | Endpoint dùng chung cho Translator/Critic/Refiner. |
| `SOURCE_GROUP_MODEL` | (bắt buộc cho nhóm) | Model dùng chung cho ba agent đó. |
| `SOURCE_GROUP_API_KEY` | (trống) | Khoá dùng chung cho ba agent đó. |
| `SOURCE_GROUP_TEMPERATURE` | `0.3` | Nhiệt độ dùng chung; đặt riêng từng agent vẫn thắng. |
| `LLM_BACKUP_BASE_URL` | (trống) | Endpoint dự phòng dùng chung cho **mọi** agent; phải đi cùng `LLM_BACKUP_MODEL`. |
| `LLM_BACKUP_MODEL` | (trống) | Model của endpoint dự phòng chung. |
| `LLM_BACKUP_API_KEY` | (trống) | Khoá của endpoint dự phòng chung. |
| `EMBED_BASE_URL` | (bắt buộc) | Endpoint embedding cho Reasoning Bank. |
| `EMBED_MODEL` | (bắt buộc) | Model embedding; đổi model làm Reasoning Bank tự reset collection. |
| `EMBED_API_KEY` | (trống) | Khoá cho endpoint embedding. |
| `EMBED_TIMEOUT` | `300` | Trần thời gian một request embedding (giây). |
| `LLM_TIMEOUT` | `300` | Trần thời gian một request cho mọi agent/embedding không đặt `_TIMEOUT` riêng. Call không stream, nên SDK read-timeout thực chất là hạn chót cho toàn bộ quá trình sinh câu trả lời: model reasoning nghĩ vài nghìn token rồi mới trả lời có thể mất hàng phút (median 47s, max 301s trong một lần chạy thật). Mặc định cũ 45s giết oan call khoẻ rồi fallback sang endpoint khác, trả tiền hai lần cho cùng một chunk. Request treo thật vẫn bị chặn bởi hard deadline trong client. |

### 2.3 Thư mục, đầu ra, checkpoint

| Biến | Mặc định | Ý nghĩa và khi nào nên đổi |
|---|---|---|
| `REASONING_BANK_PATH` | `./data/reasoning_bank` | Nơi lưu ChromaDB của Reasoning Bank. |
| `SKILLS_DIR` | `./skills` | Nơi lưu bản skill duy nhất của hệ thống (`skill.md`, `versions/`, `history.jsonl`). |
| `SKILL_APPLY` | `1` | `0` là lượt đối chứng: không bơm skill vào prompt và không promote — dùng để so cùng một tài liệu có/không skill. |
| `SKILL_MIN_SAMPLES` | `20` | Số chunk tối thiểu một lượt phải chấm điểm trước khi được promote; vài chunk không phải bằng chứng. |
| `SKILL_MIN_GAIN` | `0.2` | Mức vượt tối thiểu so với điểm đo của skill đang dùng; chênh lệch nhỏ hơn là nhiễu, không promote. |
| `LOGS_DIR` | `./logs` | Nơi ghi log JSONL của lượt chạy. |
| `CHECKPOINT_DIR` | `./data/checkpoints` | Checkpoint theo từng chunk. Đặt `off` để tắt hẳn. |
| `OUTPUT_FORMAT` | `md` | `md`, `pdf` (dựng lại PDF giữ định dạng) hoặc `both`. |
| `PDF_FONT_PATH` | (trống) | Font `.ttf` có dấu cho đầu ra PDF; trống thì tự dò Times/Arial/DejaVu. |
| `PARTIAL_OUTPUT` | `0` | `1` ghi file đầu ra ngay khi từng chunk xong, phần còn lại đánh dấu pending. |
| `RESUME` | `0` | `1` dùng lại checkpoint của lượt bị gián đoạn (CLI cũng có `--resume`). |

### 2.4 Chia đoạn và song song

| Biến | Mặc định | Ý nghĩa và khi nào nên đổi |
|---|---|---|
| `CHUNK_MAX_TOKENS` | `700` | Ngân sách token mỗi chunk. Tăng cho ngữ cảnh rộng hơn, giảm cho rẻ hơn. |
| `CHUNK_OVERLAP_TOKENS` | `80` | Token gối giữa hai chunk liền kề, để câu không bị cắt đôi. |
| `MAX_CONCURRENCY` | `1` | Số chunk dịch song song. `1` là tuần tự; trần 16. Phải đi kèm `LLM_PROVIDER_CONCURRENCY` tương ứng: nâng mỗi biến này thì mọi worker vẫn xếp hàng trên một slot provider, và pipeline sẽ cảnh báo về độ lệch đó. |

### 2.5 Chất lượng

| Biến | Mặc định | Ý nghĩa và khi nào nên đổi |
|---|---|---|
| `CRITIC_THRESHOLD` | `7.0` | Điểm tối thiểu để draft được coi là đạt (0-10). |
| `REFINE_ROUNDS` | `2` | Số vòng Refiner tối đa cho một chunk. |
| `REFINE_MIN_SCORE` | `7.0` | Draft đạt mức này thì bỏ qua Refiner, đi thẳng Verifier. |
| `RB_TOP_K` | `4` | Số trace lấy từ Reasoning Bank để đưa vào ngữ cảnh. |
| `VERIFIER_MAX_TOKENS` | `4096` | Ngân sách output của một lần Verifier. Provider reasoning tính cả token suy nghĩ vào trần này, nên verdict dài có thể cần nhiều hơn; lần retry xin gấp đôi (tối đa 32768) và mức đã học được dùng cho các chunk sau của cùng endpoint. Nếu cả hai lần vẫn bị cắt vì model suy nghĩ gần hết trần, lần thứ ba chỉ hỏi `{pass, reasons}` để câu trả lời vừa phần ngân sách còn lại. Trần cũ 8192 đã bị chạm trong một lần chạy thật và làm ba verdict phải rút xuống bản tối giản. |
| `VERIFIER_REASONING` | `auto` | `auto` để model reasoning suy nghĩ như mặc định; `off` gửi hint tắt suy nghĩ (`thinking.type = disabled`) để trần output dành cho câu trả lời — chỉ có tác dụng khi gateway hiểu hint đó. |
| `TRANSLATOR_MAX_TOKENS` | `16384` | Ngân sách output cho **lần hỏi lại** của Translator sau khi reply bị cắt ở trần provider. Lần hỏi đầu không đặt trần nào: mặc định của endpoint đã đủ cho model reasoning nghĩ ~6000 token rồi trả lời ~2600 (trung vị ~9000, cao nhất ~17000 trong một lần chạy thật). Giá trị cũ 4096 nhỏ hơn một nửa trung vị đó, nên lần hỏi lại bị cắt tiếp và chunk bị báo lỗi dù chính nó không có vấn đề. Thang leo gấp đôi tới trần cứng 32768; nếu gateway từ chối `max_tokens` quá hạn của nó (lỗi 400), hạ giá trị này. |

### 2.6 Đường tắt thích ứng

| Biến | Mặc định | Ý nghĩa và khi nào nên đổi |
|---|---|---|
| `ADAPTIVE_QUALITY` | `1` | `0` là mọi chunk đi đúng đường đầy đủ Translate-Critic-Refiner-Verify. |
| `FAST_PATH_MAX_COMPLEXITY` | `0.65` | Chunk phức tạp hơn mức này không được đi đường tắt (0-1). Đo trên run thật 609 chunk: complexity 0.60-0.79, nên 0.35 hay 0.50 không chunk nào lọt. |
| `FAST_PATH_MIN_SCORE` | `9.0` | Điểm tất định tối thiểu để một chunk được đi đường tắt (0-10). |
| `FAST_PATH_SKIP_LLM_VERIFY` | `0` | `1` bỏ luôn Verifier trả phí trên các chunk đường tắt. Tiết kiệm token, mất cổng ngữ nghĩa cuối. |

### 2.7 Nhất quán thuật ngữ

| Biến | Mặc định | Ý nghĩa và khi nào nên đổi |
|---|---|---|
| `CONSISTENCY_CHECK` | `1` | Kiểm tra thuật ngữ xuyên chunk sau khi dịch; chunk lệch được vá chứ không dịch lại. |
| `CONSISTENCY_MAX_PATCHES` | `5` | **Sàn** số bản vá mỗi lượt: thực tế vá `max(giá trị này, 30% số chunk lệch)`, và chunk lệch nhiều thuật ngữ nhất được vá trước. Mỗi lượt vá gọi `REFINER` trên một chunk; bản vá vượt kiểm tra độ dài, cấu trúc và thuật ngữ được kiểm định thêm một lượt bằng `VERIFIER`, đối chiếu nguồn và bản dịch trước vá. Chỉ nhận bản vá khi kiểm định đạt, nếu không giữ bản trước vá. `consistency_patch_budget` là ngân sách lượt vá, không bao gồm các request kiểm định bổ sung. Hai bản vá liền bị từ chối thì pass dừng, không đốt nốt ngân sách. `0` là tắt vá. |

### 2.8 Retry, timeout, hàng đợi

| Biến | Mặc định | Ý nghĩa và khi nào nên đổi |
|---|---|---|
| `LLM_MAX_RETRIES` | `3` | Trần số lần thử lại cho lỗi hạ tầng (429, 408, timeout, 5xx, lỗi kết nối). |
| `LLM_RETRY_BASE_DELAY` | `2` | Cơ sở của backoff lũy thừa: 2s, 4s, 8s… |
| `LLM_RETRY_MAX_DELAY` | `60` | Trần của một lần chờ giữa hai lần thử. |
| `LLM_RETRY_BUFFER_SECONDS` | `2` | Cộng thêm lên trên gợi ý `Retry-After` của provider. |
| `LLM_RETRY_JITTER` | `0.2` | Phần nhiễu cộng vào backoff để nhiều chunk không thử lại cùng lúc. |
| `LLM_PROVIDER_CONCURRENCY` | `1` | Trần request đồng thời cho **một** `base_url`. Gói miễn phí thường chỉ phục vụ 1. Nâng `MAX_CONCURRENCY` mà để biến này ở 1 thì mọi worker vẫn xếp hàng trên một slot. |
| `LLM_RETRY_CONCURRENCY_WAIT` | `10` | Chờ tối thiểu khi provider từ chối vì "còn request đang chạy" (429 concurrency). |
| `LLM_RETRY_SLOT_REBUILD_AFTER` | `3` | Số lần 429 "còn request đang chạy" liên tiếp trước khi dựng lại HTTP client (pool kết nối mới). |

### 2.9 Chạy lại một chunk

| Biến | Mặc định | Ý nghĩa và khi nào nên đổi |
|---|---|---|
| `CHUNK_RETRY_BASE_DELAY` | `5` | Chờ lần đầu trước khi chạy lại một chunk, rồi tăng dần. |
| `CHUNK_RETRY_MAX_DELAY` | `120` | Trần chờ giữa hai lần chạy lại một chunk. |

### 2.10 Cache, logging, chạy thử

| Biến | Mặc định | Ý nghĩa và khi nào nên đổi |
|---|---|---|
| `PROMPT_CACHE` | `auto` | `auto` để provider tự cache tiền tố; `explicit` gửi marker kiểu Anthropic; `off` không gửi marker. |
| `LOG_JSONL` | `1` | `0` là lượt chạy không ghi trace JSONL (nhanh hơn, không có gì để `polyglot trace` đọc). |
| `LLM_VERBOSE` | `0` | `1` log chi tiết mỗi request LLM (thời gian, token). |
| `LLM_VERBOSE_RESILIENCE` | `0` | `1` log chi tiết các quyết định thử lại và fallback. |
| `LLM_MOCK` | `0` | `1` chạy toàn bộ pipeline bằng mock tất định: không cần key, không ra mạng. |

## 3. Công thức sẵn dùng

**Chạy thử không cần khoá** (không cần điền nhóm nào):

```bash
LLM_MOCK=1 polyglot translate --file input/handbook.pdf --src en --tgt vi
```

**Gói miễn phí chỉ cho một request cùng lúc** (tránh 429 `concurrency_limit_exceeded`):
để `LLM_PROVIDER_CONCURRENCY=1` và `MAX_CONCURRENCY=1`. Hai tầng này độc lập:
`LLM_PROVIDER_CONCURRENCY` là hàng đợi ở ranh giới provider, `MAX_CONCURRENCY` là
số chunk chạy song song trong tiến trình. Nâng `MAX_CONCURRENCY` mà quên
`LLM_PROVIDER_CONCURRENCY` thì lượt chạy trông song song mà vẫn tuần tự.

**Model reasoning (deepseek-reasoner, qwen thinking, ...)**: token suy nghĩ bị
tính vào **cùng** ngân sách output với câu trả lời, và không có cách nào bảo
gateway đừng đếm nó. Hai hệ quả:
- `LLM_TIMEOUT` phải đủ cho cả quá trình nghĩ + trả lời (mặc định 300s). Call
  không stream nên read-timeout là hạn chót tổng, không phải hạn chót từng byte.
- Trần output đặt tay phải lớn hơn tổng nghĩ + trả lời. `VERIFIER_MAX_TOKENS=4096`
  với một model nghĩ trung vị ~4200 token là **cắt mọi verdict**; hãy để mặc định
  cho cơ chế tự học trần, hoặc nâng lên 8192. Translator không đặt trần ở lần hỏi
  đầu — chỉ lần hỏi lại sau khi bị cắt mới dùng `TRANSLATOR_MAX_TOKENS`.

**Tiết kiệm token**: giữ `ADAPTIVE_QUALITY=1`, nâng `FAST_PATH_MIN_SCORE` nếu muốn
ít chunk đi đường tắt hơn (chất lượng cao hơn), hoặc đặt
`FAST_PATH_SKIP_LLM_VERIFY=1` để bỏ Verifier trên các chunk đường tắt.

**Đầu ra PDF**: `OUTPUT_FORMAT=pdf` (hoặc `both`) và `PDF_FONT_PATH` nếu máy không
có font Unicode phổ biến.

**Planner/Verifier dùng nhà cung cấp khác**: điền trực tiếp `PLANNER_*` và
`VERIFIER_*`; nhóm đọc-nguồn không ảnh hưởng hai nhóm này.

## 4. Rút gọn `.env` đang có

Một `.env` được sao chép từ bản mẫu cũ có thể đang **đóng băng mặc định**: mọi
biến đều được gán, nên mặc định trong mã không bao giờ tới được lượt chạy. Cách
dọn, chạy trong chính thư mục này:

```bash
grep -oE '^[A-Z][A-Z0-9_]*' .env | sort > /tmp/co.txt   # tên biến bạn đang đặt
```

Rồi đối chiếu với mục 2 ở trên và xoá (hoặc comment) những dòng có giá trị
**trùng với cột mặc định** — chúng không đổi hành vi, chỉ che mặc định tương lai.
Hai giá trị cũ đáng để ý vì **khác** mặc định hiện tại của mã:
`MAX_CONCURRENCY` (bản mẫu cũ ghi `2`, mã mặc định `1`) và `REFINE_MIN_SCORE`
(bản mẫu cũ ghi `8.5`, mã mặc định `7.0`). Giữ lại thì vẫn đúng như trước, chỉ là
bạn nên biết mình đang ghi đè.

Nếu tệp có dòng `[TEMPLATE]`, xoá nó: `python-dotenv` parse nó thành một biến môi
trường tên `[TEMPLATE]` với giá trị rỗng.

## 5. Biến chỉ dành cho web UI

| Biến | Mặc định | Ý nghĩa và khi nào nên đổi |
|---|---|---|
| `PYTHON` | (tự dò) | Interpreter chạy tiến trình con của web UI. Đặt khi `python` trong PATH không phải môi trường ảo của dự án. |
| `RUN_ID` | (tự sinh) | Mã lượt chạy mà tiến trình con ghi vào mọi bản ghi trace. Web UI tự đặt cho con để `run.json` trên đĩa và trace trong log cùng chỉ về một lượt. |

Ngoài ra web UI còn tự đặt cho tiến trình con: `LOGS_DIR`, `LOG_JSONL=1`,
`PYTHONIOENCODING=utf-8` (bắt buộc trên Windows — thiếu nó thì một dòng log có
tiếng Việt làm tiến trình con chết vì lỗi encoding), và `LLM_MOCK=1` cho lượt
chạy ở chế độ mock. Chạy CLI thì không cần đặt tay biến nào trong số này.

## 6. Cấu hình endpoint qua web UI

Biểu tượng bánh răng ở góc phải header mở hộp thoại **Cấu hình endpoint**, nơi
có thể ghi đè lên `.env` cho từng agent: Planner, Translator, Critic, Refiner,
Verifier, Backup (`LLM_BACKUP_*`) và endpoint gom cụm Translator · Critic ·
Refiner (`SOURCE_GROUP_*`), cùng `MAX_CONCURRENCY`.

Cách hoạt động:

- Mỗi ô endpoint được **tự kiểm tra kết nối** khi gõ xong (tạm dừng ~0,7 s):
  thử `GET /models` trước, nếu endpoint không hỗ trợ thì thử một chat
  completion 1 token. Tích xanh nghĩa là endpoint đã trả lời được.
- Checkbox **"Gom cụm Translator · Critic · Refiner"**: tick thì ba agent đó
  dùng chung endpoint gom cụm (các ô riêng của chúng bị mờ đi), bỏ tick thì
  mỗi agent dùng ô riêng của mình.
- Nút **xóa** trên mỗi thẻ gỡ ghi đè của agent đó — agent quay lại dùng đúng
  giá trị `.env`.
- Giá trị chỉ áp dụng cho **lượt dịch mới**. Lúc bắt đầu lượt, web UI kiểm tra
  lại từng endpoint đã lưu (timeout 5 s): endpoint nào còn trả lời thì được
  đưa vào biến môi trường của tiến trình con (biến env thắng `.env`);
  endpoint nào không trả lời thì bị **bỏ qua** và pipeline dùng lại giá trị
  gốc của `.env`.
- API key được lưu tại `data/webui-settings.json` (đã gitignore) và **không
  bao giờ** được trả về trình duyệt — ô key trống nghĩa là "giữ key cũ".

Chạy bằng CLI (`polyglot translate …`) thì các ghi đè này không có tác dụng:
CLI chỉ đọc `.env` và biến môi trường.
