# Spec: UI web điều khiển và theo dõi polyglot-lc

- Ngày: 2026-10-11
- Trạng thái: đã được người dùng duyệt design trên chat, chờ duyệt spec viết
- Phạm vi: một subsystem mới trong thư mục `webui/`, không sửa `src/polyglot_lc/`

## 1. Mục tiêu

Một UI web chạy local cho dự án `polyglot-lc`, phục vụ ba việc:

1. **Điều khiển và theo dõi lượt dịch đang chạy** — chọn file, đặt tham số, bấm chạy, xem từng
   giai đoạn và từng chunk tiến triển theo thời gian thực.
2. **Xem lại lịch sử các lượt chạy** — thời lượng, số request theo tác tử, route `fast` /
   `standard` / `deep`, các chunk lỗi, và tải file kết quả.
3. **Quản lý cấu hình, glossary và skills** — sửa một whitelist cấu hình an toàn, quản lý file
   CSV glossary, xem các skill đã được promote cùng lịch sử điểm.

Người dùng là một người vận hành duy nhất, đang chạy job dịch cỡ cuốn sách. Việc chính của
trang là trả lời câu hỏi "lượt chạy này có khoẻ không" mà không phải đọc JSONL thô.

## 2. Ngoài phạm vi

- Xác thực người dùng hay tài khoản. Server chỉ bind loopback.
- Sửa đổi `src/polyglot_lc/`. Nếu gặp chặn kỹ thuật buộc phải sửa mã nguồn dự án, dừng lại và hỏi người dùng trước — không tự sửa trong cùng lượt việc này.
- Deploy lên internet, container hoá, hay phục vụ nhiều người dùng.
- Dịch trực tuyến từng đoạn nhỏ; UI chỉ điều khiển lượt chạy tài liệu đầy đủ.
- Upload PDF/EPUB mới vào `input/` bằng multipart.

## 3. Ràng buộc đã xác minh trong repo

Các ràng buộc này là lý do của nhiều quyết định dưới đây; chúng đã được kiểm chứng bằng cách
đọc mã, không phải giả định.

| # | Ràng buộc | Bằng chứng |
|---|---|---|
| C1 | `config.py` gọi `load_dotenv()` **lúc import** | [config.py:25](../../../src/polyglot_lc/config.py) |
| C2 | `load_dotenv()` **không ghi đè** biến đã có trong môi trường | ngữ nghĩa của `python-dotenv` |
| C3 | `TokenUsage` và `RetryStats` là trạng thái toàn cục cấp class | `src/polyglot_lc/llm/client.py`, `llm/resilience.py` |
| C4 | `EventSink` ghi mỗi bước thành một dòng JSONL `{ts, step, chunk_id, status, latency, error}` vào `<LOGS_DIR>/run_<ts>.jsonl` | [events.py](../../../src/polyglot_lc/ui/events.py) |
| C5 | `pytest` chỉ thu thập `tests/` | `pyproject.toml`: `testpaths = ["tests"]` |
| C6 | `.gitignore` bỏ qua `*.md` (ngoại lệ duy nhất `README.md`) và `logs/*` | [.gitignore](../../../.gitignore) |
| C7 | `TranslationPipeline.run()` nhận `file_path, src, tgt, domain, glossary, out, max_tokens, resume, checkpoint_dir` | [coordinator.py](../../../src/polyglot_lc/pipeline/coordinator.py) |
| C8 | CLI có `--out`, `--resume/--no-resume`, `--checkpoint-dir`, `--quiet` | [cli.py](../../../src/polyglot_lc/cli.py) |
| C9 | `LLM_MOCK=1` cho phép chạy demo không cần API key | `README.md` mục 6 |
| C10 | Glossary là CSV `src,tgt` mỗi dòng, đọc với `utf-8-sig` | `segment/glossary.py:load_glossary_csv` |
| C11 | Skill là `skills/<src>-<tgt>/<domain>.json` với các khoá `instruction, few_shot, rationale, mean_score, promoted_at, domain, src_lang, tgt_lang, updated_at`; lịch sử là JSONL `{"score", "ts"}` | `skills/en-vi/general.json`, `general.history.jsonl` |
| C12 | Virtualenv ở `.venv/Scripts/python.exe` (Windows) / `.venv/bin/python` | kiểm tra trên máy |
| C13 | Các cổng 8765, 8787, 8000, 5173 đều trống | kiểm tra trên máy |

## 4. Kiến trúc

**Hướng đã chọn: server điều phối, mỗi lượt dịch là một tiến trình con CLI.**

```
Trình duyệt (HTML/CSS/JS thuần)
        │  fetch JSON, polling
        ▼
webui/server.py  ── ThreadingHTTPServer, bind 127.0.0.1
        │
        ├── webui/runs.py   RunManager: spawn / tail / stop / trạng thái
        ├── webui/store.py  đọc logs, output, skills, glossaries; đọc & ghi .env
        └── webui/probe.py  process con: in JSON cấu hình hiệu lực + số chunk
                    │
                    ▼
        <venv>/python -m polyglot_lc.cli translate …   (mỗi lượt một tiến trình)
                    │  LOGS_DIR=<repo>/logs/webui/<run_id>
                    ▼
        logs/webui/<run_id>/run_<ts>.jsonl   ← nguồn sự kiện cho UI
        output/webui/<run_id>/<name>.<tgt>.md ← kết quả
```

**Quy tắc cứng 1 — `webui/*.py` không được `import polyglot_lc`.**
Từ C1 và C2: nếu server import `polyglot_lc.config`, `load_dotenv()` sẽ nạp `.env` vào
`os.environ` **của server**, rồi môi trường đó được truyền xuống tiến trình con, và ở con
`load_dotenv()` bỏ qua các biến đã tồn tại — tức con **không đọc `.env`**, và mọi thay đổi cấu
hình vừa lưu trở nên vô hiệu trong im lặng. Đây là lý do phải có `probe.py` như một tiến trình
riêng thay vì import cho gọn.

**Quy tắc cứng 2 — mỗi lượt chạy có `LOGS_DIR` riêng.**
Từ C4: đặt `LOGS_DIR=<repo>/logs/webui/<run_id>` cho tiến trình con khiến ta biết **chính xác**
file log của lượt đó, thay vì đoán "file mới nhất trong `logs/`" (sai ngay khi có hai lượt chạy
song song).

**Quy tắc cứng 3 — mỗi lượt chạy có `--out` riêng.**
Đặt `--out output/webui/<run_id>/<stem>.<tgt>.md` khiến đường dẫn kết quả là xác định, không phải
dò tìm.

## 5. Thư mục và file

```
webui/
  server.py                  HTTP server + routing + phục vụ static
  runs.py                    RunManager (spawn, tail, stop, trạng thái, run.json)
  store.py                   ConfigStore (đọc/ghi .env), RunsStore, SkillsStore, GlossaryStore
  probe.py                   entry point process con: `--config` và `--plan`
  static/index.html
  static/app.css
  static/app.js
tests/
  test_webui_server.py       routing, 404, traversal, origin, không rò key
  test_webui_store.py        .env round-trip giữ comment; CSV; skills
  test_webui_runs.py         RunManager với lệnh giả; parse event tail; stop
glossaries/                  CSV glossary do UI quản lý (thư mục mới, được commit)
```

Hai điều bắt buộc phải giữ:

- **Test nằm ở `tests/`.** Từ C5, test đặt trong `webui/` sẽ không bao giờ được thu thập.
  `webui/` là thư mục ứng dụng; `tests/` là nơi test của repo (thêm tiền tố `test_webui_` để
  nhóm lại).
- **Không có file `.md` nào trong `webui/`.** Từ C6, chúng sẽ bị ignore.

## 6. Hợp đồng HTTP

Mọi response là JSON (`application/json; charset=utf-8`), trừ route static và route tải file.
Mọi request thay đổi trạng thái yêu cầu header `Origin` hợp lệ (mục 8).

### 6.1 Static

| Method | Route | Trả về |
|---|---|---|
| GET | `/` | `webui/static/index.html` |
| GET | `/static/<file>` | asset trong `webui/static/`, chỉ các đuôi `.html .css .js .svg .woff2` |

### 6.2 Cấu hình

| Method | Route | Body / Trả về |
|---|---|---|
| GET | `/api/config` | `{"values": {KEY: value}, "editable": [...], "readonly": [...], "secrets": {KEY: "set"\|"unset"}}` |
| PUT | `/api/config` | Body `{"values": {KEY: value}}` → `{"saved": [...], "rejected": [{key, reason}]}` |

`values` lấy từ `probe.py --config`, tức **cấu hình hiệu lực** sau khi `.env` và biến môi trường
đã áp dụng — không phải giá trị mặc định chép tay. Whitelist ở mục 7.

Mỗi khoá có thêm `source` ∈ `"env"` | `"dotenv"` | `"default"`:

- `dotenv` khi giá trị hiệu lực **bằng** giá trị ghi trong `.env`;
- `env` khi giá trị hiệu lực **khác** giá trị trong `.env`, hoặc khi `.env` không có khoá đó mà
  biến môi trường lại có — tức biến môi trường đang thắng;
- `default` khi cả `.env` lẫn biến môi trường đều không có khoá đó.

UI **phải** hiển thị `source`, và cảnh báo khi người dùng lưu một khoá đang có `source == "env"`:
vì `load_dotenv()` không ghi đè biến đã tồn tại (ràng buộc C2), giá trị trong `.env` sẽ **không**
có hiệu lực cho tới khi biến môi trường đó được bỏ. Không có cảnh báo này thì UI lưu thành công
mà hành vi không đổi — đúng loại lừa dối im lặng mà cả spec này đang tránh.

### 6.3 Lượt chạy

| Method | Route | Body / Trả về |
|---|---|---|
| GET | `/api/runs` | `{"runs": [{run_id, started, status, file, src, tgt, domain, chunks, exit_code, output_path, mock}]}` sắp mới nhất trước |
| POST | `/api/runs` | Body `{file, src, tgt, domain, glossary, max_tokens, mock}` → `201 {"run_id", "chunks", "output_path", "command"}` |
| GET | `/api/runs/<id>/events?after=N` | `{"events": [...], "next": M, "status": "running"\|"done"\|"failed"\|"stopped", "exit_code": int\|null}` |
| POST | `/api/runs/<id>/stop` | `{"status": "stopping"}` hoặc `409` nếu không ở trạng thái chạy |
| GET | `/api/runs/<id>/output` | file `.md`, `text/markdown; charset=utf-8`; `404` nếu chưa có |
| GET | `/api/runs/<id>` | chi tiết lượt chạy + tham số đã dùng |

`status` được suy ra: `running` khi tiến trình còn sống; `stopped` khi bị dừng theo yêu cầu;
`failed` khi exit code khác 0; `done` khi exit code 0.

`after=N` là **index sự kiện**, không phải timestamp. Server đọc file log một lần, trả các sự kiện
từ index `N`, và `next` là tổng số sự kiện hiện có — client gửi lại `next` ở lần gọi sau. Cách này
không phụ thuộc đồng hồ và chịu được việc file đang được ghi thêm.

### 6.4 Input, glossary, skills

| Method | Route | Body / Trả về |
|---|---|---|
| GET | `/api/inputs` | `{"files": [{name, size, mtime}]}` quét `input/` |
| PUT | `/api/inputs/<name>` | **thân request là bytes thô** của file; ghi vào `input/<name>` |
| GET | `/api/glossaries` | `{"files": [{name, entries}]}` quét `glossaries/` |
| GET | `/api/glossaries/<name>` | `{"name", "pairs": [[src, tgt], ...]}` |
| PUT | `/api/glossaries/<name>` | Body `{"pairs": [[src, tgt], ...]}` → ghi CSV |
| GET | `/api/skills` | `{"skills": [{pair, domain, mean_score, promoted_at, instruction}]}` |
| GET | `/api/skills/<pair>/<domain>` | skill đầy đủ + `{"history": [{score, ts}]}` |

Upload không dùng multipart (mục 9). `pair` là `<src>-<tgt>`.

## 7. Cấu hình: whitelist và bí mật

**Có thể sửa qua UI** (ghi vào `.env`):

`CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `MAX_CONCURRENCY`, `CRITIC_THRESHOLD`,
`REFINE_ROUNDS`, `REFINE_MIN_SCORE`, `RB_TOP_K`, `UNIFIED_PLANNER`, `ADAPTIVE_QUALITY`,
`FAST_PATH_MAX_COMPLEXITY`, `FAST_PATH_MIN_SCORE`, `FAST_PATH_SKIP_LLM_VERIFY`,
`CONSISTENCY_CHECK`, `CONSISTENCY_MAX_PATCHES`, `PARTIAL_OUTPUT`, `RESUME`, `PDF_EXTRACTOR`,
`LLM_MOCK`

**Chỉ hiển thị, không cho sửa**: `LOGS_DIR`, `SKILLS_DIR`, `REASONING_BANK_PATH`,
`CHECKPOINT_DIR` — đổi đường dẫn từ UI có thể làm lượt chạy sau ghi ra ngoài dự kiến, mà lợi ích
gần như bằng không.

**Bí mật — không bao giờ chạm tới.** Với mọi biến khớp `*_API_KEY`, và mọi biến có một
thành phần tên (tách bởi `_`) **bằng** `KEY`/`TOKEN`/`SECRET`/`PASSWORD`, hoặc có tên đã bỏ
`_` **kết thúc** bằng một trong các từ đó (`OPENAI_APIKEY`, `x_token`), server:

- không đọc giá trị, kể cả để hiển thị hay che một phần;
- không trả về trong bất kỳ response nào, kể cả `raw`;
- không ghi, không xoá, không sửa dòng đó khi PUT `/api/config`;
- chỉ trả `"set"` hoặc `"unset"` dựa trên việc biến có mặt trong `.env` hay môi trường.

Ghi `.env` phải **atomic** (file tạm cùng thư mục rồi `os.replace`), giữ nguyên thứ tự, comment và
mọi dòng không thuộc whitelist; chỉ thay đúng dòng của khoá cần đổi, và thêm dòng mới ở cuối nếu
khoá chưa tồn tại.

**Thứ tự ưu tiên của cấu hình** (phải hiển thị cho người dùng, xem mục 6.2): biến môi trường của
tiến trình server > `.env` > default trong `config.py`. Vì `load_dotenv()` không ghi đè (C2) và
tiến trình con thừa hưởng môi trường của server, một biến đã `export` ở shell sẽ thắng `.env`.
Đây là hành vi của dự án, không phải điều spec này đổi — spec chỉ buộc UI nói ra sự thật đó.

## 8. An toàn

| Rủi ro | Biện pháp |
|---|---|
| Lộ ra mạng ngoài | Bind cứng `127.0.0.1`, không có tham số đổi thành `0.0.0.0` |
| CSRF từ trang khác trong trình duyệt | Mọi `POST`/`PUT` kiểm `Origin`; thiếu hoặc khác `http://127.0.0.1:<port>` / `http://localhost:<port>` → `403` |
| Đọc/ghi ngoài thư mục cho phép | Từ chối mọi `<name>` chứa `/`, `\`, `..`, hoặc rỗng; kiểm sau khi đã `Path.resolve()` và xác nhận vẫn nằm trong thư mục gốc |
| Rò API key | Mục 7 |
| Process con mồ côi khi dừng | POSIX: `start_new_session=True` rồi `os.killpg`; Windows: `CREATE_NEW_PROCESS_GROUP` rồi `taskkill /T /F /PID` |
| Đường dẫn `.env` bị ghi hỏng | Ghi atomic, giữ comment |
| Ghi đè dữ liệu người dùng | `PUT /api/inputs/<name>` từ chối nếu file đã tồn tại, trừ khi query `?overwrite=1` |

## 9. Quyết định kỹ thuật và lý do

| Quyết định | Lý do |
|---|---|
| **Polling `?after=N` mỗi ~700ms thay vì SSE** | Với `ThreadingHTTPServer`, SSE cần vòng đời kết nối dài, heartbeat, và xử lý client ngắt giữa chừng; polling chỉ là tra dict + đọc file. Chọn cách nhàm chán hơn. |
| **Upload bằng thân request thô, không multipart** | `fetch(url, {method:'PUT', body: file})` gửi bytes trực tiếp. Tránh phải parse multipart, và tránh module `cgi` (đã bị xoá ở Python 3.13). |
| **`probe.py` là process con, không import trong server** | Quy tắc cứng 1. Đồng thời tránh chép lại default của `config.py` vào webui. |
| **Dừng lượt chạy bằng kill cây process** | Chỉ kill PID cha để lại process con mồ côi đang giữ provider connection. |
| **`--out` và `LOGS_DIR` riêng cho từng lượt** | Quy tắc cứng 2 và 3. |
| **`run.json` trong thư mục lượt chạy** | Lịch sử phải sống sót khi server restart; trạng thái trong RAM sẽ mất. |
| **Lịch sử quét `logs/webui/*/run.json`** | Không cần cơ sở dữ liệu cho dữ liệu vốn đã nằm trên đĩa. |

## 10. Thiết kế hình ảnh

**Chủ thể:** thế giới của dự án là **mực in trên giấy** — PDF, dấu chấm dẫn mục lục, glyph
ligature. Và việc khó nhất của pipeline là **hai mực phải rơi đúng cùng một chỗ**: một thuật ngữ,
một bản dịch, xuyên qua các chunk. Đó là ngôn ngữ của *dấu đăng ký in* — cyan và magenta lệch nhau
là bản in hỏng. Bảng màu lấy từ đó, không phải màu dùng chung cho trang khác.

### 10.1 Token

| token | hex | dùng cho |
|---|---|---|
| `--paper` | `#F2F3F5` | nền (giấy lạnh, không phải kem) |
| `--panel` | `#FFFFFF` | bảng, thẻ |
| `--ink` | `#14161A` | chữ |
| `--rule` | `#D7DAE0` | đường kẻ 1px |
| `--cyan` | `#0E6E7D` | đang chạy / đạt / accent duy nhất |
| `--magenta` | `#B3247A` | lệch thuật ngữ, chunk đã bị vá |
| `--amber` | `#A96A11` | retry, cảnh báo |
| `--red` | `#B4232F` | lỗi |

### 10.2 Chữ — ba vai, không webfont, không CDN

| vai | stack | dùng cho |
|---|---|---|
| văn bản đã dịch | `ui-serif, Georgia, "Iowan Old Style", "Times New Roman", serif` | khung proof |
| chrome UI | `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif` | nhãn, nút, tiêu đề |
| dữ liệu | `ui-monospace, "Cascadia Mono", Consolas, monospace` + `font-variant-numeric: tabular-nums` | id chunk, thời lượng, số request |

Không dùng webfont vì UI phải chạy được offline. Ba vai này mang thông tin: chữ có chân là tài
liệu, mono là cỗ máy.

### 10.3 Bố cục

```
┌──────────────┬────────────────────────────────────────────────────────┐
│ LAUNCHER     │  ingest › detect › segment › glossary › plan › chunks ›  │
│  file ▾      │  consistency › assemble › skillopt                       │
│  src/tgt     ├────────────────────────────────────────────────────────┤
│  domain      │  CHUNK MATRIX    ■■■■■■■■■□□□□□□□                        │
│  glossary ▾  │   fast ■ standard ■ deep ■ retry ■ fail ■ chờ □          │
│  □ LLM_MOCK  ├────────────────────────────────────────────────────────┤
│  [ Chạy ]    │  PROOF   output/webui/…/<stem>.<tgt>.md   (serif)        │
├──────────────┤  ────────────────────────────────────────────────────────│
│ LỊCH SỬ      │  SỰ KIỆN   plan ok 1.2s · chunk-004 verify retry · …     │
│ 16:17 ✓ 88   │  (mono, tabular, cuộn, chỉ thêm dòng mới)                │
│ 15:20 ✗ 12   │                                                          │
└──────────────┴────────────────────────────────────────────────────────┘
```

Cột trái cố định 280px. Cột phải bốn dải xếp dọc: dải giai đoạn, chunk matrix, proof, sự kiện.
Dưới 900px thì cột trái thu thành thanh ngang trên cùng.

### 10.4 Nguyên tắc

1. Một màn hình trả lời được "khoẻ hay không" mà không phải cuộn.
2. Phần tử đáng nhớ là **chunk matrix**: N ô tự đầy khi chunk xong; nhìn là biết còn bao xa và
   chunk nào có vấn đề. Ô đổi màu theo `path` (`fast` / `standard` / `deep`) khi biết, và theo
   `retry` / `fail` khi sự kiện nói vậy.
3. Một chuyển động duy nhất: các ô matrix đầy vào khi sự kiện tới. Không fade-in từng mục, không
   transition trên mọi card.
4. Màu mang nghĩa, không trang trí. Không emoji. Không nhãn IN HOA.
5. Mọi con số là mono canh cột.
6. Nội dung viết ở thể chủ động, câu chữ thường, và lỗi nói việc gì đã xảy ra cùng việc cần làm.

### 10.5 Ánh xạ sự kiện → giao diện

| Nguồn | Hiển thị |
|---|---|
| `step` ∈ {ingest, detect_lang, segment, glossary, plan, consistency, assemble_output, skillopt_update} | dải giai đoạn |
| `step` ∈ {translate, critic, refine, verify, retranslate, save_trace} + `chunk_id` | ô matrix tương ứng, dòng sự kiện |
| `status` = `start` \| `ok` \| `fail` \| `retry` \| `info` | cyan tiến trình / cyan đạt / đỏ lỗi / amber retry / xám thông tin |
| `latency` | cột mono canh phải |
| `error` (khi `status == "info"`) | ghi chú của consistency pass, tô magenta |

## 11. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Chưa cài `.venv` | Server vẫn chạy; `POST /api/runs` trả `503` kèm hướng dẫn tạo virtualenv |
| `LOG_JSONL=0` trong `.env` | UI cảnh báo lượt chạy sẽ không có sự kiện, và đề nghị bật; lượt chạy vẫn chạy được |
| `probe.py --plan` lỗi (file hỏng, thiếu thư viện, **hoặc file input không tồn tại**) | `POST /api/runs` trả `422` kèm stderr của probe; **không** khởi động lượt chạy, không tạo thư mục lượt chạy nào |
| Tên file input chứa `..`, `/`, `\` hoặc trỏ ra ngoài `input/` | `403`; tên sai bị từ chối **sau khi `unquote`**, trước khi ghép đường dẫn |
| File log bị ghi dở một dòng | Bỏ qua dòng JSON hỏng, giữ các dòng còn lại, đếm số dòng hỏng |
| Server khởi động lại khi một lượt đang chạy | `run.json` có `pid`; nếu pid không còn sống thì `status = "failed"` với ghi chú "server đã khởi động lại khi lượt chạy đang chạy" |
| `stop` khi tiến trình đã xong | `409`, không kill PID đã được tái sử dụng |
| Trùng tên khi upload | `409` trừ khi `?overwrite=1` |
| `PUT /api/config` với khoá ngoài whitelist | Khoá đó vào `rejected` kèm lý do; các khoá hợp lệ vẫn được lưu |

## 12. Kiểm thử

Test nằm ở `tests/` (ràng buộc C5). Không test nào cần API key hay gọi provider.

**`tests/test_webui_store.py`**
- ghi `.env`: comment, dòng trống và dòng ngoài whitelist được giữ nguyên; chỉ dòng của khoá cần
  đổi bị thay; khoá mới được thêm ở cuối.
- `*_API_KEY` không bao giờ xuất hiện trong output của `ConfigStore` (đọc và ghi).
- Không tên nào trong whitelist bị coi là bí mật, và `*_TOKENS` (ngân sách token) không bị nhầm
  thành credential.
- Ghi `.env` giữ nguyên kiểu xuống dòng CRLF, không biến cả file thành LF.
- CSV glossary round-trip, kể cả khi file có BOM.
- `SkillsStore` đọc được `skills/<pair>/<domain>.json` và `*.history.jsonl`.

**`tests/test_webui_probe.py`**
- Phân loại `source`: giá trị khớp `.env` → `dotenv`; giá trị khác `.env` do biến môi trường
  thắng → `env`; khoá không có ở `.env` lẫn môi trường → `default`.
- `--plan` trả số chunk bằng `len(chunk_document(load_document(path)))`.

**`tests/test_webui_server.py`**
- `GET /` trả `index.html`; `GET /static/<file>` trả đúng content-type; đuôi lạ bị từ chối.
- `GET /api/unknown` → `404` JSON.
- `PUT /api/inputs/../secret` và `PUT /api/inputs/a/b` → `403`.
- `POST`/`PUT` với `Origin: http://evil.example` → `403`; thiếu `Origin` → `403`.
- `GET /api/config` không chứa bất kỳ giá trị key nào (test khẳng định chuỗi giá trị key giả không
  xuất hiện trong body).
- `GET /api/runs/<id>/events?after=N` trả đúng các sự kiện từ index N và `next` đúng.
- `POST /api/runs` khi `probe.py` lỗi → `422`.

**`tests/test_webui_runs.py`**
- `RunManager` với một lệnh giả (`python -c "print(...)"`) để kiểm tra vòng đời:
  `running` → `done`, `exit_code`, ghi `run.json`.
- `stop()` trên lệnh giả chạy dài → trạng thái `stopped`, tiến trình con không còn sống.
- File log có một dòng JSON hỏng → các dòng còn lại vẫn đọc được.
- `probe.py --plan` trên một file `.md` nhỏ tạo tại chỗ: số chunk bằng
  `len(chunk_document(load_document(path)))` theo chính thư viện.

**Cách chạy:** `python -m pytest tests/test_webui_store.py tests/test_webui_server.py tests/test_webui_runs.py`
— và toàn bộ suite `python -m pytest` phải tiếp tục xanh (289 test hiện có không được vỡ).

## 13. Tiêu chí nghiệm thu

1. `python webui/server.py` mở được UI ở `http://127.0.0.1:8787` mà không cần cài thêm gì.
2. Chạy được một lượt dịch thật với `LLM_MOCK=1` từ UI: chọn file trong `input/`, bấm Chạy, thấy
   dải giai đoạn tiến triển, chunk matrix đầy dần, và file kết quả mở được từ UI.
3. Sửa `CHUNK_MAX_TOKENS` trong UI, lưu, chạy lượt mới, và số chunk thay đổi tương ứng — chứng
   minh `.env` được con đọc thật (đây là phép thử trực tiếp cho ràng buộc C1/C2).
4. Lịch sử liệt kê cả lượt của lần chạy trước sau khi khởi động lại server.
5. `grep -r "API_KEY" webui/static/` không ra giá trị nào, và response của `GET /api/config` không
   chứa giá trị key (đã có test).
6. `ruff check .` sạch và `python -m pytest` xanh.
7. Không có file nào trong `src/polyglot_lc/` bị sửa.

## 14. Rủi ro đã biết

- **Polling 700ms × nhiều tab** có thể tạo tải nhỏ lên server khi có job dài; chấp nhận được cho
  một người dùng, và có thể nới khoảng polling ở client nếu cần.
- **Kill cây process khác nhau giữa Windows và POSIX** là chỗ dễ sai; có test riêng cho `stop`.
  Máy đích là Windows nên nhánh đó phải được kiểm tra thật, không chỉ suy luận.
- **`probe.py --plan` chạy `chunk_document`** tốn thời gian trên PDF rất lớn; chấp nhận vì nó chỉ
  chạy một lần khi bắt đầu lượt.
- **Chunk matrix cần biết `N` trước**; lấy từ `probe.py --plan`. Nếu probe lỗi thì không có lượt
  chạy, nên không có trạng thái "matrix thiếu dữ liệu".
- **Mục 10.4 nói ô matrix đổi màu theo `path` (`fast`/`standard`/`deep`), nhưng log không có
  trường đó.** `polyglot_lc.ui.events.Event` chỉ mang `ts, step, chunk_id, status, latency,
  error` — router và độ phức tạp không ghi `path` vào sự kiện. Vì vậy UI hiện tô màu theo những
  gì log **thật sự** có: `đang chạy` / `xong` / `có ghi chú thuật ngữ` (từ sự kiện `info` của
  consistency pass) / `phải thử lại` / `lỗi` / `chờ`. Muốn có đúng ba mức `fast/standard/deep`
  thì phải thêm một trường vào `Event` và cho router phát nó — đó là thay đổi ở `polyglot_lc`,
  ngoài phạm vi plan này. Ghi lại ở đây để không ai tưởng ô màu là ngẫu nhiên.
- **`stop` giết tiến trình nhưng không thu hồi được công đã gọi provider**; một lượt bị dừng vẫn
  tính tiền cho các request đã gửi. UI nói "đã dừng", không nói "đã hoàn tác".
