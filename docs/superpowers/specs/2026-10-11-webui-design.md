# Spec: UI web dịch và đọc tài liệu PDF

- Ngày: 2026-10-11, sửa lại 2026-10-12 (bỏ toàn bộ bề mặt cấu hình)
- Trạng thái: đã duyệt, đã thiết kế lại và đã triển khai
- Phạm vi: một subsystem mới trong thư mục `webui/`

## 1. Mục tiêu

Một UI web chạy local cho `polyglot-lc`, phục vụ **đọc và dịch tài liệu PDF**, ba việc:

1. **Dịch một tài liệu** — chọn file trong `input/` hoặc tải file mới lên, đặt ngôn ngữ, lĩnh vực,
   từ điển, ngân sách token, và chọn kết quả xuất ra (Markdown, PDF giữ định dạng, hoặc cả hai).
2. **Theo dõi lượt dịch đang chạy** — tiến độ theo chunk, các giai đoạn của pipeline, và khi có
   trục trặc thì biết chính xác bước nào, chunk nào, lời nhắn gì.
3. **Đọc kết quả** — hai khung cạnh nhau: bản gốc đã trích xuất và bản dịch, cộng với việc mở/tải
   file kết quả. Lịch sử các lượt trước nằm ở cột trái.

Người dùng là một người vận hành duy nhất, đang dịch một cuốn sách. Trang phải trả lời được
"lượt chạy này có khoẻ không, và bản dịch đã đọc được chưa".

**Điều trang này cố tình không làm:** nó không phải bảng điều khiển cấu hình. Không có màn hình
cài đặt, không có trạng thái API key, không có công tắc chạy thử, không có nhật ký sự kiện thô,
và không có bất kỳ chỗ nào trong ứng dụng — hay trong bất kỳ response nào — đọc hoặc ghi `.env`
(mục 7). Cấu hình là việc của terminal.

## 2. Ngoài phạm vi

- Xác thực người dùng hay tài khoản. Server chỉ bind loopback.
- Deploy lên internet, container hoá, phục vụ nhiều người dùng.
- Dịch trực tuyến từng đoạn nhỏ; UI chỉ điều khiển lượt chạy tài liệu đầy đủ.
- Upload bằng multipart; không OCR (PDF không có lớp chữ bị từ chối, UI nói rõ lý do).
- **Mọi bề mặt cấu hình.** Xem mục 7: ứng dụng không biết tên một khoá cấu hình nào.

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
| C8 | CLI có `--out`, `--format`, `--resume/--no-resume`, `--checkpoint-dir`, `--quiet` | [cli.py](../../../src/polyglot_lc/cli.py) |
| C9 | `LLM_MOCK=1` cho phép chạy demo không cần API key; lượt chạy **thừa hưởng** môi trường của server | `README.md` mục 6, [runs.py `env_for`](../../../webui/runs.py) |
| C10 | Glossary là CSV `src,tgt` mỗi dòng, đọc với `utf-8-sig` | `segment/glossary.py:load_glossary_csv` |
| C11 | Skill là `skills/<src>-<tgt>/<domain>.json`; lịch sử là JSONL `{"score", "ts"}` | `skills/en-vi/general.json`, `general.history.jsonl` |
| C12 | Chỉ PDF có lớp chữ được nhận; `input/` chỉ chứa `.pdf` là đầu vào hợp lệ | [server.py `INPUT_SUFFIX`](../../../webui/server.py), `ingest/` |
| C13 | Kết quả của một lượt là `<stem>.<tgt>.md` và (nếu có) `<stem>.<tgt>.pdf` cùng thư mục | `pipeline/assembler.output_paths` |
| C14 | Trình duyệt không tự gửi `Origin` cho request `GET` cùng origin, nhưng **có** gửi cho `POST`/`PUT` cùng origin | hành vi của Chromium/Firefox |

C14 là lý do cách bảo vệ `Origin` hoạt động được mà không cần token: request đọc không cần header,
request ghi thì trình duyệt tự thêm.

## 4. Kiến trúc

**Hướng đã chọn: server điều phối, mỗi lượt dịch là một tiến trình con CLI.**

```
Trình duyệt (HTML/CSS/JS thuần)
        │  fetch JSON, polling 700ms
        ▼
webui/server.py  ── ThreadingHTTPServer, bind 127.0.0.1
        │
        ├── webui/runs.py   RunManager: spawn / tail / stop / trạng thái
        ├── webui/store.py  đọc logs/webui/<run_id>, skills/, glossaries/
        └── webui/probe.py  tiến trình con: in JSON (plan | text | memory)
                    │
                    ▼
        <python> -m polyglot_lc.cli translate …      (mỗi lượt một tiến trình)
                    │  LOGS_DIR=<repo>/logs/webui/<run_id>
                    ▼
        logs/webui/<run_id>/run_<ts>.jsonl   ← nguồn sự kiện cho UI
        output/webui/<run_id>/<stem>.<tgt>.{md,pdf}  ← kết quả
```

**Quy tắc cứng 1 — `webui/*.py` không được `import polyglot_lc`.**
Từ C1 và C2: nếu server import `polyglot_lc.config`, `load_dotenv()` sẽ nạp `.env` vào
`os.environ` **của server**, rồi môi trường đó được truyền xuống tiến trình con, và ở con
`load_dotenv()` bỏ qua các biến đã tồn tại — tức con **không đọc `.env`** nữa. Đây là lý do
`probe.py` phải là tiến trình riêng thay vì import cho gọn: nó là **cách duy nhất** để hỏi
pipeline thật một câu (đọc tài liệu, đếm chunk, đọc bộ nhớ dịch) mà không đầu độc môi trường.

**Quy tắc cứng 2 — mỗi lượt chạy có `LOGS_DIR` riêng.**
Từ C4: đặt `LOGS_DIR=<repo>/logs/webui/<run_id>` cho tiến trình con khiến ta biết **chính xác**
file log của lượt đó, thay vì đoán "file mới nhất trong `logs/`" (sai ngay khi có hai lượt chạy).

**Quy tắc cứng 3 — mỗi lượt chạy có `--out` riêng.**
Đặt `--out output/webui/<run_id>/<stem>.<tgt>.md` khiến đường dẫn kết quả là xác định, không phải
dò tìm; PDF đi kèm suy ra từ chính đường dẫn đó (C13).

## 5. Thư mục và file

```
webui/
  paths.py                   phân giải đường dẫn an toàn (safe_child, is_within)
  server.py                  HTTP server + routing + phục vụ static
  runs.py                    RunManager (spawn, tail, stop, trạng thái, run.json)
  store.py                   RunsStore, SkillsStore, GlossaryStore
  probe.py                   tiến trình con: --plan | --text | --memory
  static/index.html          khung trang
  static/app.css             hệ thống thiết kế (token, layout, mọi trạng thái)
  static/app.js              hành vi (polling, khung đọc, hộp thoại, ngăn kéo)
tests/
  test_webui_paths.py        safe_child: traversal, tên rỗng, Windows separator
  test_webui_store.py        run.json, CSV glossary, skills, lịch sử điểm
  test_webui_probe.py        --plan khớp chunker thật; lỗi ra stderr, exit ≠ 0
  test_webui_runs.py         RunManager với lệnh giả; sự kiện; stop; ghi chú
  test_webui_server.py       routing, Origin, traversal, upload, lượt chạy, artefact
glossaries/                  CSV glossary do UI quản lý (được commit, có .gitkeep)
```

Hai điều bắt buộc phải giữ:

- **Test nằm ở `tests/`.** Từ C5, test đặt trong `webui/` sẽ không bao giờ được thu thập.
- **Không có file `.md` nào trong `webui/`.** Từ C6, chúng sẽ bị ignore.

## 6. Hợp đồng HTTP

Mọi response là JSON (`application/json; charset=utf-8`), trừ route static và route tải file.
Mọi request thay đổi trạng thái yêu cầu header `Origin` hợp lệ (mục 8).

### 6.1 Static

| Method | Route | Trả về |
|---|---|---|
| GET | `/` | `webui/static/index.html` |
| GET | `/static/<file>` | asset trong `webui/static/`, chỉ các đuôi `.html .css .js .svg .woff2` |

Static trả `Cache-Control: no-store` và `X-Content-Type-Options: nosniff`.

### 6.2 Cấu hình — không có

**Không tồn tại route nào cho cấu hình.** `GET /api/config`, `PUT /api/config`,
`GET /api/config/<x>` và `POST /api/settings` đều trả `404` JSON, và router **không giữ store nào
có thể** đọc hay ghi `.env` (test khẳng định `getattr(api, "config", None) is None`).

Bốn test khoá tính chất này lại, vì nó là cả mục đích của thiết kế:

1. `/api/config` (đọc và ghi) là `404`, và `.env` trên đĩa **không đổi**;
2. không response nào của `/`, `/api/health`, `/api/inputs`, `/api/runs`, `/api/glossaries`,
   `/api/skills`, `/api/memory` chứa `CHUNK_MAX_TOKENS`, `TRANSLATOR_API_KEY`, `EDITABLE` hay
   `dotenv` — kể cả **tên** khoá;
3. ba asset đã phát hành (`index.html`, `app.css`, `app.js`) không chứa `api/config`, `.env`,
   `LLM_MOCK`, `EDITABLE`, "API key", "Tùy chọn dịch" hay "Chạy thử";
4. `probe.py --config` vẫn tồn tại như **công cụ dòng lệnh** cho người vận hành, nhưng không có
   đường nào từ nó tới trình duyệt.

### 6.3 Lượt chạy

| Method | Route | Body / Trả về |
|---|---|---|
| GET | `/api/runs` | `{"runs": [...]}` mới nhất trước; mỗi dòng có `run_id, started, status, file, src, tgt, domain, chunks, exit_code, output_path, note, has_md, has_pdf` |
| POST | `/api/runs` | Body `{file, src, tgt, domain, glossary, max_tokens, format, resume}` → `201 {"run_id", "chunks", "output_path", "artifacts"}` |
| GET | `/api/runs/<id>` | `{"run", "md", "md_path", "pdf", "pdf_path", "artifacts", "problems"}` |
| GET | `/api/runs/<id>/events?after=N` | `{"events", "next", "status", "exit_code", "malformed", "has_events"}` |
| POST | `/api/runs/<id>/stop` | `{"status", "exit_code"}`; `409` nếu không ở trạng thái chạy |
| GET | `/api/runs/<id>/output[?download=1]` | `.md`, `text/markdown; charset=utf-8`; `404` nếu chưa có |
| GET | `/api/runs/<id>/pdf[?download=1]` | `.pdf`, `application/pdf` |
| GET | `/api/runs/<id>/bundle` | `application/zip` chứa **đúng** những file đã có |
| GET | `/api/runs/<id>/source` | `{"text", "chars", "blocks", "title", "cached"}` |

Ba quyết định trong bảng này đáng nói ra:

- **`/api/runs` không trả `command`.** Đường lệnh có tên interpreter, đường dẫn tuyệt đối và các
  cờ — chi tiết của người phát triển, không phải của người đọc bản dịch. `mock` và `pid` cũng vậy.
- **`format` là một lựa chọn trong `md | pdf | both`**, khớp đúng `--format` của CLI. Hai checkbox
  sẽ cho phép trạng thái thứ tư (không chọn gì) mà server chỉ có thể trả `400`, và biến "cả hai"
  thành kết quả tình cờ của việc tick hai ô.
- **`/api/runs/<id>/source` trả văn bản của chính bộ trích xuất** (qua `probe --text`), cache theo
  `(đường dẫn, size, mtime_ns)`, tối đa 4 mục. Đọc lại cả cuốn sách mỗi lần đổi khung sẽ làm
  trang có cảm giác hỏng, còn đọc bằng một cách khác thì khung so sánh sẽ hiển thị một văn bản
  mà bộ dịch chưa từng thấy.

`status` được suy ra: `running` khi tiến trình còn sống; `stopped` khi bị dừng theo yêu cầu;
`failed` khi exit code khác 0; `done` khi exit code 0. Mỗi lần poll đều hoà giải lại với tiến
trình con và với đĩa, nên một lượt chết trong lúc server tắt không thể "đang chạy" mãi.

`after=N` là **index sự kiện**, không phải timestamp: server trả các sự kiện từ index `N` và
`next` là tổng số sự kiện hiện có. Cách này không phụ thuộc đồng hồ và chịu được việc file đang
được ghi thêm. `malformed` đếm số dòng JSONL hỏng đã bị bỏ qua, và `has_events` cho biết lượt
chạy có ghi log hay không — thiếu nó thì client không phân biệt được "chưa có gì xảy ra" với
"lượt này sẽ không bao giờ báo gì".

### 6.4 Input, glossary, skills, memory

| Method | Route | Body / Trả về |
|---|---|---|
| GET | `/api/inputs` | `{"files": [{name, size, mtime}]}` quét `input/`, **chỉ `.pdf`** |
| PUT | `/api/inputs/<name>` | thân request là bytes thô; `?overwrite=1` để ghi đè |
| GET | `/api/glossaries` | `{"files": [{name, entries}]}` quét `glossaries/` |
| GET | `/api/glossaries/<name>` | `{"name", "pairs": [[src, tgt], ...]}` |
| PUT | `/api/glossaries/<name>` | Body `{"pairs": [[src, tgt], ...]}` → ghi CSV |
| GET | `/api/skills` | `{"skills": [{pair, domain, mean_score, instruction, ...}]}` |
| GET | `/api/skills/<pair>/<domain>` | skill đầy đủ + `{"history": [{score, ts}]}` |
| GET | `/api/memory` | `{"memory": {path, total, sampled, pairs: [...]}, "cached"}` |

Upload không dùng multipart (mục 9). `pair` là `<src>-<tgt>`.

`/api/memory` là **chỉ đọc** và đi qua `probe --memory`, thứ mở collection ChromaDB bằng
`get_collection` + `get()`. Nó không bao giờ `create_collection`, vì lần đọc của một bảng hiển
thị không được phép **tạo** dữ liệu; ngân hàng chưa tồn tại thì được báo là rỗng. Báo cáo được
giữ lại 60 giây trong server (mở ChromaDB tốn một lần import).

## 7. Cấu hình: không có bề mặt nào

Đây là thay đổi lớn so với bản 2026-10-11, và là phần dễ bị làm hỏng lại nhất.

**Ứng dụng không biết gì về cấu hình.** Không đọc, không ghi, không liệt kê, không kiểm tra
trạng thái; không trả về giá trị, **cũng không trả về tên khoá**. Ngay cả việc "khoá này đã được
đặt hay chưa" cũng là thông tin về cấu hình nên cũng không có.

Hệ quả trực tiếp:

- Lượt chạy **thừa hưởng môi trường của tiến trình server** (`env_for` chỉ thêm `LOGS_DIR`,
  `LOG_JSONL`, `PYTHONIOENCODING`, và `LLM_MOCK` khi bản ghi nói đây là lượt chạy thử — do test
  hoặc terminal đặt, không do trang gửi).
- Những gì trang được phép gửi **chỉ là những gì thuộc về một lượt chạy**: ngôn ngữ nguồn và
  đích, lĩnh vực, glossary, ngân sách token mỗi chunk, định dạng kết quả, và cờ tiếp tục.
- Muốn đổi `CHUNK_MAX_TOKENS`, `MAX_CONCURRENCY` hay bất cứ thiết lập nào khác thì sửa `.env` rồi
  chạy lại server — đó là việc của người vận hành ở terminal, và `python webui/probe.py --config`
  là công cụ để xem cấu hình hiệu lực.
- Bí mật vì thế không có đường tới trình duyệt: không có route nào đọc chúng, nên không có route
  nào có thể trả chúng. Phần che bí mật (mục 8) chỉ còn để xử lý văn bản **do provider trả về**
  trong log.

## 8. An toàn

| Rủi ro | Biện pháp |
|---|---|
| Lộ ra mạng ngoài | Bind cứng `127.0.0.1`, không có tham số đổi thành `0.0.0.0` |
| CSRF từ trang khác trong trình duyệt | Mọi `POST`/`PUT` kiểm `Origin`; thiếu hoặc khác `http://127.0.0.1:<port>` / `http://localhost:<port>` → `403`. Cách này chỉ dùng được vì C14: trình duyệt luôn gửi `Origin` cho request ghi cùng origin |
| Đọc/ghi ngoài thư mục cho phép | Từ chối mọi `<name>` chứa `/`, `\`, `..`, hoặc rỗng; kiểm sau khi đã `unquote` và `Path.resolve()`. `..%2F.env` phải bị từ chối **vì là đường dẫn**, không phải vì đuôi file |
| Rò credential từ log | `_redact` thay mọi chuỗi dạng `sk-…`/`pk-…`/`api-…`/`token-…` bằng `***` trên **mọi** văn bản trả ra: `note` của bản ghi, `error` của sự kiện, và `problems`. Test dùng một chuỗi canary và soi JSON của cả ba route |
| Process con mồ côi khi dừng | POSIX: `start_new_session=True` rồi `os.killpg`; Windows: `CREATE_NEW_PROCESS_GROUP` rồi `taskkill /T /F /PID` |
| PID được tái sử dụng | `stop()` hoà giải trạng thái trước, và từ chối (`409`) nếu lượt chạy đã xong, thay vì signal một pid lạ |
| Ghi đè dữ liệu người dùng | `PUT /api/inputs/<name>` trả `409` nếu file đã tồn tại, trừ khi có `?overwrite=1`; trang chỉ gửi cờ đó sau khi người dùng thấy lời đề nghị ghi đè |
| Đọc file tuỳ ý qua `run.json` bị sửa tay | `output_path` trong bản ghi được kiểm `is_within(root)` trước khi đọc; ra ngoài repo → `403` |
| Upload quá lớn | `MAX_UPLOAD_BYTES` (512 MB) bị kiểm **trước khi** đọc vào bộ nhớ; không phải PDF (thiếu `%PDF-` trong 1 KB đầu) → `422` |

## 9. Quyết định kỹ thuật và lý do

| Quyết định | Lý do |
|---|---|
| **Polling `?after=N` mỗi ~700ms thay vì SSE** | Với `ThreadingHTTPServer`, SSE cần vòng đời kết nối dài, heartbeat, và xử lý client ngắt giữa chừng; polling chỉ là tra dict + đọc file. Chọn cách nhàm chán hơn. |
| **Upload bằng thân request thô, không multipart** | `fetch`/`XMLHttpRequest` gửi bytes trực tiếp. Tránh phải parse multipart, và tránh module `cgi` (đã bị xoá ở Python 3.13). |
| **`XMLHttpRequest` chỉ cho upload** | Đây là API duy nhất báo được tiến độ gửi lên; phần còn lại của trang dùng `fetch`. |
| **`probe.py` là process con, không import trong server** | Quy tắc cứng 1. Đồng thời tránh chép lại default của `config.py` vào `webui`. |
| **Dừng lượt chạy bằng kill cây process** | Chỉ kill PID cha để lại process con mồ côi đang giữ provider connection. |
| **Không hỏi xác nhận khi dừng** | Các chunk đã xong nằm trong checkpoint nên công việc không mất, và nút "Tiếp tục" nằm ngay cạnh. Một hộp thoại chắn trước hành động đảo ngược được chỉ dạy người dùng bấm bừa. |
| **`--out` và `LOGS_DIR` riêng cho từng lượt** | Quy tắc cứng 2 và 3. |
| **`run.json` trong thư mục lượt chạy** | Lịch sử phải sống sót khi server restart; trạng thái trong RAM sẽ mất. |
| **Lịch sử quét `logs/webui/*/run.json`** | Không cần cơ sở dữ liệu cho dữ liệu vốn đã nằm trên đĩa. |
| **Bỏ hẳn `ConfigStore`, `/api/config`, và bảng chunk matrix** | Bề mặt cấu hình là thứ phải bỏ (mục 7). Matrix đổi màu theo `path` (`fast`/`standard`/`deep`) mà `Event` không hề mang trường đó — một thành phần không thể nói thật thì không nên tồn tại. Tiến độ theo chunk được thay bằng thanh tiến độ + dải giai đoạn, cả hai đều đọc từ dữ liệu thật. |

## 10. Thiết kế hình ảnh

Trang này là **nơi đọc tài liệu**, không phải bảng điều khiển của lập trình viên. Tài liệu là
trung tâm: khung đọc chiếm phần lớn màn hình, và phần chrome quanh nó chỉ để bắt đầu một lượt,
theo dõi trung thực, và mang kết quả đi.

### 10.1 Token

Nền trung tính lạnh, một accent duy nhất. Accent là màu mực của chính dự án — cyan mà pipeline
dùng cho "thuật ngữ này, ở mọi nơi" — để giao diện và sản phẩm nói cùng một thứ tiếng.

| token | hex | dùng cho |
|---|---|---|
| `--bg` | `#F8FAFC` | nền trang |
| `--surface` | `#FFFFFF` | thẻ, khung, thanh trên |
| `--sunken` | `#F1F5F9` | vùng chìm, hàng đang chọn |
| `--border` | `#E2E8F0` / `--border-strong` `#CBD5E1` | đường kẻ, viền điều khiển |
| `--text` | `#0F172A` | chữ |
| `--soft` `--muted` | `#334155` `#64748B` | chữ phụ, chú thích (đạt 4.5:1 trên `--surface`) |
| `--accent` | `#0F6E7D` (+ `--accent-weak` `#E6F1F3`) | accent duy nhất: đang chạy, xong, lựa chọn đang bật |
| `--patch` | `#B3247A` | dấu vá thuật ngữ |
| `--warn` | `#B45309` | đã dừng, nhắc nhở |
| `--danger` | `#B91C1C` | lỗi |
| `--radius` | `10px` (nhỏ `6px`, hộp thoại `14px`, pill `999px`) | bo góc theo cấp bậc, không tuỳ hứng |

Bảng màu tối đầy đủ được khai báo lại trong `@media (prefers-color-scheme: dark)`.

### 10.2 Chữ — ba vai, không webfont, không CDN

| vai | token | stack | dùng cho |
|---|---|---|---|
| tài liệu | `--font-doc` | `ui-serif, "Iowan Old Style", Cambria, Constantia, Georgia, …` | **cả hai khung đọc**: đây là văn bản duy nhất được đọc dài |
| giao diện | `--font-ui` | `system-ui, -apple-system, "Segoe UI", Roboto, …` | nhãn, nút, tiêu đề, câu giải thích |
| dữ liệu | `--font-data` | `ui-monospace, "Cascadia Mono", Consolas, …` | mọi con số người ta so sánh: số chunk, ký tự, thời gian, mã thoát |

`.mono`, `.count`, `.meta`, `.note`, `.badge`, `.chip` đều `font-variant-numeric: tabular-nums` để
cột số không nhảy. Không webfont, không CDN, không `@import`: **trang phải chạy được offline**, và
điều đó được test bằng cách soi cả HTML lẫn CSS tìm `http://`, `https://`, `//cdn`.

### 10.3 Bố cục

```
┌─────────────────────────────────────────────────────────────────────────┐
│ polyglot-lc                                     [Công cụ] [Dịch mới]    │
├──────────────┬──────────────────────────────────────────────────────────┤
│ LỊCH SỬ      │  <tên tài liệu>                              [đang dịch]  │
│ Tất cả 5 · … │  vi → en · legal · 27 khối · 3 phút trước                │
│ ▸ lượt gần   ├──────────────────────────────────────────────────────────┤
│ ▸ lượt cũ    │  [Tải Markdown] [Tải PDF] [Tải tất cả] [Mở PDF] [Dừng]   │
│              ├──────────────────────────────────────────────────────────┤
│              │  ■■■■■■■■□□□  18 / 27 khối · đã chạy 0:07                 │
│              │  Đang chạy: dịch lại                                     │
│              │  ① ingest ② detect ③ segment ④ plan ⑤ translate …        │
│              ├───────────────────────────┬──────────────────────────────┤
│              │  BẢN GỐC (serif)          │  BẢN DỊCH (serif)            │
│              │  77.160 ký tự · 595 khối  │  1.204 từ                    │
│              │  …                        │  …                            │
└──────────────┴───────────────────────────┴──────────────────────────────┘
```

Cột trái rộng 304px, cuộn riêng. Dưới 900px thì nó thu thành thanh ngang trên cùng và trang
chuyển sang cuộn dọc; dưới 1100px hai khung đọc xếp dọc thay vì cạnh nhau, vì hai cột hẹp thì
việc so sánh — lý do tồn tại của chúng — trở nên vô nghĩa.

### 10.4 Nguyên tắc

1. Không có gì trên trang nói về cấu hình của công cụ; mọi con số đều nói về **tài liệu**.
2. Trạng thái phải đọc được bằng mắt: mỗi trạng thái (`running`/`done`/`failed`/`stopped`) có
   màu, chữ, và vị trí riêng — và **mọi giá trị mà script ghi ra đều có luật trong CSS**. Hợp
   đồng đó được ghi ngay đầu `app.css` và có test canh:
   `.job[data-state=empty|running|done|failed|stopped]`, `.job-row`, `.badge`, `.bar[data-state=…]`,
   `.step[data-state=…]`, `.pane[data-state=…]`, `.alert[data-kind=…]`, `.drop[data-state=…]`,
   `.chip[aria-pressed=true]`, `.view-switch button[aria-pressed=true]`.
3. Chỉ một chuyển động: thanh tiến độ và nhịp thở của khung đang tải. Tắt hết khi
   `prefers-reduced-motion`.
4. Màu mang nghĩa, không trang trí. Không emoji. Không nhãn IN HOA.
5. Nội dung ở thể chủ động, câu chữ thường, tiếng Việt; mọi nút nói việc nó làm ("Dừng",
   "Tiếp tục", "Tải tất cả (.zip)"), và mọi lỗi nói **việc gì đã xảy ra** cùng **việc cần làm**.
6. Không dựng HTML từ văn bản của server, không bao giờ: `textContent` là đường duy nhất. Tên
   file, tên bước và lỗi của provider đều là chữ người dùng hoặc log sinh ra.
7. Mỗi điều khiển có nhãn (`<label for>`), và vùng sống động được đánh dấu
   (`aria-live` trên banner và bộ đếm tiến độ).

### 10.5 Ánh xạ sự kiện → giao diện

| Nguồn | Hiển thị |
|---|---|
| Tập `step` đã thấy trong log | Dải giai đoạn: mỗi tên bước một lần, theo thứ tự xuất hiện lần đầu. Bước không có trong từ điển tiếng Việt của trang được hiển thị bằng chính tên nó |
| `status` = `start` → `running`; `ok`/`patched` → `ok`; `fail` → `fail`; `retry` → `running`; `info` → không đổi trạng thái | Trạng thái của bước **chỉ đi lên**: một `start` sau đó không xoá được `fail` mà bước đã báo |
| `chunk_id` + `status` ∈ {ok, fail, patched} | Đếm vào thanh tiến độ: một chunk đã tới trạng thái cuối thì không quay lại, kể cả khi nó lỗi |
| `status` ∈ {fail, retry} | Vào khối "Vấn đề", kèm `step` và `chunk_id` |
| Lượt chạy hỏng mà không có sự kiện nào | Khối "Vấn đề" lấy `stderr`/`stdout` của tiến trình con làm lời giải thích |
| `note` của bản ghi | Thông báo ngay dưới tên tài liệu (api cảnh báo), ví dụ "đã dừng từ web UI (pid …)" |
| `malformed` > 0 | Banner cảnh báo số dòng log chưa đọc được |
| `has_events == false` **và** lượt chạy đã kết thúc | Banner cảnh báo lượt chạy không ghi log. Lượt đang chạy thì không: nó chỉ chưa mở file log mà thôi |

## 11. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| `probe.py --plan` lỗi (file hỏng, thiếu thư viện, hoặc file input không tồn tại) | `POST /api/runs` trả `422` kèm stderr của probe; **không** khởi động lượt chạy, không tạo thư mục lượt chạy nào |
| Tên file input chứa `..`, `/`, `\` hoặc trỏ ra ngoài `input/` | `403`; tên sai bị từ chối **sau khi `unquote`**, trước khi ghép đường dẫn |
| Upload thứ không phải PDF | `422` kèm "no %PDF- header"; đuôi khác `.pdf` → `403`; thân rỗng → `400`; quá `MAX_UPLOAD_BYTES` → `413` |
| Upload trùng tên | `409`, trang đổi ô kéo thả sang trạng thái "đã có — bấm lại để ghi đè" và chỉ gửi `?overwrite=1` ở lần bấm tiếp theo |
| File log bị ghi dở một dòng | Bỏ qua dòng JSON hỏng, giữ các dòng còn lại, đếm vào `malformed` và nói ra trên banner |
| Server khởi động lại khi một lượt đang chạy | `run.json` có `pid`; nếu pid không còn sống thì `status = "failed"` với ghi chú "server đã khởi động lại khi lượt chạy đang chạy" |
| `LOG_JSONL=0` khiến lượt chạy không ghi log | Bản ghi có ghi chú giải thích; ghi chú **được rút lại** ngay khi file log xuất hiện, để một lượt khoẻ mạnh không mang mãi cảnh báo rằng nó không báo cáo gì |
| `stop` khi tiến trình đã xong | `409`, không kill PID đã được tái sử dụng |
| `probe --text` trên PDF không có lớp chữ | `422` với lý do thật; khung bản gốc chuyển sang trạng thái hỏng và nói ra, thay vì quay mãi |
| File input của một lượt cũ đã bị xoá | `404` kèm tên file; khung bản gốc nói file đã mất |
| `--memory` lỗi (thiếu ChromaDB, ngân hàng hỏng) | `502` kèm lý do; ngăn kéo hiển thị lý do đó |
| Không có kết nối tới server (đã tắt) | Mọi lời gọi `fetch` biến thành câu tiếng Việt "không kết nối được máy chủ" trên banner, không phải "Failed to fetch" |
| Bản ghi `run.json` trỏ ra ngoài repo | `403` khi đọc `output`/`pdf`/`bundle` |

## 12. Kiểm thử

Test nằm ở `tests/` (ràng buộc C5). Không test nào cần API key hay gọi provider.

**`tests/test_webui_paths.py`** — `safe_child`: `..`, `/`, `\`, tên rỗng, tên Unicode. **`tests/test_webui_store.py`** — `run.json` round-trip kể cả bản ghi hỏng; CSV glossary round-trip kể cả file có BOM; `SkillsStore` đọc `skills/<pair>/<domain>.json` và `*.history.jsonl`; không store nào chạm `.env`.

**`tests/test_webui_probe.py`**
- `--plan` trả số chunk bằng `len(chunk_document(load_document(path)))`.
- `--plan` trên file không tồn tại: stdout rỗng, stderr có tên file, exit ≠ 0.

**`tests/test_webui_runs.py`**
- `RunManager` với lệnh giả: lệnh đúng, `LOGS_DIR`/`LOG_JSONL`/`PYTHONIOENCODING` được đặt, hai
  lượt có hai thư mục log khác nhau, `run.json` được ghi.
- Vòng đời `running` → `done`/`failed`, `exit_code`, `stop` → `stopped`, `409` khi đã xong.
- Ghi chú "không có file sự kiện" **bị xoá** khi file log xuất hiện, kể cả khi nó mang
  câu chữ của bản server cũ (bản ghi cũ nằm lại trên đĩa vẫn được dọn).

**`tests/test_webui_server.py`**
- Static: `GET /` trả HTML, `/static/*.css` đúng content-type, đuôi lạ `404`, file thiếu `404` JSON.
- **Không có cấu hình**: `/api/config` đọc/ghi `404`; `.env` không đổi; `api.config is None`.
- **Không response nào chứa tên thiết lập hay tên credential**, trên cả bảy route văn bản.
- Origin: `POST`/`PUT` với origin lạ hoặc thiếu → `403`; cả hai cách viết loopback đều được; `GET`
  không cần Origin.
- Traversal: `../.env`, `a/b`, `..%2F.env`, `..\..\.env`, `a%5Cb`, `%2E%2E%2Fx`, tên rỗng → `403`.
- Upload: chỉ PDF thật được nhận; không ghi gì khi từ chối; `409` rồi `200` với `?overwrite=1`;
  tên có dấu cách và chữ Việt round-trip được.
- Lượt chạy: `POST` → `201`, `format` lạ → `400`, thiếu `file` → `400`, tên escaping → `403`,
  probe lỗi → `422`; `GET /api/runs` **không** có `command`/`mock`/`pid` nhưng có `has_md`/`has_pdf`.
- Artefact: Markdown trả đúng bytes với `text/markdown` và **không** ép tải; tải xuống thì
  `Content-Disposition` có `filename*=UTF-8''…` (tên tiếng Việt không hợp lệ trong header trần);
  đường dẫn ngoài repo → `403`; PDF-only run có `pdf_path` đúng và output `404`; bundle chứa
  **đúng** những file đã có, và `404` khi chưa có gì.
- Nguồn: `/source` lấy từ pipeline, cache lại (test đếm số lần gọi probe), `422` cho PDF không có
  lớp chữ, `403` cho bản ghi có đường dẫn, `404` khi file input đã mất.
- Vấn đề: dựng từ event log; và một lượt chết trước sự kiện đầu thì lời giải thích lấy từ stderr.
- Rò bí mật: một chuỗi canary đặt vào `note`, `stderr` và `error` của sự kiện **không** xuất hiện
  trong JSON của ba route, và `***` xuất hiện ở chỗ nó từng nằm.
- Memory: báo cáo đi qua và được dùng lại; ngân hàng hỏng → `502`.
- Glossary/skills: round-trip, lịch sử điểm, `404` cho skill không tồn tại, tên escaping → `403`.
- **Trang đã phát hành** (đọc chính `webui/static/`): mọi id mà script tìm đều có trong HTML; mọi
  trạng thái script ghi đều có luật CSS và hợp đồng ở đầu sheet; token thiết kế và ba vai chữ có
  mặt; `prefers-reduced-motion` và `prefers-color-scheme: dark` được khai báo; trang không cần gì
  từ mạng; mỗi điều khiển đều có nhãn, `aria-live` được đánh dấu, không trùng id; **không có bề
  mặt lập trình viên** nào trong ba asset; một lựa chọn định dạng duy nhất gồm đúng
  `md|pdf|both` với Markdown là mặc định; script không dùng `innerHTML`/`insertAdjacentHTML`.

**Cách chạy:** `python -m pytest` (toàn bộ suite, hiện 403 test, trong đó 105 thuộc
`tests/test_webui_*.py`), rồi `python -m ruff check .` và `python -m ruff format --check .`
— cả ba đều phải sạch (ruff 0.16 cũng đọc các khối code trong tài liệu `.md`, nên chúng cũng
phải ở đúng định dạng).

## 13. Tiêu chí nghiệm thu

1. `python -m webui.server` mở được UI ở `http://127.0.0.1:8787` mà không cần cài thêm gì.
2. Chạy được một lượt dịch từ UI: chọn file trong `input/`, chọn định dạng, bấm "Bắt đầu dịch",
   thấy thanh tiến độ và dải giai đoạn tiến triển, rồi đọc được bản dịch trong khung bên phải và
   tải về được. *(Đã nghiệm thu bằng `LLM_MOCK=1` với hai tài liệu, gồm một lần chạy `both` cho ra
   Markdown 50 B + PDF 683 kB và một bundle `.zip` chứa đúng hai file đó.)*
3. Dừng được một lượt đang chạy: trạng thái chuyển sang "đã dừng", thanh tiến độ giữ nguyên chỗ đã
   làm, và nút "Tiếp tục" xuất hiện; bấm nó thì lượt mới chạy tiếp và trang đi theo lượt mới.
   *(Đã nghiệm thu trên lượt 27 chunk: dừng ở 19/27, tiếp tục tới 27/27.)*
4. Tải file lên được bằng kéo thả hoặc bấm chọn, có tiến độ, và lần thứ hai với cùng tên thì được
   hỏi trước khi ghi đè. *(Đã nghiệm thu: lần đầu bị từ chối 409 kèm lời đề nghị, lần bấm thứ hai
   ghi đè và file xuất hiện trong `input/`.)*
5. Lịch sử liệt kê cả lượt của lần chạy trước sau khi khởi động lại server, và bộ lọc theo trạng
   thái hoạt động.
6. Ngăn kéo "Công cụ" quản lý được glossary (tạo, sửa, chọn cho lượt mới), xem được bộ nhớ dịch và
   các skill đã học.
7. `python -m ruff check .` sạch và `python -m pytest` xanh.
8. Không có giá trị API key nào ở bất cứ đâu trong response (đã có test), và **không có đường nào**
   để trình duyệt đọc hay ghi cấu hình.
9. Trình duyệt không có lỗi console nào trong suốt các thao tác trên.

## 14. Rủi ro đã biết

- **Polling 700ms × nhiều tab** có thể tạo tải nhỏ lên server khi có job dài; chấp nhận được cho
  một người dùng, và có thể nới khoảng polling ở client nếu cần.
- **Kill cây process khác nhau giữa Windows và POSIX** là chỗ dễ sai; có test riêng cho `stop`.
  Máy đích là Windows nên nhánh đó phải được kiểm tra thật, không chỉ suy luận.
- **`probe --plan` chạy `chunk_document`** tốn thời gian trên PDF rất lớn; chấp nhận vì nó chỉ
  chạy một lần khi bắt đầu lượt (trang hiển thị "Đang đọc tài liệu và lập kế hoạch…" trong lúc đó).
- **`probe --text` trên một cuốn sách tốn hàng chục MB** (cuốn 7 MB cho 77.160 ký tự). Cache 4 mục
  theo `(path, size, mtime)` là đủ cho một người dùng; chưa có giới hạn kích thích trả về.
- **Khung bản dịch hiển thị nguyên văn Markdown**, không render. Render sẽ cần một parser và
  (`innerHTML` bị cấm) một đường dựng DOM khác hẳn; với thời lượng này thì đọc đúng file đã ghi là
  lựa chọn trung thực hơn.
- **Bộ nhớ dịch và skill là điểm ghi chung của cả dự án**, không riêng của UI. Mỗi lượt chạy thử
  (`LLM_MOCK=1`) cũng ghi một trace và có thể cập nhật `skills/<pair>/<domain>`; người kiểm tra
  tay nên biết điều đó trước khi chạy thử nhiều lần trên máy thật.
