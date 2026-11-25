# polyglot-lc

Dịch máy tài liệu PDF dài, giữ nguyên bố cục trang.

Đầu vào là PDF **có lớp văn bản** (không OCR). Đầu ra là Markdown và/hoặc PDF dựng
lại đúng bố cục trang gốc với nội dung đã dịch. Tài liệu được chia theo ngân sách token,
dịch song song nhiều chunk, kiểm định từng chunk và lưu checkpoint, tích hợp cơ chế retry/fallback
với backup endpoint để giữ cho lượt dịch sống.

Dùng được từ CLI hoặc từ một web UI cục bộ. Mọi endpoint đều là API tương thích
OpenAI, cấu hình riêng cho từng vai trò trong pipeline.

---

## Yêu cầu

- Python 3.11 trở lên.
- PDF đầu vào có lớp văn bản. Tài liệu scan ảnh không được hỗ trợ.
- Ít nhất một endpoint chat completion tương thích OpenAI.

## Cài đặt

```bash
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows:      .venv\Scripts\Activate.ps1

pip install -e ".[dev]"        # hoặc: uv pip install -e ".[dev]"
```

## Cấu hình

Sao chép `.env.example` thành `.env`, rồi điền bốn nhóm endpoint(BASE_URL, API_KEY, MODEL): PLANNER, SOURCE_GROUP, VERIFIER, EMBED
Các biến còn lại (chia đoạn, chất lượng, retry/timeout, concurrency, checkpoint,
endpoint dự phòng…) đều có mặc định dùng được ngay; tham khảo đầy đủ ở
[`docs/configuration.md`](docs/configuration.md).

Chạy không cần key:

```bash
LLM_MOCK=1 polyglot translate --file input/handbook.pdf
```

## Sử dụng

```bash
polyglot translate --file input/handbook.pdf
polyglot translate --file report.pdf --out report.vi.md --format both
```

| Tuỳ chọn                 | Mô tả                                                                                  |
| ------------------------ | -------------------------------------------------------------------------------------- |
| `--file / -f`            | File PDF đầu vào (bắt buộc)                                                            |
| `--out / -o`             | Đường dẫn gốc của đầu ra: `<gốc>.md` và/hoặc `<gốc>.pdf` (mặc định `<stem>.vi.md`)     |
| `--format / -F`          | `md` (mặc định) \| `pdf` (dựng lại PDF giữ định dạng) \| `both` (env: `OUTPUT_FORMAT`) |
| `--resume / --no-resume` | Dùng lại checkpoint của lượt chạy bị gián đoạn (env: `RESUME`)                         |
| `--reuse-foreign / --no-reuse-foreign` | Khi resume, dùng lại cả chunk đã dịch bằng model/endpoint khác (mặc định bật) |
| `--checkpoint-dir`       | Nơi lưu checkpoint theo chunk (env: `CHECKPOINT_DIR`)                                  |
| `--run-id`               | Tên của lượt chạy này (mặc định tự sinh)                                               |
| `--quiet / -q`           | Chỉ in báo cáo cuối                                                                    |

Khi xong, CLI in báo cáo token/request, số chunk đã xong, đường dẫn file đầu ra và log JSONL.

# Mã thoát:

- `0` khi mọi chunk đã dịch và xác minh.
- `1` khi thiếu chunk hoặc gặp lỗi nghiêm trọng.
- `2` khi file đã ghi nhưng lượt chạy vẫn.
- `incomplete` sau các vòng quét — supervisor phân biệt được "hỏng" với "cần chạy thêm một lượt".

Một lượt chạy được ghi lại thành JSONL; đọc lại bằng:

```bash
polyglot trace           # lượt mới nhất dưới logs/
polyglot trace run_id    # theo run id
```

## Giao diện web

```bash
python -m webui.server               # rồi mở http://127.0.0.1:8787
python -m webui.server --port 9000   # đổi cổng, vẫn chỉ bind loopback
```

# Thao tác chính:

- Config endpoint: Cấu hình endpoint cho các lượt mới nằm trong hộp thoại ở góc phải header;
  nó ghi đè `.env`.
- Dịch văn bản: Chọn file PDF đầu vào và định dạng văn đầu ra, rồi bấm **Bắt đầu dịch**.
  Trang hiện tiến độ theo khối, cho phép tải văn bản nguồn được trích xuất(.md),
  bản dịch(.md) và PDF. Hiển thị tiến độ, trạng thái và báo cáo của lượt dịch trên webui.
  Cho phép người dùng dừng và chạy tiếp lượt dịch.

## Kiểm thử

```bash
pytest -q                        # toàn bộ suite, không cần API key
python -m ruff check .           # lint
python -m ruff format --check .  # định dạng
```

Test dùng HTTP giả lập và PDF tự dựng trong `tests/pdfdoc.py`; không test nào mở
cổng hay gọi provider.

## Cấu trúc dự án

```
src/polyglot_lc/
├── cli.py                 # CLI typer + báo cáo token/request
├── config.py              # cấu hình từng agent, chính sách cache, các knob của workflow
├── ingest/                # đọc PDF, lọc code, toạ độ và kiểu chữ của từng block
├── segment/               # chunker, chia mục, glossary
├── analysis/              # gọi planner, chấm độ phức tạp
├── agents/                # planner, translator, critic, refiner, verifier, consistency
├── orchestrator/          # LangGraph state, router, graph
├── pipeline/              # coordinator, assembler, pdf_output, checkpoint, consistency
├── reasoning_bank/        # trace bền vững (ChromaDB)
├── skillopt/              # tối ưu prompt dịch qua các lần chạy
├── llm/                   # client tương thích OpenAI: retry, timeout, hàng đợi, cache
└── ui/                    # báo cáo bước + event sink JSONL

webui/                     # web UI cục bộ (chạy polyglot_lc như tiến trình con)
├── server.py              # HTTP server: routing, static, API
├── runs.py                # vòng đời lượt chạy: spawn, trạng thái, stop, resume
├── store.py               # run.json, skills, glossaries
└── static/                # index.html, app.css, app.js
```

## Tài liệu

| Tài liệu                                         | Nội dung                                                                                      |
| [`docs/configuration.md`](docs/configuration.md) | Toàn bộ biến môi trường, mặc định và khi nào nên đổi; công thức cho các tình huống thường gặp |
| [`docs/features.md`](docs/features.md)           | Kiến trúc pipeline, hợp đồng giữa các giai đoạn, cơ chế chất lượng và các giới hạn đã biết    |

## Giới hạn

- Không hỗ trợ PDF OCR ở đầu vào.
- PDF dựng lại ở mức đang phát triển, định dạng đầu ra có thể không chính xác.
- Kết quả của quy trình dịch bị giới hạn bởi mô hình LLM được sử dụng,
  chất lượng và chi phí sử dụng đối với từng mô hình chưa được đánh giá khách quan.
- Chưa có cơ chế quản lý dữ liệu phiên người dùng do hệ thống chỉ chạy cục bộ.
