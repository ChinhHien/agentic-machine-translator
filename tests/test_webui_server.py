"""Tests for webui.server: routing, static serving, and the API surface."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from webui.runs import RunManager
from webui.server import Api
from webui.store import GlossaryStore, RunsStore

#: A minimal upload body that looks like a PDF, because that is what the route
#: insists on now: anything else is refused before it is written.
PDF_BODY = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nstub\n"


@pytest.fixture()
def api(tmp_path):
    """An Api over a repo-shaped temp dir, with the real static layout."""
    (tmp_path / "webui" / "static").mkdir(parents=True)
    (tmp_path / "webui" / "static" / "index.html").write_text(
        "<!doctype html><title>x</title>", encoding="utf-8"
    )
    (tmp_path / "webui" / "static" / "app.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "webui" / "static" / "secret.py").write_text("nope", encoding="utf-8")
    (tmp_path / ".env").write_text("CHUNK_MAX_TOKENS=700\n", encoding="utf-8")
    return Api(
        root=tmp_path,
        manager=RunManager(tmp_path, python="PY"),
        runs=RunsStore(tmp_path / "logs" / "webui"),
        glossaries=GlossaryStore(tmp_path / "glossaries"),
        port=8787,
    )


def call(api, method, path, **kw):
    from urllib.parse import parse_qs, urlsplit

    parts = urlsplit(path)
    return api.handle(
        method,
        parts.path,
        parse_qs(parts.query),
        kw.get("body", b""),
        {"Origin": "http://127.0.0.1:8787", **kw.get("headers", {})},
    )


def record(tmp_path, run_id="r1", **over):
    """A run record whose output lives under the repository, unless overridden."""
    from webui.store import RunRecord

    out = tmp_path / "output" / "webui" / run_id
    out.mkdir(parents=True, exist_ok=True)
    fields = {
        "run_id": run_id,
        "started": 1.0,
        "status": "done",
        "file": "doc.pdf",
        "src": "vi",
        "tgt": "en",
        "domain": "tech",
        "chunks": 2,
        "exit_code": 0,
        "output_path": str(out / "doc.en.md"),
        "mock": False,
        "pid": None,
        "command": ["x"],
        "note": "",
    }
    fields.update(over)
    return RunRecord(**fields)


def test_index_and_static_are_served(api):
    r = call(api, "GET", "/")
    assert r.status == 200 and r.content_type.startswith("text/html")
    assert b"<title>x</title>" in r.body

    r = call(api, "GET", "/static/app.css")
    assert r.status == 200 and r.content_type.startswith("text/css")
    assert r.body == b"body{}"


def test_static_rejects_non_whitelisted_extensions(api):
    assert call(api, "GET", "/static/secret.py").status == 404


def test_static_serves_nothing_for_a_missing_file(api):
    r = call(api, "GET", "/static/missing.css")
    assert r.status == 404 and r.content_type.startswith("application/json")


def test_unknown_route_is_json_404(api):
    r = call(api, "GET", "/api/unknown")
    assert r.status == 404 and r.content_type.startswith("application/json")


def test_the_real_handler_dispatches_delete(api):
    """`Api.handle` knows DELETE, but the HTTP surface only forwards the methods
    the handler implements. This pins `do_DELETE` so deleting in the browser
    does not come back as an unsupported-method 501."""
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    import webui.server as server_module

    # A Handler subclass with `api` bound at class level, the way the real
    # factory in `make_server` wires it. A JSON 404 (rather than an HTML 501
    # "Unsupported method") is proof the DELETE reached the router at all.
    routed = type("RoutedHandler", (server_module.Handler,), {"api": api})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), routed)
    try:
        thread = threading.Thread(target=httpd.handle_request)
        thread.start()
        req = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_address[1]}/api/runs/nope",
            method="DELETE",
            headers={"Origin": "http://127.0.0.1:8787"},
        )
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        body = json.loads(err.value.read().decode("utf-8"))
        thread.join(timeout=5)
    finally:
        httpd.server_close()
    assert body["error"] == "no such run: nope"


def test_health(api):
    r = call(api, "GET", "/api/health")
    assert r.status == 200 and r.body == b'{"ok": true}'


# --- No configuration surface ----------------------------------------------


def test_there_is_no_configuration_route_at_all(api):
    """The application has no view of `.env`: not a value, not a key name.

    `/api/config` existed to list settings and to write them back into `.env`.
    It is gone, and this test is what keeps it gone — a 404 for reads *and*
    writes is the contract the rest of the UI now depends on.
    """
    assert call(api, "GET", "/api/config").status == 404
    assert (
        call(api, "PUT", "/api/config", body=b'{"values": {"CHUNK_MAX_TOKENS": 900}}').status == 404
    )
    assert call(api, "GET", "/api/config/anything").status == 404
    assert call(api, "POST", "/api/settings", body=b"{}").status == 404

    # Nothing was written, and the router holds no store that could.
    assert (api.root / ".env").read_text(encoding="utf-8") == "CHUNK_MAX_TOKENS=700\n"
    assert getattr(api, "config", None) is None


def test_no_response_mentions_a_setting_or_a_credential_name(api):
    """Even the shape of the configuration stays out of every answer."""
    for path in (
        "/",
        "/api/health",
        "/api/inputs",
        "/api/runs",
        "/api/glossaries",
        "/api/skills",
        "/api/memory",
    ):
        body = call(api, "GET", path).body
        for name in (b"CHUNK_MAX_TOKENS", b"TRANSLATOR_API_KEY", b"EDITABLE", b"dotenv"):
            assert name not in body, (path, name)


# --- Origin, traversal, uploads --------------------------------------------


def test_mutations_require_a_loopback_origin(api):
    for headers in ({"Origin": "http://evil.example"}, {}):
        r = api.handle("PUT", "/api/inputs/a.pdf", {}, PDF_BODY, headers)
        assert r.status == 403, headers


def test_origin_may_be_either_loopback_name(api):
    """Both spellings of loopback are the same page for our purposes.

    Each upload uses its own file name on purpose: re-uploading the same name
    is refused by the overwrite rule, which `test_upload_does_not_overwrite…`
    covers. Asserting 200 twice for one name would contradict that rule.
    """
    pairs = (("http://127.0.0.1:8787", "a.pdf"), ("http://localhost:8787", "b.pdf"))
    for origin, name in pairs:
        r = call(api, "PUT", f"/api/inputs/{name}", body=PDF_BODY, headers={"Origin": origin})
        assert r.status == 200, origin


def test_reads_do_not_need_an_origin(api):
    """A plain GET of the page must work; only mutations are guarded."""
    assert call(api, "GET", "/api/inputs", headers={"Origin": "http://evil.example"}).status == 200


def test_path_traversal_is_rejected(api):
    for bad in ("../.env", "a/b", "..%2F.env"):
        assert call(api, "PUT", f"/api/inputs/{bad}", body=PDF_BODY).status == 403, bad


def test_traversal_variants_are_rejected(api, tmp_path):
    """Windows separators and encoded ones are refused just as firmly."""
    for bad in ("..\\..\\.env", "a%5Cb", "%2E%2E%2Fx", ""):
        r = call(api, "PUT", f"/api/inputs/{bad}", body=PDF_BODY)
        assert r.status == 403, bad
    assert not (tmp_path / ".env").read_bytes().endswith(b"%PDF-1.4")  # .env untouched


def test_upload_validates_what_the_bytes_are(api, tmp_path):
    """Only a real PDF is accepted, and nothing is written when it is not."""
    assert call(api, "PUT", "/api/inputs/notes.txt", body=PDF_BODY).status == 403
    assert call(api, "PUT", "/api/inputs/notes.pdf", body=b"").status == 400
    r = call(api, "PUT", "/api/inputs/notes.pdf", body=b"<html>not a pdf</html>")
    assert r.status == 422 and b"PDF" in r.body
    assert not (tmp_path / "input" / "notes.pdf").exists()

    ok = call(api, "PUT", "/api/inputs/notes.pdf", body=PDF_BODY)
    assert ok.status == 200 and json.loads(ok.body)["size"] == len(PDF_BODY)
    assert (tmp_path / "input" / "notes.pdf").read_bytes() == PDF_BODY


def test_upload_refuses_a_file_over_the_limit(api, monkeypatch):
    """A body larger than the cap is refused *before* it is written."""
    from webui import server

    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 16)
    r = call(api, "PUT", "/api/inputs/big.pdf", body=PDF_BODY + b"x" * 32)
    assert r.status == 413 and b"limit" in r.body


def test_upload_does_not_overwrite_without_consent(api, tmp_path):
    assert call(api, "PUT", "/api/inputs/a.pdf", body=PDF_BODY).status == 200
    assert call(api, "PUT", "/api/inputs/a.pdf", body=PDF_BODY).status == 409
    assert (tmp_path / "input" / "a.pdf").read_bytes() == PDF_BODY
    second = PDF_BODY + b"different\n"
    assert call(api, "PUT", "/api/inputs/a.pdf?overwrite=1", body=second).status == 200
    assert (tmp_path / "input" / "a.pdf").read_bytes() == second


def test_input_names_with_spaces_and_unicode_round_trip(api, tmp_path):
    """Tên có dấu cách / ngoài ASCII vẫn tải lên và đọc lại được."""
    from urllib.parse import quote

    name = "Think Data dữ liệu.pdf"
    r = call(api, "PUT", f"/api/inputs/{quote(name)}", body=PDF_BODY)
    assert r.status == 200
    assert (tmp_path / "input" / name).read_bytes() == PDF_BODY

    names = [f["name"] for f in json.loads(call(api, "GET", "/api/inputs").body)["files"]]
    assert name in names


def test_upload_refuses_a_document_the_run_could_not_read(api, tmp_path):
    """A file the launcher would never list must not be writable either: it
    would look accepted and then fail on the first step of the run."""
    r = call(api, "PUT", "/api/inputs/notes.md", body=b"# hello")
    assert r.status == 403
    assert not (tmp_path / "input" / "notes.md").exists()
    # Path attempts are still refused as path attempts, not as type errors.
    assert call(api, "PUT", "/api/inputs/..%2F.env", body=b"x").status == 403


def test_inputs_listing_reports_size_and_tolerates_a_missing_dir(api, tmp_path):
    assert json.loads(call(api, "GET", "/api/inputs").body)["files"] == []

    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.pdf").write_bytes(PDF_BODY)
    # Not a document this project can translate, so it is not offered.
    (tmp_path / "input" / "notes.md").write_bytes(b"# hello")
    (tmp_path / "input" / "subdir.pdf").mkdir()
    files = json.loads(call(api, "GET", "/api/inputs").body)["files"]
    assert [(f["name"], f["size"]) for f in files] == [("a.pdf", len(PDF_BODY))]
    assert files[0]["mtime"] > 0


# --- Runs ------------------------------------------------------------------


def test_runs_endpoints_with_a_stubbed_manager(api, monkeypatch):
    rec = record(
        api.root,
        file="doc.pdf",
        chunks=3,
        status="running",
        exit_code=None,
        output_path="/tmp/o.md",
        mock=True,
        pid=1,
    )
    api.manager.start = lambda **kw: rec
    api.manager.status = lambda rid: rec
    api.manager.events = lambda rid, after: ([{"step": "plan"}], 1, 0)

    r = call(api, "GET", "/api/runs")
    assert r.status == 200 and json.loads(r.body)["runs"] == []

    r = call(
        api,
        "POST",
        "/api/runs",
        body=json.dumps({"file": "doc.pdf", "src": "vi", "tgt": "en", "domain": "tech"}).encode(),
    )
    assert r.status == 201
    started = json.loads(r.body)
    assert started["run_id"] == "r1" and started["chunks"] == 3

    events = json.loads(call(api, "GET", "/api/runs/r1/events?after=0").body)
    assert events["next"] == 1 and events["status"] == "running" and events["exit_code"] is None
    assert events["malformed"] == 0 and events["has_events"] is False
    assert call(api, "GET", "/api/runs/r1/events?after=nope").status == 400

    detail = json.loads(call(api, "GET", "/api/runs/r1").body)
    assert detail["run"]["file"] == "doc.pdf"
    assert detail["artifacts"] == [] and detail["md"] is False and detail["pdf"] is False


def test_run_summary_carries_download_flags_and_keeps_the_command_back(api):
    """The list has to say what is downloadable; it must not ship the command
    line, which names the interpreter and absolute paths — a developer detail."""
    out = record(api.root, "r9").output_path
    from pathlib import Path

    Path(out).write_text("# bản dịch\n", encoding="utf-8")
    api.runs.write_record(record(api.root, "r9"))

    (row,) = json.loads(call(api, "GET", "/api/runs").body)["runs"]
    assert row["has_md"] is True and row["has_pdf"] is False
    assert "command" not in row and "mock" not in row and "pid" not in row


def test_post_run_passes_every_run_preference_to_the_manager(api):
    """The preferences the page may send are the ones that belong to one run."""
    seen: dict = {}

    def start(**kw):
        seen.update(kw)
        return record(api.root, "r1", status="running", exit_code=None, mock=False)

    api.manager.start = start
    r = call(
        api,
        "POST",
        "/api/runs",
        body=json.dumps(
            {
                "file": "doc.pdf",
                "src": "en",
                "tgt": "vi",
                "domain": "legal",
                "glossary": None,
                "max_tokens": 512,
                "format": "both",
                "resume": True,
            }
        ).encode(),
    )
    assert r.status == 201
    assert seen["output_format"] == "both" and seen["resume"] is True
    assert seen["max_tokens"] == 512 and seen["domain"] == "legal"


def test_post_run_with_a_failing_probe_is_422(api):
    from webui.runs import RunPlanError

    def boom(**kw):
        raise RunPlanError("probe exit 1: không đọc được file")

    api.manager.start = boom
    r = call(
        api,
        "POST",
        "/api/runs",
        body=json.dumps({"file": "doc.pdf", "src": "vi", "tgt": "en", "domain": "tech"}).encode(),
    )
    assert r.status == 422 and b"probe" in r.body


def test_post_run_with_an_escaping_file_name_is_403(api):
    r = call(
        api,
        "POST",
        "/api/runs",
        body=json.dumps(
            {"file": "../../.env", "src": "vi", "tgt": "en", "domain": "tech"}
        ).encode(),
    )
    assert r.status == 403


def test_post_run_without_a_file_is_a_400(api):
    r = call(api, "POST", "/api/runs", body=json.dumps({"src": "vi"}).encode())
    assert r.status == 400


def test_post_run_rejects_an_unknown_format(api):
    r = call(
        api, "POST", "/api/runs", body=json.dumps({"file": "doc.pdf", "format": "docx"}).encode()
    )
    assert r.status == 400 and b"format" in r.body


def test_stopping_a_finished_run_is_a_409(api):
    rec = record(api.root, chunks=1, exit_code=0, output_path="o.md")

    def refuse(rid):
        raise RuntimeError("not running")

    api.manager.stop = refuse
    assert call(api, "POST", "/api/runs/r1/stop").status == 409
    api.manager.stop = lambda rid: rec
    assert call(api, "POST", "/api/runs/r1/stop").status == 200
    assert json.loads(call(api, "POST", "/api/runs/r1/stop").body)["status"] == "done"


def test_deleting_a_run_calls_the_manager_and_requires_an_origin(api):
    """Deleting is a mutation like any other: guarded by Origin, forwarded to
    the manager, and answered with the id it erased."""
    seen: list = []
    api.manager.delete = lambda rid: seen.append(rid)

    refused = api.handle("DELETE", "/api/runs/r1", {}, b"", {})
    assert refused.status == 403
    assert seen == []

    r = call(api, "DELETE", "/api/runs/r1")
    assert r.status == 200 and json.loads(r.body)["deleted"] == "r1"
    assert seen == ["r1"]


def test_deleting_an_unknown_run_is_a_404(api):
    def fail(rid):
        raise KeyError(rid)

    api.manager.delete = fail
    assert call(api, "DELETE", "/api/runs/nope").status == 404


def test_run_endpoints_say_404_for_an_unknown_run(api):
    api.runs = RunsStore(api.root / "logs" / "webui")
    for path in (
        "/api/runs/nope",
        "/api/runs/nope/events",
        "/api/runs/nope/output",
        "/api/runs/nope/pdf",
        "/api/runs/nope/source",
        "/api/runs/nope/bundle",
    ):
        assert call(api, "GET", path).status == 404, path
    assert call(api, "POST", "/api/runs/nope/stop").status == 404


# --- Artefacts -------------------------------------------------------------


def test_run_output_is_served_as_markdown(api, tmp_path):
    out = record(api.root)
    from pathlib import Path

    # Written as bytes on purpose: write_text would translate the newline, and
    # the route must hand back exactly what is on disk.
    Path(out.output_path).write_bytes("# bản dịch\n".encode())
    api.runs.write_record(out)

    r = call(api, "GET", "/api/runs/r1/output")
    assert r.status == 200 and r.content_type.startswith("text/markdown")
    assert r.body == "# bản dịch\n".encode()
    assert r.disposition is None, "viewing must not force a download"


def test_downloads_carry_a_filename_browser_can_read(api):
    """A Vietnamese document name is not valid in a bare header."""
    from pathlib import Path
    from urllib.parse import quote

    out = record(api.root, file="Báo cáo.pdf")
    target = Path(out.output_path).with_name("Báo cáo.en.md")
    target.write_text("# bản dịch\n", encoding="utf-8")
    api.runs.write_record(record(api.root, file="Báo cáo.pdf", output_path=str(target)))

    r = call(api, "GET", "/api/runs/r1/output?download=1")
    assert r.status == 200
    assert "attachment" in r.disposition
    assert quote("Báo cáo.en.md", safe="") in r.disposition
    assert "filename*=UTF-8" in r.disposition


def test_run_output_outside_the_repo_is_refused(api, tmp_path):
    """A hand-edited run.json must not turn this route into a file reader."""
    outside = tmp_path.parent / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    api.runs.write_record(record(api.root, "r2", output_path=str(outside)))
    assert call(api, "GET", "/api/runs/r2/output").status == 403
    assert call(api, "GET", "/api/runs/r2/pdf").status == 403


def test_run_detail_names_the_rebuilt_pdf_and_serves_it(api, tmp_path):
    """A PDF-only run writes no Markdown, so the page cannot find the PDF by
    guessing: the detail has to carry the path the server itself would read."""
    out = record(api.root)
    from pathlib import Path

    pdf = Path(out.output_path).with_suffix(".pdf")
    pdf.write_bytes(b"%PDF-1.4 rebuilt")
    api.runs.write_record(out)

    detail = json.loads(call(api, "GET", "/api/runs/r1").body)
    assert detail["pdf"] is True and detail["md"] is False
    assert detail["pdf_path"] == str(pdf)
    assert detail["artifacts"] == [
        {"kind": "pdf", "name": pdf.name, "size": len(b"%PDF-1.4 rebuilt"), "path": str(pdf)}
    ]

    served = call(api, "GET", "/api/runs/r1/pdf")
    assert served.status == 200 and served.content_type == "application/pdf"
    assert served.body == b"%PDF-1.4 rebuilt"
    # The Markdown of a PDF-only run really is absent: the page has to say so
    # instead of showing the record's path as if the file were there.
    assert call(api, "GET", "/api/runs/r1/output").status == 404


def test_run_bundle_holds_exactly_what_the_run_wrote(api):
    from pathlib import Path

    out = record(api.root)
    # Bytes, not text: the archive has to carry the file exactly, and
    # `write_text` would translate the newline on the way to disk.
    Path(out.output_path).write_bytes("# bản dịch\n".encode())
    Path(out.output_path).with_suffix(".pdf").write_bytes(b"%PDF-1.4 rebuilt")
    api.runs.write_record(out)

    r = call(api, "GET", "/api/runs/r1/bundle")
    assert r.status == 200 and r.content_type == "application/zip"
    assert "attachment" in r.disposition and ".zip" in r.disposition
    with zipfile.ZipFile(io.BytesIO(r.body)) as archive:
        assert sorted(archive.namelist()) == ["doc.en.md", "doc.en.pdf"]
        assert archive.read("doc.en.md") == "# bản dịch\n".encode()
        assert archive.read("doc.en.pdf").startswith(b"%PDF-")


def test_run_bundle_says_so_when_there_is_nothing_yet(api):
    api.runs.write_record(record(api.root, "r3", status="running", exit_code=None))
    r = call(api, "GET", "/api/runs/r3/bundle")
    assert r.status == 404 and b"not written" in r.body


# --- Source text -----------------------------------------------------------


def test_run_source_comes_from_the_pipeline_and_is_cached(api, tmp_path, monkeypatch):
    """The pane a reader compares against has to be the extractor's own text.

    Reading it costs a whole extraction, so the second look at the same file
    must not pay for it again — the test counts the probe calls.
    """
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "doc.pdf").write_bytes(PDF_BODY)
    api.runs.write_record(record(api.root))

    calls: list = []

    def probe(args):
        calls.append(args)
        return {"text": "# Tiêu đề\n\nĐoạn văn.", "chars": 21, "blocks": 2, "title": "doc.pdf"}

    monkeypatch.setattr(api.manager, "probe", probe)

    first = call(api, "GET", "/api/runs/r1/source")
    assert first.status == 200
    payload = json.loads(first.body)
    assert payload["text"].startswith("# Tiêu đề") and payload["cached"] is False
    assert calls and calls[0][0] == "--text"

    second = json.loads(call(api, "GET", "/api/runs/r1/source").body)
    assert second["cached"] is True and len(calls) == 1


def test_run_source_reports_the_probe_reason(api, tmp_path, monkeypatch):
    """A scanned PDF has no text layer; the page must say that, not "error"."""
    from webui.runs import RunPlanError

    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "doc.pdf").write_bytes(PDF_BODY)
    api.runs.write_record(record(api.root))

    def probe(args):
        raise RunPlanError("PDF không có lớp văn bản (không OCR)")

    monkeypatch.setattr(api.manager, "probe", probe)
    r = call(api, "GET", "/api/runs/r1/source")
    assert r.status == 422 and "lớp văn bản".encode() in r.body


def test_run_source_refuses_a_record_that_names_a_path(api):
    api.runs.write_record(record(api.root, "r4", file="../.env"))
    assert call(api, "GET", "/api/runs/r4/source").status == 403


def test_run_source_says_when_the_input_file_is_gone(api, tmp_path):
    (tmp_path / "input").mkdir()
    api.runs.write_record(record(api.root, "r5", file="gone.pdf"))
    r = call(api, "GET", "/api/runs/r5/source")
    assert r.status == 404 and b"gone.pdf" in r.body


# --- Problems and redaction ------------------------------------------------


def test_problems_come_from_the_event_log(api):
    api.runs.write_record(record(api.root, "r1", status="failed", exit_code=1))
    api.manager.events = lambda rid, after: (
        [
            {
                "step": "translate",
                "chunk_id": "chunk-003",
                "status": "fail",
                "error": "provider outage",
            },
            {"step": "verify", "chunk_id": "chunk-001", "status": "retry", "error": "429"},
            {"step": "plan", "chunk_id": "", "status": "ok", "error": ""},
        ],
        3,
        0,
    )

    problems = json.loads(call(api, "GET", "/api/runs/r1").body)["problems"]
    assert [(p["step"], p["chunk_id"], p["message"]) for p in problems] == [
        ("translate", "chunk-003", "provider outage"),
        ("verify", "chunk-001", "429"),
    ]


def test_a_problem_entry_carries_the_cause_the_pipeline_recorded(api):
    """The pipeline's closed-set cause reaches the browser, so the reader learns
    *what kind* of failure it was and not only what the provider said."""
    api.runs.write_record(record(api.root, "r1", status="failed", exit_code=1))
    api.manager.events = lambda rid, after: (
        [
            {
                "step": "translate",
                "chunk_id": "chunk-000",
                "status": "fail",
                "error": "429 too many requests",
                "type": "error",
                "cause": "rate_limit",
            },
            {
                "step": "verify",
                "chunk_id": "chunk-001",
                "status": "fail",
                "error": "dropped marker",
                "cause": "placeholder",
            },
        ],
        2,
        0,
    )

    problems = json.loads(call(api, "GET", "/api/runs/r1").body)["problems"]
    assert [p["cause"] for p in problems] == ["rate_limit", "placeholder"]


def test_a_problem_without_a_cause_reports_an_empty_one(api):
    """An older log has no causes; the field is present and blank, not missing."""
    api.runs.write_record(record(api.root, "r1", status="failed", exit_code=1))
    api.manager.events = lambda rid, after: (
        [{"step": "translate", "chunk_id": "chunk-000", "status": "fail", "error": "boom"}],
        1,
        0,
    )
    problems = json.loads(call(api, "GET", "/api/runs/r1").body)["problems"]
    assert problems[0]["cause"] == "" and problems[0]["message"] == "boom"


def test_the_run_detail_tallies_the_route_mix_from_the_decisions(api):
    api.runs.write_record(record(api.root, "r1"))
    api.manager.events = lambda rid, after: (
        [
            {"type": "decision", "kind": "fast_path", "chunk_id": "c0", "chosen": True},
            {"type": "decision", "kind": "fast_path", "chunk_id": "c1", "chosen": False},
            {"type": "decision", "kind": "refine_gate", "chunk_id": "c1", "chosen": True},
            {"type": "decision", "kind": "fast_path", "chunk_id": "c2", "chosen": False},
            {"type": "decision", "kind": "refine_gate", "chunk_id": "c2", "chosen": False},
        ],
        5,
        0,
    )

    detail = json.loads(call(api, "GET", "/api/runs/r1").body)
    assert detail["routes"] == {"fast": 1, "standard": 1, "deep": 1}


def test_a_run_with_no_decisions_reports_empty_routes(api):
    api.runs.write_record(record(api.root, "r1"))
    api.manager.events = lambda rid, after: ([{"step": "plan", "status": "ok"}], 1, 0)
    assert json.loads(call(api, "GET", "/api/runs/r1").body)["routes"] == {
        "fast": 0,
        "standard": 0,
        "deep": 0,
    }


def test_the_child_is_told_which_run_it_is(api):
    """So the id in ``run.json`` and the id in the trace records are one value."""
    from webui.runs import RunManager

    rec = record(api.root, "r9")
    env = RunManager(api.root, python="PY").env_for(rec)
    assert env["RUN_ID"] == "r9"
    assert env["LOGS_DIR"].endswith(str(rec.run_id))


def test_problems_explain_a_run_that_died_before_any_event(api, tmp_path):
    """A crash at import time writes no event at all; the child's own output is
    the only place that says why the run never started."""
    api.runs.write_record(record(api.root, "r1", status="failed", exit_code=2))
    (api.root / "logs" / "webui" / "r1" / "stderr.log").write_text(
        "Configuration error: TRANSLATOR_API_KEY is not set\n", encoding="utf-8"
    )

    problems = json.loads(call(api, "GET", "/api/runs/r1").body)["problems"]
    assert len(problems) == 1
    assert "Configuration error" in problems[0]["message"]


def test_no_response_ever_echoes_a_credential(api, tmp_path):
    """Provider errors quote keys back at us; the browser must never see one.

    The canary goes in every channel a response quotes — an event error, the
    child's stderr, and the record's own note — and is looked for in the JSON of
    every route that can carry text.
    """
    canary = "sk-CANARY-DO-NOT-LEAK-1234567890"
    api.runs.write_record(
        record(api.root, "r1", status="failed", exit_code=1, note=f"đã dừng; lỗi {canary}")
    )
    (api.root / "logs" / "webui" / "r1" / "stderr.log").write_text(
        f"401 Incorrect API key provided: {canary}\n", encoding="utf-8"
    )
    api.manager.events = lambda rid, after: (
        [
            {
                "step": "translate",
                "chunk_id": "chunk-000",
                "status": "fail",
                "error": f"openai.AuthenticationError: {canary}",
            },
        ],
        1,
        0,
    )

    bodies = [
        call(api, "GET", "/api/runs").body,
        call(api, "GET", "/api/runs/r1").body,
        call(api, "GET", "/api/runs/r1/events?after=0").body,
    ]
    for body in bodies:
        assert canary.encode() not in body
        assert b"***" in body, "the masked marker should be visible where the key was"

    # The record's own note is redacted too, on its way out.
    detail = json.loads(call(api, "GET", "/api/runs/r1").body)
    assert canary not in detail["run"]["note"]
    assert detail["problems"][0]["message"].endswith("***")


# --- The tool drawer's endpoints are gone ----------------------------------


def test_nothing_answers_on_the_drawer_endpoints(api, tmp_path):
    """The page has no tool drawer, so the routes that fed it are gone too.

    What used to be reachable here — the term lists, the memory report and the
    learned skills — is reporting a reader no longer asks for, and the pipeline
    keeps a glossary and a reasoning bank of its own without a screen in front
    of them. A route left behind would be a surface nobody can turn off, so
    each of them has to 404 like any other name the server does not know.
    """
    for method, path in (
        ("GET", "/api/memory"),
        ("GET", "/api/glossaries"),
        ("GET", "/api/glossaries/g.csv"),
        ("PUT", "/api/glossaries/g.csv"),
        ("GET", "/api/skills"),
        ("GET", "/api/skills/en-vi/general"),
    ):
        answer = call(api, method, path, body=b'{"pairs": []}')
        assert answer.status == 404, (method, path, answer.status)
    # A retired route must not have become a way to write anywhere either.
    assert (tmp_path / ".env").read_text(encoding="utf-8").startswith("CHUNK_MAX_TOKENS")


# --- The shipped page ------------------------------------------------------


def repo_api():
    """An Api over the real repository, for tests that read the shipped page."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return Api(
        root=root,
        manager=RunManager(root, python="PY"),
        runs=RunsStore(root / "logs" / "webui"),
        glossaries=GlossaryStore(root / "glossaries"),
        port=8787,
    )


def test_index_has_the_elements_the_frontend_binds_to():
    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    for element_id in (
        "nav-new",
        "banner",
        "jobs",
        "jobs-count",
        "jobs-filter",
        "jobs-empty",
        "job",
        "job-title",
        "job-meta",
        "job-status",
        "job-actions",
        "job-alert",
        "progress",
        "bar",
        "bar-fill",
        "progress-count",
        "progress-elapsed",
        "progress-step",
        "steps",
        "problems",
        "problems-list",
        "panes",
        "pane-source",
        "pane-target",
        "source-text",
        "target-text",
        "source-note",
        "target-note",
        "source-meta",
        "target-meta",
        "view-split",
        "view-source",
        "view-target",
        "new-job",
        "new-job-form",
        "new-job-title",
        "new-job-alert",
        "new-job-start",
        "new-job-cancel",
        "drop",
        "file-input",
        "upload-bar",
        "upload-fill",
        "upload-note",
        "src",
        "tgt",
        "formats",
        "format-md",
        "format-pdf",
        "format-both",
    ):
        assert f'id="{element_id}"' in html, element_id


def test_the_language_choices_name_the_languages_in_full():
    """A reader picks "Tiếng Việt", not "vi".

    The value stays the code the backend expects; the page is where the name in
    words lives. Both ends of the run are dropdowns, so the list of accepted
    codes is the page's own — a typo can no longer be typed in.
    """
    import re

    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    for select_id, chosen in (("src", "auto"), ("tgt", "vi")):
        block = re.search(rf'<select[^>]*id="{select_id}"[^>]*>(.*?)</select>', html, re.DOTALL)
        assert block, select_id
        options = re.findall(r'<option value="([a-z]+)"[^>]*>([^<]+)</option>', block.group(1))
        codes = [code for code, _ in options]
        assert chosen in codes, (select_id, codes)
        assert "en" in codes, "English has to be offered as an input language"
        for code, label in options:
            assert label != code, f"{code} should be spelled out"
        assert re.search(rf'<option value="{chosen}"[^>]*selected', block.group(1))
    # The abbreviation is not what a reader is asked to pick from.
    assert "<datalist" not in html


def test_stylesheet_defines_the_design_tokens():
    """Read the real sheet: the `api` fixture serves a stand-in of its own."""
    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    for token, value in (
        ("--bg", "#F8FAFC"),
        ("--surface", "#FFFFFF"),
        ("--border", "#E2E8F0"),
        ("--text", "#0F172A"),
        ("--muted", "#64748B"),
        ("--accent", "#0F6E7D"),
        ("--patch", "#B3247A"),
        ("--warn", "#B45309"),
        ("--danger", "#B91C1C"),
        ("--radius", "10px"),
    ):
        assert f"{token}: {value}" in css, token


def test_the_page_needs_nothing_from_the_network():
    """The UI must work offline: no webfont, no CDN, no remote asset."""
    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    for remote in ("http://", "https://", "//cdn"):
        assert remote not in html, remote
        assert remote not in css, remote
    assert "@import" not in css


def test_the_three_type_roles_reduced_motion_and_dark_scheme_are_declared():
    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    for role, family in (
        ("--font-doc", "ui-serif"),
        ("--font-ui", "system-ui"),
        ("--font-data", "ui-monospace"),
    ):
        assert f"{role}: {family}" in css, role
    assert "font-variant-numeric: tabular-nums" in css
    assert "prefers-reduced-motion" in css
    assert "prefers-color-scheme: dark" in css


def test_every_control_has_a_label_and_live_regions_are_marked():
    from html.parser import HTMLParser

    class Scan(HTMLParser):
        def __init__(self):
            super().__init__()
            self.controls: list[str] = []
            self.labels: set[str] = set()
            self.live: set[str] = set()
            self.ids: list[str] = []

        def handle_starttag(self, tag, attrs):
            found = dict(attrs)
            if found.get("id"):
                self.ids.append(found["id"])
            if tag in {"input", "select", "textarea"}:
                self.controls.append(found.get("id") or "")
            if tag == "label" and found.get("for"):
                self.labels.add(found["for"])
            if found.get("aria-live"):
                self.live.add(found.get("id") or "")

    scan = Scan()
    scan.feed(call(repo_api(), "GET", "/").body.decode("utf-8"))
    assert scan.controls, "the page should offer controls"
    for control in scan.controls:
        assert control in scan.labels, f"control without a label: {control}"
    assert {"banner", "progress-count"} <= scan.live
    assert len(scan.ids) == len(set(scan.ids)), "duplicate element id"


def test_the_page_has_no_developer_surface():
    """The redesign's whole point: what the product does not offer, it does not
    offer anywhere — no settings screen, no key status, no demo switch, no raw
    step log, and no `.env` name anywhere in the shipped assets."""
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "webui" / "static"
    shipped = {
        name: (static / name).read_text(encoding="utf-8")
        for name in ("index.html", "app.css", "app.js")
    }
    for forbidden in (
        "api/config",
        "Tùy chọn dịch",
        "Chạy thử",
        "API key",
        "/api/inputs?config",
        ".env",
        "LLM_MOCK",
        "EDITABLE",
    ):
        for name, text in shipped.items():
            assert forbidden not in text, (name, forbidden)
    # No inline markup built from server text, ever.
    assert "innerHTML" not in shipped["app.js"]
    assert "insertAdjacentHTML" not in shipped["app.js"]


def test_app_js_never_renders_html_from_server_text():
    """Nội dung đến từ log và tên file người dùng nhập, nên phải là text."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "innerHTML" not in js
    assert "textContent" in js
    assert "insertAdjacentHTML" not in js and "document.write" not in js


def test_app_js_covers_the_steps_the_page_promises():
    """A promise in the UI that no code implements is worse than no promise."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    for name in (
        "pollRun",
        "renderJobs",
        "renderJob",
        "renderProgress",
        "renderPanes",
        "renderProblems",
        "renderBanner",
        "startRun",
        "stopRun",
        "resumeRun",
        "uploadFile",
        "openJob",
    ):
        assert name in js, name
    # The polling endpoint, the artefacts and the source pane all have to be real.
    for route in (
        "/events?after=",
        "/api/runs/",
        "/output",
        "/pdf",
        "/bundle",
        "/source",
    ):
        assert route in js, route
    # ... and the dialog runs the document the reader dropped, so there is no
    # library to read and no preference to carry over between runs.
    assert "/api/glossaries" not in js and "/api/memory" not in js and "/api/skills" not in js


def test_output_format_is_one_choice_of_the_three_the_flag_accepts():
    """The control and `--format` have to name the same three outcomes.

    Two checkboxes also allow a fourth state — neither — which the server can
    only answer with a 400, and they make "both" the accidental result of
    leaving two boxes ticked. One choice, `md`/`pdf`/`both`, has no such state.
    """
    import re
    from pathlib import Path

    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    group = re.search(r'<fieldset[^>]*id="formats"[^>]*>(.*?)</fieldset>', html, re.DOTALL)
    assert group, "the format choice should be one fieldset"
    inputs = re.findall(r"<input[^>]*>", group.group(1))
    assert len(inputs) == 3, inputs
    for tag in inputs:
        assert 'type="radio"' in tag and 'name="format"' in tag, tag
    values = [re.search(r'value="([a-z]+)"', tag).group(1) for tag in inputs]
    assert values == ["md", "pdf", "both"], values
    ticked = [tag for tag in inputs if "checked" in tag]
    assert len(ticked) == 1 and 'value="md"' in ticked[0], "Markdown is the default"

    # The script reads the same three radios. Match on the lookups rather than
    # on the expression, so reformatting the ternary does not fail the check.
    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    for radio in ("format-pdf", "format-both"):
        assert re.search(rf'byId\("{radio}"\)\.checked', js), radio
    assert "Chọn ít nhất một" not in js


def test_every_id_the_script_reaches_for_exists_in_the_page():
    """`byId` returns null for a misspelled id, and the page then fails one
    click at a time instead of once, loudly."""
    import re
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    wanted = set(re.findall(r'byId\("([^"]+)"\)', js))
    assert wanted, "the script should be reaching for elements by id"
    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    for element_id in sorted(wanted):
        assert f'id="{element_id}"' in html, element_id


def test_the_stylesheet_styles_every_state_the_script_sets():
    """A state with no rule is a control that never changes appearance: the
    page looks the same running, finished and broken. Every attribute value the
    script writes has to be reachable in the sheet — ``wait`` and ``idle`` are
    the look of a thing that has not happened yet, so they are the default, but
    they are still named in the contract comment at the top of the sheet."""
    import re
    from pathlib import Path

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    for state in ("running", "done", "failed", "stopped", "empty"):
        assert f'.job[data-state="{state}"]' in css, state
    for state in ("running", "done", "failed", "stopped"):
        assert f'.job-row[data-state="{state}"]' in css, state
        assert f'.badge[data-state="{state}"]' in css, state
    for state in ("running", "ok", "fail"):
        assert f'.bar[data-state="{state}"]' in css, state
    for state in ("running", "ok", "fail"):
        assert f'.step[data-state="{state}"]' in css, state
    for state in ("empty", "loading", "done", "failed"):
        assert f'.pane[data-state="{state}"]' in css, state
    for kind in ("warn", "error", "ok"):
        assert f'.alert[data-kind="{kind}"]' in css, kind
    for state in ("over", "busy", "bad"):
        assert f'.drop[data-state="{state}"]' in css, state
    assert '.chip[aria-pressed="true"]' in css
    assert '.view-switch button[aria-pressed="true"]' in css
    assert ".job[data-state=empty|running|done|failed|stopped]" in css, (
        "the state contract belongs at the top of the sheet"
    )

    # ... and the script has to be the one setting them.
    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    for attribute in ("dataset.state", "dataset.kind"):
        assert attribute in js, attribute
    # Same tolerance as above: the call may be wrapped over several lines.
    assert re.search(r'setAttribute\(\s*"aria-pressed"', js), "aria-pressed"
