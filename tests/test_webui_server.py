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


def test_a_second_server_will_not_share_the_port(api):
    """Two copies of the server must never listen on one port.

    On Windows the default ``SO_REUSEADDR`` let a second process bind a port a
    server was already listening on, so requests split between an old build and
    a new one — which is how a fixed route kept answering with the old code
    while the browser had the new page. The second start has to fail instead.
    """
    from webui.server import make_server

    first = make_server(api, 0)
    try:
        with pytest.raises(OSError):
            make_server(api, first.server_address[1])
    finally:
        first.server_close()


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

    A re-upload of the same name now overwrites in place (see
    ``test_upload_overwrites_a_previous_version``), so re-using a name is safe
    and still must not be refused.
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


def test_upload_overwrites_a_previous_version(api, tmp_path):
    """A repeat upload of the same document refreshes the bytes in place: the
    server overwrites, so there is no separate consent step and no 409."""
    assert call(api, "PUT", "/api/inputs/a.pdf", body=PDF_BODY).status == 200
    assert (tmp_path / "input" / "a.pdf").read_bytes() == PDF_BODY
    second = PDF_BODY + b"different\n"
    assert call(api, "PUT", "/api/inputs/a.pdf", body=second).status == 200
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
        body=json.dumps({"file": "doc.pdf", "src": "vi", "tgt": "en"}).encode(),
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
                "glossary": None,
                "max_tokens": 512,
                "format": "both",
            }
        ).encode(),
    )
    assert r.status == 201
    assert seen["output_format"] == "both"
    assert seen["max_tokens"] == 512


def test_post_run_with_a_failing_probe_is_422(api):
    from webui.runs import RunPlanError

    def boom(**kw):
        raise RunPlanError("probe exit 1: không đọc được file")

    api.manager.start = boom
    r = call(
        api,
        "POST",
        "/api/runs",
        body=json.dumps({"file": "doc.pdf", "src": "vi", "tgt": "en"}).encode(),
    )
    assert r.status == 422 and b"probe" in r.body


def test_post_run_with_an_escaping_file_name_is_403(api):
    r = call(
        api,
        "POST",
        "/api/runs",
        body=json.dumps(
            {"file": "../../.env", "src": "vi", "tgt": "en"}
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


def test_run_detail_reports_the_translator_model_the_run_used(api, tmp_path):
    """The detail carries the model from the child's own `run_start` event, so
    a stopped run's "Tiếp tục" can pin it and keep the checkpoint identity.
    """
    api.runs.write_record(record(api.root, "r1", status="stopped"))
    log_dir = api.root / "logs" / "webui" / "r1"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run_20260101_000000.jsonl").write_text(
        json.dumps(
            {
                "type": "run_start",
                "agents": [
                    {"name": "planner", "model": "gpt-4o-mini"},
                    {"name": "translator", "model": "gpt-4o"},
                    {"name": "critic", "model": "gpt-4o"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    detail = json.loads(call(api, "GET", "/api/runs/r1").body)
    assert detail["translator_model"] == "gpt-4o"

    # A run with no log yet has no answer: the field is present and empty.
    api.runs.write_record(record(api.root, "r2", status="running", exit_code=None))
    empty = json.loads(call(api, "GET", "/api/runs/r2").body)
    assert empty["translator_model"] == ""


def test_post_run_to_resume_continues_that_run(api):
    """ "Tiếp tục" carries the run being continued and nothing else.

    Its document, languages, glossary, token budget, output format and
    Translator model are all stored on that run's own record, so there is no
    second set of settings to drift out of step with the checkpoints. The answer
    names the same run: the page has nowhere to jump to.
    """
    seen: dict = {}

    def resume(run_id):
        seen["run_id"] = run_id
        return record(api.root, run_id, status="running", exit_code=None)

    api.manager.resume = resume
    r = call(
        api,
        "POST",
        "/api/runs",
        body=json.dumps({"resume": True, "run_id": "r1"}).encode(),
    )
    assert r.status == 200  # nothing was created, only continued
    assert seen["run_id"] == "r1"
    assert json.loads(r.body)["run_id"] == "r1"


def test_post_run_to_resume_needs_the_run_to_continue(api):
    """Resuming without naming the run would leave the server guessing which
    checkpoints to continue; the page always knows, so it must say."""
    r = call(
        api,
        "POST",
        "/api/runs",
        body=json.dumps({"resume": True, "file": "doc.pdf"}).encode(),
    )
    assert r.status == 400 and b"run_id" in r.body


def test_post_run_to_resume_maps_a_missing_run_and_a_running_one(api):
    def missing(run_id):
        raise KeyError(run_id)

    api.manager.resume = missing
    r = call(api, "POST", "/api/runs", body=json.dumps({"resume": True, "run_id": "nope"}).encode())
    assert r.status == 404 and b"nope" in r.body

    def still_running(run_id):
        raise RuntimeError(f"run {run_id} is still running")

    api.manager.resume = still_running
    r = call(api, "POST", "/api/runs", body=json.dumps({"resume": True, "run_id": "r1"}).encode())
    assert r.status == 409 and b"still running" in r.body


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


def test_run_source_downloads_as_markdown_in_both_shapes(api, tmp_path, monkeypatch):
    """The source column can be kept as a file, not only read as a pane.

    Two shapes, because the extraction carries machinery a reader does not want
    to see: the text the pipeline wrote (markers and all), and the same text
    with the emphasis said the Markdown way and the code markers gone. The name
    of each says which it is, so the two do not land in a folder looking
    identical.
    """
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "doc.pdf").write_bytes(PDF_BODY)
    api.runs.write_record(record(api.root))
    extracted = (
        "Đoạn văn ⟦BOLD⟧đậm⟦/BOLD⟧ và ⟦ITALIC⟧nghiêng⟦/ITALIC⟧.\n\n"
        "⟦CODE_1⟧\n\n"
        "Đoạn tiếp ⟦BOLD⟧lẻ"
    )
    monkeypatch.setattr(
        api.manager,
        "probe",
        lambda args: {"text": extracted, "chars": len(extracted), "blocks": 3, "title": "doc.pdf"},
    )

    kept = call(api, "GET", "/api/runs/r1/source?download=1&markers=keep")
    assert kept.status == 200 and kept.content_type.startswith("text/markdown")
    assert "attachment" in kept.disposition and "doc.goc.md" in kept.disposition
    assert kept.body.decode("utf-8") == extracted

    plain = call(api, "GET", "/api/runs/r1/source?download=1&markers=strip")
    assert plain.status == 200 and "doc.goc-sach.md" in plain.disposition
    # A pair becomes the Markdown it stood for; the code marker and an opener
    # with no closer leave nothing behind, since neither has text of its own.
    assert plain.body.decode("utf-8") == (
        "Đoạn văn **đậm** và *nghiêng*.\n\n\n\nĐoạn tiếp lẻ"
    )

    # A download that names no shape is the readable one, never the machinery:
    # the markers are what this surface exists to hide.
    assert call(api, "GET", "/api/runs/r1/source?download=1").body == plain.body
    # And the pane still gets its JSON, which is the only consumer of the rest.
    payload = json.loads(call(api, "GET", "/api/runs/r1/source").body)
    assert payload["text"] == extracted and payload["cached"] is True


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


# --- The translation so far -------------------------------------------------


def _checkpoints(tmp_path, chunks, key="abc123"):
    """A checkpoint directory laid out the way the pipeline writes it."""
    directory = tmp_path / "data" / "checkpoints" / key
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps({"file": "doc.pdf"}), encoding="utf-8")
    for chunk_id, text in chunks.items():
        (directory / f"{chunk_id}.json").write_text(
            json.dumps({"chunk_id": chunk_id, "chunk_hash": "h", "translated": text}),
            encoding="utf-8",
        )
    return directory


def _names_checkpoints(api, path):
    """The run's own early record naming where its chunks are written."""
    api.manager.events = lambda rid, after: (
        [{"type": "checkpoint", "doc_key": "abc123", "path": str(path)}],
        1,
        0,
    )


def test_the_translation_so_far_comes_out_of_the_runs_own_chunks(api, tmp_path):
    """A run writes each finished chunk's own text as that chunk completes, so
    the column a reader compares against can fill in behind the run instead of
    an empty page that promises something later. It names the directory in its
    own log, because that directory is the pipeline's configuration and the page
    is not allowed to go looking for it."""
    api.runs.write_record(record(api.root))
    _names_checkpoints(
        api,
        _checkpoints(tmp_path, {"chunk-000": "Câu một.", "chunk-001": "Câu hai."}),
    )
    payload = json.loads(call(api, "GET", "/api/runs/r1/partial").body)
    assert payload["pieces"] == 2
    assert payload["text"] == "Câu một.\n\nCâu hai."


def test_chunks_land_out_of_order_and_come_back_in_the_documents_order(api, tmp_path):
    """Chunks run concurrently, so their files appear in whatever order they
    finish; an id is a padded number, and read as plain text `chunk-1000` sorts
    before `chunk-999` — a scrambled document in exactly the thousand-chunk
    books a reader sits and watches fill in."""
    api.runs.write_record(record(api.root))
    directory = _checkpoints(
        tmp_path,
        {
            "chunk-1000": "cuối cùng",
            "chunk-999": "gần cuối",
            "chunk-000": "đầu tiên",
            "chunk-501": "",  # written but empty: a chunk still to be translated
        },
    )
    # Named the way the pipeline names it: relative to the repository, because
    # the child ran with the repository as its working directory.
    _names_checkpoints(api, directory.relative_to(api.root))
    payload = json.loads(call(api, "GET", "/api/runs/r1/partial").body)
    assert payload["pieces"] == 3
    assert payload["text"] == "đầu tiên\n\ngần cuối\n\ncuối cùng"


def test_a_run_with_nothing_translated_yet_says_so_plainly(api):
    """No checkpoint pointer is not a fault: a run that keeps no checkpoints,
    or has not finished its first chunk, has nothing to show — and an unknown
    run is still a 404."""
    api.runs.write_record(record(api.root))
    api.manager.events = lambda rid, after: ([], 0, 0)
    assert json.loads(call(api, "GET", "/api/runs/r1/partial").body) == {
        "pieces": 0,
        "text": "",
    }
    assert call(api, "GET", "/api/runs/nope/partial").status == 404


def test_a_checkpoint_directory_outside_the_repository_is_refused(api, tmp_path):
    """The directory comes out of a log file, and a path recorded in a log is a
    pointer — never permission to read whatever it happens to name."""
    outside = tmp_path.parent / "checkpoints-elsewhere"
    outside.mkdir(parents=True, exist_ok=True)
    (outside / "chunk-000.json").write_text(
        json.dumps({"chunk_id": "chunk-000", "translated": "bí mật"}), encoding="utf-8"
    )
    api.runs.write_record(record(api.root))
    _names_checkpoints(api, outside)
    r = call(api, "GET", "/api/runs/r1/partial")
    assert r.status == 200
    assert json.loads(r.body) == {"pieces": 0, "text": ""}
    assert "bí mật".encode() not in r.body


def test_the_page_shows_the_translation_while_the_run_is_still_writing_it():
    """The pane is redrawn from the run's own chunks, and asked for only when
    the number of finished chunks moves: watching a hundred chunks must not turn
    into a request on every poll, and a run whose whole translation is written
    already has nothing partial to show."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "async function refreshPartial" in js
    assert "/partial`" in js
    assert "if (done === state.partialDone)" in js
    assert "if (state.partialText)" in js
    # A column that is filling in grows at the end: redrawing it from the top
    # would throw the reader's place away with every chunk that lands.
    assert "shown.startsWith(text.textContent)" in js
    # A checkpoint pointer is not a line of work: the log draws the run's work.
    assert 'kind === "decision" || kind === "checkpoint"' in js


def test_the_page_counts_chunks_from_one_for_the_reader():
    """The pipeline counts chunks from zero; the page counts them from one.

    A reader counts pages from one, and a list that starts at `chunk-000`
    reads as an off-by-one bug rather than as a name. So the web UI renumbers
    what it shows — the list, the tooltip, the log lines, the caption under the
    progress bar and the report — while the ids the pipeline speaks (the log
    file, the checkpoints) stay exactly as they are.
    """
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "function displayChunkId(" in js
    assert "function displayChunkText(" in js
    # Every place a chunk is named to the reader goes through the renumbering.
    assert 'element("span", "chunk-id", displayChunkId(cell.id))' in js
    assert "parts = [displayChunkId(cell.id)]" in js
    assert "chunk: displayChunkId(event.chunk_id)" in js
    assert "displayChunkId(last.chunk_id)" in js
    assert "displayChunkId(entry.id)" in js
    # A message the pipeline wrote with the raw id ("chunk-000: ...") is
    # renumbered too, or the row would show two numbers for one chunk.
    assert "displayChunkText(cell.error)" in js
    assert "displayChunkText(" in js
    # The ids themselves are untouched: the cells are still keyed by the
    # pipeline's own `chunk-000`, so the events still match them.
    assert 'const id = `chunk-${String(i).padStart(3, "0")}`;' in js


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


def test_the_events_endpoint_passes_resume_records_through(api):
    """The web UI's progress bar folds the replayed chunk count in from the
    pipeline's early `resume` record; the endpoint must ship it intact, not
    filter it out like an unknown type."""
    api.runs.write_record(record(api.root, "r1", status="running", exit_code=None))
    api.manager.events = lambda rid, after: (
        [
            {"type": "run_start", "file": "doc.pdf", "chunks": 4},
            {
                "type": "resume",
                "resume_requested": True,
                "resumed_chunks": 3,
                "checkpoint_usable": True,
            },
            {"step": "translate", "chunk_id": "chunk-003", "status": "ok"},
        ],
        3,
        0,
    )
    events = json.loads(call(api, "GET", "/api/runs/r1/events?after=0").body)
    assert [(e.get("type"), e.get("step")) for e in events["events"]] == [
        ("run_start", None),
        ("resume", None),
        (None, "translate"),
    ]
    resume_events = [e for e in events["events"] if e.get("type") == "resume"]
    assert resume_events[0]["resumed_chunks"] == 3
    assert resume_events[0]["resume_requested"] is True
    assert resume_events[0]["checkpoint_usable"] is True


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


def test_the_page_opens_under_the_mark_and_hands_it_over():
    """The first paint, and the three things that take it away.

    The splash covers a page that does not exist yet, so it has to be in the
    markup before the script runs; the sheet has to be able to take it away on
    its own, because a script that never arrives would otherwise leave it sitting
    there for good; and the script hands it over when the first list of runs has
    landed, not when the files did.
    """
    import re
    from pathlib import Path

    r = call(repo_api(), "GET", "/static/logo.png")
    assert r.status == 200 and r.content_type == "image/png"
    assert r.body.startswith(b"\x89PNG\r\n\x1a\n"), "the mark has to be a real PNG"

    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    splash = re.search(r'<div id="preloader"[^>]*>(.*?)\n</div>', html, re.DOTALL)
    assert splash, "the splash belongs in the markup, before anything runs"
    assert "/static/logo.png" in splash.group(1)
    assert "<noscript><style>#preloader { display: none; }</style></noscript>" in html, (
        "with no script there is nothing to hand the splash over"
    )

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    assert re.search(r"\.preloader \{[^}]*transition: opacity", css)
    assert ".preloader--leaving { opacity: 0; }" in css
    assert "preloader-timeout" in css, "the sheet must be able to take it away alone"

    # The mark goes with the light, not on a timer beside it: the light's own
    # duration and easing are what the dissolve runs on, so the mark holds while
    # the light is above it, decays as the light crosses it, and is gone by the
    # time the light is under it — in a window of any height.
    assert "preloader-scan 1.6s cubic-bezier(0.4, 0, 0.2, 1)" in css
    assert re.search(r"\.preloader-logo\.scanning \{[^}]*preloader-mark-swept", css)
    assert "preloader-mark-swept 1.6s cubic-bezier(0.4, 0, 0.2, 1)" in css
    swept = css[css.index("@keyframes preloader-mark-swept {") : css.index(".preloader-logo-img")]
    # Centred on the crossing: the mark's own middle is the stage's middle, and
    # the sweep's easing has the light there at 35% of the sweep.
    assert "0%, 20% { opacity: 1; }" in swept
    assert "50%, 100% { opacity: 0; }" in swept
    # What this replaced: a fade over the whole sweep, which had the mark half
    # gone while the light was still over empty stage.
    assert "transition: opacity 1.6s linear" not in css

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    # The mark and the light start on the same step, or the two share nothing.
    assert re.search(
        r'line\.classList\.add\("active"\)[\s\S]*?logo\.classList\.add\("scanning"\)', js
    ), "the mark's dissolve has to start with the sweep"
    assert re.search(r"function hideSplash", js) and "SPLASH_HOLD_MS" in js
    assert re.search(r"wait\(SPLASH_HOLD_MS\)", js), (
        "the mark must hold on the stage's own clock before the fade"
    )
    assert re.search(r"loadRuns\(\)", js), (
        "the first list must still load under the splash"
    )


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


def test_a_notice_dismisses_itself_and_can_be_waved_away():
    """A notice is a passing message, not a permanent fixture: it fades out on
    its own — sooner for a plain confirmation than for a failure — or leaves at
    once when clicked, instead of squatting on the screen for the whole session."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "BANNER_MS" in js and "bannerTimer" in js and "bannerFade" in js
    assert "setTimeout(hideBanner" in js, "the bar has to schedule its own exit"
    assert 'byId("banner").addEventListener("click", hideBanner)' in js

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    assert "transition: opacity" in css
    assert ".banner--leaving { opacity: 0; }" in css


def test_delete_drops_the_row_at_once_and_does_not_lean_on_the_dialog_close_event():
    """What the reader saw: clicking "Xóa" left the run on screen until they
    reloaded the page themselves. Two things had to hold for that, and both are
    pinned here.

    The verdict used to be read off the dialog's `close` event alone, and the
    webview this page is opened in closes a `method="dialog"` submit without
    ever dispatching `close` (the dialog shuts, `returnValue` is set, nothing
    fires). The promise then never resolved, so the DELETE was never sent at
    all — the trash looked dead. The verdict now comes from the dialog's own
    `submit` and `cancel`, which do arrive, with the browser's own close
    prevented so the dialog shuts on the event that decides the answer.

    And the erasing no longer ends in `location.reload()`: the run leaves the
    list on the spot once the server has answered. A reload was the previous
    answer to this same complaint, and it cannot be reached at all while the
    verdict above is pending — a page that reloads itself to erase a row is
    also asking the reader to lose their place for it."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    confirm = js[js.index("function confirmDelete") : js.index("\nasync function deleteRun")]
    assert 'dlg.addEventListener("submit"' in confirm
    assert 'dlg.addEventListener("cancel"' in confirm
    assert "event.preventDefault()" in confirm, "ours closes it, not the host"
    assert "event.submitter" in confirm, "the button that submitted decides the answer"

    fence = js.index("async function deleteRun")
    body = js[fence : js.index("\n}\n", fence)]
    # No page reload: the row goes, and the page stays where the reader is.
    assert "location.reload()" not in body
    # In that order: the guard, the erase, then the row.
    assert body.index("await confirmDelete()") < body.index('request("DELETE"')
    erased = body.index("clearTimeout(state.timer)")
    assert body.index('request("DELETE"') < erased
    assert "state.runs = state.runs.filter" in body[erased:]
    assert "renderJobs()" in body[erased:]
    # The list is then read back from the server, so the row is gone from the
    # page and from the server's own answer alike.
    assert "await loadRuns()" in body[erased:]
    # The failure path returns before anything is dropped: a delete that did
    # not happen leaves the row exactly where it was, message and all.
    failed = body[body.index("catch (error)") : erased]
    assert "return;" in failed
    # And the panel that held the erased run is emptied rather than left
    # showing a run that no longer exists.
    assert "renderEmptyJob()" in body


def test_app_js_never_renders_html_from_server_text():
    """Nội dung đến từ log và tên file người dùng nhập, nên phải là text."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "innerHTML" not in js
    assert "textContent" in js
    assert "insertAdjacentHTML" not in js and "document.write" not in js


def test_app_js_poll_run_does_not_shadow_a_top_level_helper():
    """The live poll loop used to declare `const resumeInfo = resumeInfo(...)` inside
    `pollRun`. Because a function declaration and a const in the same scope
    cannot coexist under the same name, the const's temporal dead zone made
    `pollRun` throw a `ReferenceError` on the very line it read the resume fact,
    killing the polling loop: the progress bar and step rail then froze and the
    UI looked "stuck / not starting" while the run was in fact translating.

    Pin the fix: inside `pollRun` no `const`/`let`/`var` may redeclare a name
    that is also a top-level `function` in the file.
    """
    import re
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    # Top-level helper names: `function name(` or `async function name(` at col 0.
    top_level = set(re.findall(r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\b", js, re.MULTILINE))
    # The pollRun body, from its declaration to the next top-level function.
    m = re.search(
        r"^async function pollRun\(.*?\n(?:^async function |^function )",
        js,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "pollRun not found in app.js"
    body = m.group(0)
    # Any local `const|let|var X` whose X is also a top-level function name is a
    # redeclaration that shadows the helper within this scope (the bug).
    local_names = re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\b", body)
    offenders = [n for n in local_names if n in top_level]
    assert not offenders, f"pollRun shadows top-level helpers: {offenders}"


def test_the_status_tab_shows_every_chunk():
    """A separate detail tab lists the whole task — one row per chunk, carrying
    its state and the work it is on — because a long run's progress bar cannot
    say *which* chunks went wrong or what they were doing. The list is derived
    from the events the page already polls, so it needs no route of its own."""
    from pathlib import Path

    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    for element_id in (
        "job-tabs",
        "tab-result",
        "tab-status",
        "tab-panel-result",
        "tab-panel-status",
        "chunks",
        "chunks-summary",
    ):
        assert f'id="{element_id}"' in html, element_id
    # A real tablist, with each panel named by the tab that opens it.
    assert 'id="job-tabs" class="tabs" role="tablist"' in html
    assert 'aria-controls="tab-panel-status"' in html
    assert 'id="tab-panel-status" class="tab-panel tab-panel--list" role="tabpanel"' in html
    # A list of chunks, not a table of cells: each row is an item.
    assert '<ol id="chunks" class="chunks"' in html

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    for name in ("chunkStates", "chunkWork", "renderChunks", "renderTabs"):
        assert f"function {name}" in js, name
    # Each row shows its chunk's state *and* what that chunk is doing.
    for field in ("chunk-id", "chunk-state", "chunk-work"):
        assert f'"{field}"' in js, field
    assert 'element("li", "chunk")' in js
    # The chunker names its chunks from zero, so the list must count from zero
    # too: one-based ids hide `chunk-000` and invent a chunk past the end.
    assert "for (let i = 0; i < count; i += 1)" in js

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    for state in ("wait", "running", "ok", "fail"):
        assert f'.chunk[data-state="{state}"]' in css, state
    # The rows are styled as a list: id, state and work share one line.
    for field in ("chunk-id", "chunk-state", "chunk-work"):
        assert f".{field}" in css, field
    # The list follows the chunk the run is on, and stops following the moment
    # the reader scrolls somewhere else: their place wins, and scrolling back to
    # the run starts it again.
    assert "function followRunning" in js
    assert 'scrollIntoView({ block: "nearest" })' in js
    assert 'byId("chunks").addEventListener("scroll"' in js
    # A chunk with a message opens it in full — by pointer and by keyboard — and
    # the list can be narrowed to the chunks that went wrong. Rebuilding the
    # rows must not throw the reader back to the top of a hundred chunks.
    assert "function toggleChunkError" in js
    assert "function chunksSummary" in js
    assert '"chunk-error"' in js
    assert 'role", "button"' in js
    assert "list.scrollTop = scroll" in js
    for element_id in ("chunk-filter", "chunk-filter-all", "chunk-filter-fail"):
        assert f'id="{element_id}"' in html, element_id
    assert ".chunk-error" in css


def test_the_status_tabs_scroll_their_list_and_not_themselves():
    """Two scrollbars were drawn side by side in the status and history tabs:
    the tab's own, for the few pixels its caption and filters pushed past the
    window, and the list's, for the run's own length. The list is the part that
    is long, so it is the part that scrolls — the tab holds its counts and its
    filters still and draws no bar of its own. The list keeps its own ceiling on
    a phone, where the panel is a block and the page is what scrolls."""
    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    for panel in ("tab-panel-status", "tab-panel-history"):
        assert f'id="{panel}" class="tab-panel tab-panel--list" role="tabpanel"' in html, panel

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    tab = css.split(".tab-panel--list {", 1)[1].split("}", 1)[0]
    assert "display: flex" in tab and "flex-direction: column" in tab
    assert "overflow: hidden" in tab
    # The list shrinks inside the panel and scrolls there, rather than pushing
    # the panel taller: a default `min-height: auto` would keep it at its full
    # length and bring the second scrollbar back.
    start = css.index(".tab-panel--list .chunks")
    lists = css[start : css.index("}", start)]
    assert "min-height: 0" in lists
    assert "max-height: none" in lists
    mobile = css.split("@media (max-width: 767px) {", 1)[1]
    assert (
        ".chunks, .log, .tab-panel--list .chunks, .tab-panel--list .log { max-height: 60vh; }"
        in mobile
    )


def test_the_log_tab_carries_the_whole_run():
    """The run's log lives beside the result and the chunk list, and it is the
    run's own order that it keeps: a chunk that stumbled and then finished shows
    both lines, so a stumble is not read as that chunk's verdict for the rest of
    the run. The card it replaced could only ever list what went wrong."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    for element_id in ("tab-history", "tab-panel-history", "history"):
        assert f'id="{element_id}"' in html, element_id
    assert 'aria-controls="tab-panel-history"' in html
    assert '<ol id="history" class="log"' in html
    # The card this replaced is gone, not merely hidden.
    assert 'id="problems"' not in html
    assert 'id="problems-list"' not in html

    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    for name in ("logEntry", "logRow", "renderLog", "resetLog"):
        assert f"function {name}" in js, name
    assert "LOG_STATUS_WORDS" in js
    # Written and drawn a poll at a time: a few thousand events must not be
    # rebuilt into nodes on every tick.
    assert "for (let i = state.logged; i < state.events.length;" in js
    # Every event reaches the log. The exceptions are a routing decision,
    # already summarised in "Đường đi", and the `checkpoint` pointer this page
    # reads to find a run's chunks — and nothing filters out a failure or a
    # chunk's later success, which is what kept the old card showing one error
    # for a whole run.
    assert 'kind === "decision"' in js
    assert 'kind === "decision" || kind === "checkpoint"' in js
    assert 'status !== "fail" && status !== "retry"' not in js
    # Which chunks a resume replayed is a line in the log, not a caption under
    # the progress bar for the rest of the run.
    assert "Đang resume" not in js
    assert "Tiếp tục" in js

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    for status in ("start", "ok", "retry", "fail", "note"):
        assert f'.log-line[data-status="{status}"]' in css, status
    assert (".problems" not in css) and (".problem-" not in css)


def test_the_history_panel_scrolls_and_keeps_its_action_in_reach():
    """A rail with many runs must scroll its list rather than hide the newest
    entry behind the panel's edge, and the action that starts a run belongs
    where the history ends: a reader who has just read the history is the one
    deciding to start another. The frame-drawing effect was asked for, so its
    four edges and its label swap are pinned here — along with the focus ring
    the demo switched off, which a keyboard reader still needs."""
    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    aside = html.split("<aside", 1)[1].split("</aside>", 1)[0]
    assert 'id="nav-new"' in aside
    assert '<div class="buttons">' in aside
    # The visible label is drawn from these attributes, so the button carries
    # its own name for anyone who never sees them.
    assert 'aria-label="Dịch tài liệu mới"' in aside
    assert 'data-title="Dịch tài liệu mới"' in aside
    assert "data-text=" in aside
    topbar = html.split("<header", 1)[1].split("</header>", 1)[0]
    assert "nav-new" not in topbar
    assert "topbar-actions" not in html

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    rail = css.split(".rail {", 1)[1].split("}", 1)[0]
    assert "flex-direction: column" in rail and "overflow: hidden" in rail
    jobs = css.split(".jobs {", 1)[1].split("}", 1)[0]
    assert "flex: 1" in jobs and "overflow-y: auto" in jobs
    for rule in (
        ".buttons button:before",
        ".buttons button:after",
        ".buttons button span:before",
        ".buttons button p:before",
        ".buttons button p:after",
        ".buttons button:hover",
        ".buttons button:focus-visible",
    ):
        assert rule in css, rule
    # `:focus { outline: 0 }` is what the demo did; this page keeps its ring.
    assert ".buttons button:focus " not in css.replace(".buttons button:focus-visible ", "")


def test_folding_the_library_slides():
    """Two things about the library's own column.

    The column was a plain snap before: `data-rail` changed the grid's first
    track and the column was suddenly 64px wide with nothing in between. Both
    tracks are lengths, so the track itself is worth interpolating — the column
    travels.
    """
    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    layout = css.split(".layout {", 1)[1].split("}", 1)[0]
    assert "transition: grid-template-columns" in layout
    assert "grid-template-columns: var(--rail) minmax(0, 1fr)" in layout
    # The folded shape is its own rule, so the two ends of the transition are
    # both plain lengths and the sheet still decides what each end is.
    assert '.layout[data-rail="collapsed"] { grid-template-columns: var(--rail-folded)' in css
    for token in ("--rail: 240px", "--rail-folded: 64px"):
        assert token in css, token


def test_folding_the_library_is_a_cross_fade_and_not_a_swap():
    """The fold used to swap one shape for another inside a single frame:
    the words, the filters and the heading were `display: none`d and the letter
    tiles appeared already in place, so a column that took 220ms to travel
    changed everything it held in an instant. Both shapes are now drawn at once
    and only the opacity between them changes, and what cannot fade — where a
    word sits, how wide an entry is, the height its last line needs — travels
    instead of being dropped."""
    # The sheet ships with CRLF endings; a rule spanning two lines is matched
    # with them normalised, so the check is about the rule and not the platform.
    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8").replace("\r\n", "\n")

    def block(selector):
        start = css.index(selector)
        return css[start : css.index("}", start)]

    # The tile is the row's own, drawn in both shapes and invisible in the open
    # one — which is what makes folding a cross-fade rather than an appearance.
    tile = block(".job-row::before {")
    assert "content: attr(data-mark)" in tile
    assert "position: absolute" in tile and "opacity: 0" in tile
    assert "transition: opacity" in tile
    assert (
        '.layout[data-rail="collapsed"] .job-row::before,\n'
        '.layout[data-rail="collapsed"] .job-row::after { opacity: 1; }' in css
    )

    # Every word is faded out of the folded column, and none of them is dropped.
    folded = block('.layout[data-rail="collapsed"] .job-row-name,')
    assert "opacity: 0" in folded and "position: absolute" in folded
    assert "display: none" not in folded
    assert '.layout[data-rail="collapsed"] .rail-name { flex-basis: 0; opacity: 0; }' in css
    # A name folded by width carries the mark beside it to the middle as it
    # goes; one that is dropped takes the mark there in the frame it goes.
    assert "flex: 0 1 100%" in block(".rail-name {")

    # The heading and the filters are carried away by a ceiling that travels, so
    # the rows under them are not left jumping up by their height.
    head = block(".rail-head {")
    assert "max-height: 32px" in head and "overflow: hidden" in head
    assert "max-height 220ms" in head and "margin-bottom 220ms" in head
    assert "max-height: 64px" in block(".rail .chips {")

    # What the fold can travel, it travels: the row's own floor, its inset and
    # the state's edge, and the width of the entry itself.
    row = block(".job-row {")
    assert "min-height: 52px" in row and "overflow: hidden" in row
    assert "min-height 220ms" in row and "padding 220ms" in row
    item = block(".job-item {")
    assert "width: 100%" in item and "width 220ms" in item
    # Centred rather than stretched, so an entry shrinks around its own middle
    # instead of being moved to the middle of the column in one frame.
    assert "align-items: center" in block(".jobs {")
    assert '.layout[data-rail="collapsed"] .job-row {\n  min-height: 36px;' in css
    assert '.layout[data-rail="collapsed"] .job-item { width: 36px; }' in css

    # `display: none` survives in the folded column for one thing only: the
    # trash button, which is not drawn until the pointer is on the row anyway.
    collapsed = css[
        css.index('.layout[data-rail="collapsed"] .rail {') : css.index(".rail-scrim {")
    ]
    assert collapsed.count("display: none") == 1
    assert ".job-delete { display: none; }" in collapsed


def test_the_report_tab_ends_the_run_in_numbers():
    """The run's own tally, in one place: which chunks failed, what the pipeline
    spent on each, and how many had to be translated again. Derived from the
    events the page already polls, so a live run is reported from the same
    figures a finished one is. Milliseconds are the pipeline's unit: printing
    6000 beside an 's' would claim a step took an hour and a half."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    for element_id in ("tab-report", "tab-panel-report", "report"):
        assert f'id="{element_id}"' in html, element_id
    assert 'aria-controls="tab-panel-report"' in html
    # Empty panels are not drawn: the report clears itself when no run is open.
    assert 'id="tab-panel-report" class="tab-panel" role="tabpanel"' in html

    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    for name in ("renderReport", "reportFor", "reportFacts", "reportTable", "formatDuration"):
        assert f"function {name}" in js, name
    assert "chunkTotals" in js
    # The three questions the report answers, each from the events themselves.
    assert '"retranslate"' in js
    assert "duration_ms" in js
    assert "Chunk lỗi" in js and "Chunk chậm nhất" in js and "Chunk phải dịch lại" in js
    # A run still going says its figures are a snapshot, not a verdict.
    assert "số liệu tạm tính" in js

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    for rule in (".report-facts", ".report-table", ".fact-label"):
        assert rule in css, rule


def test_a_running_run_reports_the_work_it_is_on():
    """The line under the bar says what the run is doing now, whatever brought
    it here: a resumed run is a run, and holding that line on the resume itself
    left a reader watching one frozen sentence while chunks were translated."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "function progressWord" in js
    assert "Đang chạy: " in js
    # The resume wording is reached only before this attempt's first step.
    assert "info.requested && info.resumed === 0" in js


def test_a_resume_that_replayed_nothing_does_not_claim_a_mismatch():
    """`checkpoint_usable` only says checkpointing is switched on, so it cannot
    prove that a model or a file changed. A previous attempt stopped before its
    first chunk finished reads exactly the same, and the page used to accuse a
    mismatch every time — the reader had touched nothing. The page must say
    only what is certain: no checkpoint was reused."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "function resumeNotice" in js
    assert "không khớp" not in js, "the page must not assert a cause it never checked"
    assert "Không dùng lại được checkpoint nào" in js
    # Both places that report a resume that replayed nothing share one wording,
    # so the progress line and the notice can never drift apart.
    assert js.count("resumeNotice(") >= 3  # the definition plus its callers


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
        "renderLog",
        "renderBanner",
        "startRun",
        "stopRun",
        "resumeRun",
        "uploadFile",
        "openJob",
        "resumeInfo",  # the web UI reads the pipeline's own early `resume` record
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


def test_the_rail_is_a_few_phases_and_ticks_them_as_the_run_moves_on():
    """The rail was one row per pipeline stage — thirteen of them, most of which
    run once per chunk inside the translation loop, so the rail reported the same
    stage hundreds of times and its markers flickered while the translation was
    still going. It is now a handful of phases, with the per-chunk stages behind
    one row: "dịch & kiểm tra" stands for translate, critic, verify, refine,
    retranslate, collect and save_trace together. A phase is ticked once the
    whole input has gone through it — the run has moved on to a later phase —
    and the explanation that used to sit above the rail as a key is gone with
    it."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    # The rail is one list, named by its own label; nothing explains its marks
    # above it any more.
    assert '<ol id="steps" class="steps" aria-label="Các bước của pipeline"></ol>' in html
    assert "steps-head" not in html and "steps-key" not in html
    assert "số = thứ tự" not in html
    assert "↻" not in html

    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    for name in ("STEP_GROUPS", "STEP_PHASE", "phaseStates", "renderSteps", "STEP_STATE_WORDS"):
        assert name in js, name
    # Six phases, each one named, and the translation loop is one of them: the
    # stages that run per chunk are inside it, not beside it.
    for label in (
        '"đọc & chia tài liệu"',
        '"lập kế hoạch"',
        '"dịch & kiểm tra"',
        '"nhất quán"',
        '"ghép kết quả"',
        '"lưu kinh nghiệm"',
    ):
        assert label in js, label
    block = js[js.index("const STEP_GROUPS") : js.index("];", js.index("const STEP_GROUPS"))]
    # Six rows and no more: the count is the whole point of the change.
    assert len(re.findall(r"^  \[", block, re.MULTILINE)) == 6
    grouped = block[block.index('"dịch & kiểm tra"') : block.index('"nhất quán"')]
    for stage in (
        "translate",
        "critic",
        "verify",
        "refine",
        "retranslate",
        "collect",
        "save_trace",
    ):
        assert f'"{stage}"' in grouped, stage
        # Each stage has exactly one home, and the per-chunk ones are inside the
        # translation loop rather than in a row of their own beside it.
        assert block.count(f'"{stage}"') == 1, stage
    # The phases that are not the loop are single stages, in the pipeline's own
    # order: the loop is the only row that swallows several.
    for phase, stage in (("nhất quán", "consistency"), ("ghép kết quả", "assemble_output")):
        assert block.index(f'"{stage}"') > block.index(f'"{phase}"')
    # A stage the map has not been told about still gets a row of its own.
    assert "STEP_PHASE.get(stage) || stage" in js
    # The rail is told how the run ended, because that is what a tick means:
    # a later phase on the rail, or the run finishing, and nothing else.
    assert "renderSteps(state.events, run.status)" in js
    assert 'const finished = status === "done"' in js
    assert "if (finished || !newest)" in js
    assert 'entry.last === "fail"' in js
    # The per-event mapping and the `recovered` marker it fed are gone rather
    # than merely unused, and so is the strongest-state ladder before them.
    assert "stepStates" not in js
    assert "stepStateFor" not in js
    assert "STEP_RANK" not in js
    assert "STEPS_NOT_RAILED" not in js
    assert '.step[data-state="recovered"]' not in js
    # The label carries the pipeline's own stage names and what the phase is
    # doing, so the short row is still an exact one.
    assert "item.title" in js
    assert "phase.stages.join" in js
    assert "lần lỗi trong lượt này" in js

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    assert '.step[data-state="recovered"]' not in css
    assert '.step[data-state="ok"]::before' in css
    assert ".steps-head" not in css and ".steps-key" not in css
    # The contract at the top of the sheet names every value the script writes,
    # and the rail's own line names exactly the three that are left: a phase on
    # the rail has been reached, so there is no `wait` to write.
    contract = css.split("State contract", 1)[1].split("*/", 1)[0]
    step_lines = [
        " ".join(line.split()) for line in contract.splitlines() if ".step[data-state=" in line
    ]
    assert step_lines == ["* .step[data-state=running|ok|fail]"]


def test_the_run_settings_moved_from_the_title_to_the_report():
    """What a run was asked to do — the language pair, the chunk
    count, when it started — sat in a row of chips under the title, competing
    with the badge and the controls for the same corner and saying nothing about
    the translation being read. Those are facts about the translation, so they
    are listed with the rest of the run's figures in the report tab."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    # The heading keeps the document's name and the badge: its chip row is gone.
    assert "show(meta, false)" in js
    assert 'element("span", "tag", tag)' not in js
    # And the three facts are in the report, ahead of the verdict that ends it.
    for label in ('["Cặp ngôn ngữ"', '["Số chunk"', '["Bắt đầu"'):
        assert label in js, label
    # Listed before the verdict, so the report still ends on how the run went.
    facts = js[js.index("function reportFacts(report)") : js.index("function reportTable")]
    assert facts.index('["Cặp ngôn ngữ"') < facts.index('"Kết thúc"')


def test_the_running_clock_measures_the_attempt_and_not_the_whole_run():
    """A continued run must not report the time it spent failed as running time.

    A run keeps the day it was created, and one that sat failed for an hour
    before the reader pressed "Tiếp tục" showed an hour of running time while
    the attempt actually being watched was a minute old — a healthy run read
    as one that had been stuck for an hour. The clock under the progress bar
    starts again with every attempt, which is what the newest `run_start`
    record says.
    """
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "function attemptStart(events, run)" in js
    # The attempt is the newest run_start in the log.
    assert 'event.type === "run_start"' in js
    # The clock reads the attempt, not the run's creation.
    assert "formatClock(until - began)" in js
    assert "formatClock(until - run.started)" not in js


def test_the_resume_notice_is_raised_once_and_not_on_every_poll():
    """The notice about a continued run is a fact about the start of the attempt.

    The resume record stays in the log for the whole run, so re-stamping the
    banner on every poll made it blink in and out for as long as the run went
    on — a warning about the attempt's first minute chasing the reader through
    the hour that followed. It is raised once, for the record it belongs to.
    """
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    # The notice is tied to the resume record it describes, by its time.
    assert "ts: Number(event.ts) || 0," in js
    assert "resume.ts !== state.resumeSeen" in js
    assert "state.resumeSeen = resume.ts;" in js


def test_the_view_switch_has_a_row_of_its_own_and_the_badge_its_own_space():
    """Two controls that collided with what they sat beside: the switch sat flush
    above the right-hand column, right-aligned, with its own words landing on
    that column's title, and the badge sat 8px from the button that changes the
    state it names."""
    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    assert 'class="panes-note-label"' in html
    assert 'aria-labelledby="view-label"' in html

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    note = css.split(".panes-note {", 1)[1].split("}", 1)[0]
    assert "justify-content: space-between" in note
    assert "margin-bottom" in note
    side = css.split(".job-head-side {", 1)[1].split("}", 1)[0]
    assert "gap: var(--sp-4)" in side


def test_a_chunk_the_run_walked_past_is_not_held_up_as_broken():
    """The pipeline does not stop for one bad chunk: a step that times out leaves
    that chunk behind and translates the next ones. Holding the abandoned chunk
    red until the end of the run showed early chunks broken beside later chunks
    that had finished, with the progress bar still climbing. Once the run has
    overtaken a chunk, the failure is history — but it stays reachable: in the
    tooltip, in the message a row opens, and in the filter that answers "which
    chunks went wrong?"."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    for element_id in ("chunk-filter-all", "chunk-filter-fail"):
        assert f'id="{element_id}"' in html, element_id
    # The filter lists chunks that hit a problem, including one that moved on.
    assert "Chunk từng lỗi" in html
    assert "Chỉ chunk lỗi" not in html

    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    # The furthest chunk the run has touched, and a chunk below it that failed.
    assert "entry.index < highest" in js
    assert "entry.recovered = true" in js
    assert "CHUNK_LEFT_WORD" in js
    assert 'row.dataset.recovered = "true"' in js
    # A chunk that failed and was left behind is answered by its message, so the
    # filter cannot be a state test alone.
    assert 'cell.state !== "fail" && !cell.error' in js
    # The state's own word and the tooltip both stop naming the step it died on.
    assert "lỗi trước đó" in js
    assert "chunk từng lỗi" in js

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    assert '.chunk[data-recovered="true"]' in css
    marker = css.split('.chunk[data-recovered="true"] .chunk-id::after', 1)[1].split("}", 1)[0]
    assert "↻" in marker


def test_a_run_waiting_out_a_provider_says_so_under_the_bar():
    """A chunk that failed and is waiting to be run again is still working.

    The pipeline now runs such a chunk again until it is translated, so a wait
    of minutes is expected — and a caption that goes on saying "đang chạy"
    with no word about the wait reads as a hang. The last event a waiting run
    writes is the wait itself, which is what the caption is drawn from.
    """
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert '"Đang chờ thử lại"' in js
    # A provider slot collision — the wait that can last an hour — names what
    # it is actually waiting for, not just that a retry is pending.
    assert '"Đang chờ provider nhả slot"' in js
    # Keyed on the status the pipeline writes for a wait, so the caption cannot
    # be reached by any other event.
    assert 'last.status || "") === "retry"' in js
    # And it carries the wait's own words: what failed, and how long.
    assert "last.error || last.message" in js
    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    assert '.log-line[data-status="retry"] .log-word' in css


def test_the_run_controls_sit_beside_the_state_they_change():
    """Stopping a run is not downloading its output. "Dừng" and "Tiếp tục" belong
    at the top-right of the panel, on the line of the badge that names the run's
    state, not two rows lower in with the download links."""
    from pathlib import Path

    html = call(repo_api(), "GET", "/").body.decode("utf-8")
    assert 'id="job-control"' in html
    # It is in the header, above the row it used to share with the downloads.
    assert html.index('id="job-control"') < html.index('id="job-actions"')
    assert 'class="job-head-side"' in html

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert 'clear(byId("job-control"))' in js
    assert "control.append(stop)" in js
    assert "control.append(resume)" in js
    assert "box.append(stop)" not in js
    assert "box.append(resume)" not in js

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    for rule in (".job-head-side {", ".job-control {", ".job-control:empty {"):
        assert rule in css, rule


def test_the_run_controls_swap_their_word_for_a_mark():
    """A run's own two controls are the only buttons whose label steps aside: on
    hover the word slides out to the right and fades, and the mark — a pause for
    "Dừng", a play for "Tiếp tục" — slides in from the left and keeps breathing.
    The word stays the button's name, and the mark is decoration: a screen reader
    must not be told the same thing twice, and a reader who never hovers must
    still see the sentence. "Dừng" keeps the plain control's surface; "Tiếp tục"
    takes the warn tone the stopped run's own title wears."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    for name in ("swapButton", "markIcon", "CONTROL_MARKS"):
        assert name in js, name
    # Each control names its own mark, so a label and a glyph cannot drift apart.
    assert 'swapButton("Dừng", CONTROL_MARKS.stop)' in js
    assert 'swapButton("Tiếp tục", CONTROL_MARKS.resume, "btn--resume")' in js
    assert 'CONTROL_MARKS = { stop: "pause", resume: "play" }' in js
    # The word is the accessible name; the mark is built node by node and hidden.
    assert 'element("span", "btn-label", label)' in js
    assert 'element("span", "btn-mark")' in js
    assert 'svg.setAttribute("aria-hidden", "true")' in js
    assert "createElementNS" in js
    # Both glyphs exist, and a mark the pipeline does not know is not invented.
    assert '"pause"' in js and '"play"' in js

    css = call(repo_api(), "GET", "/static/app.css").body.decode("utf-8")
    for rule in (".btn--swap {", ".btn-label {", ".btn-mark {", ".btn--resume {"):
        assert rule in css, rule
    # Reacting to the pointer is for buttons that can be pressed at all.
    assert ".btn--swap:hover:not(:disabled) .btn-label" in css
    assert ".btn--swap:hover:not(:disabled) .btn-mark" in css
    assert "@keyframes mark-breathe" in css
    # The label leaves the button rather than lingering in its padding.
    assert "translateX(calc(100% + 12px))" in css
    resume = css.split(".btn--resume {", 1)[1].split("}", 1)[0]
    assert "var(--warn)" in resume and "var(--warn-weak)" in resume
    # "Dừng" is left as the plain control it already was: no stop-specific tone.
    assert ".btn--stop" not in css


def test_a_fresh_run_does_not_claim_it_resumed_anything():
    """The pipeline writes its `resume` record on every attempt, a fresh run
    included: the record describes the checkpoint store, and the store is built
    before the run knows whether it was asked to resume. The log drew that record
    as a line, so a brand-new run opened with "Tiếp tục · Không dùng lại được
    checkpoint nào cho lượt này — các chunk sẽ dịch từ đầu" — which reads as a
    resume that discarded everything, on a run that had nothing to discard. Only
    a resume that was actually asked for has anything to say."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    # The guard is on the record's own word for it, not on the chunk count: a
    # fresh run and a resume that matched nothing both replay zero chunks.
    assert "if (!event.resume_requested)" in js
    resume_branch = js.split('} else if (kind === "resume") {', 1)[1].split("} else if", 1)[0]
    assert "return null;" in resume_branch
    # Nothing still claims the record is absent from a fresh run: it is there,
    # and that is exactly why the reader must not be shown it.
    assert "Absent on a fresh run" not in js
    assert "Every attempt writes one, a fresh run included" in js

    # Why the guard has to live in the reader: the pipeline writes the record for
    # both cases, and only says which one it was in `resume_requested`.
    coordinator = (root / "src" / "polyglot_lc" / "pipeline" / "coordinator.py").read_text(
        encoding="utf-8"
    )
    record = coordinator.split('type="resume"', 1)[1].split("checkpoint_usable", 1)[0]
    assert "resume_requested=bool(resume)" in record, record
    assert "resumed_chunks=result.resumed_chunks" in record, record
