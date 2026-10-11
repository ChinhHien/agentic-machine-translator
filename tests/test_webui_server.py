"""Tests for webui.server: routing, static serving, and the API surface."""

from __future__ import annotations

import json

import pytest

from webui.runs import RunManager
from webui.server import Api
from webui.store import (
    EDITABLE_KEYS,
    READONLY_KEYS,
    ConfigStore,
    GlossaryStore,
    RunsStore,
    SkillsStore,
)


@pytest.fixture()
def api(tmp_path):
    """An Api over a repo-shaped temp dir, with the real static layout."""
    (tmp_path / "webui" / "static").mkdir(parents=True)
    (tmp_path / "webui" / "static" / "index.html").write_text(
        "<!doctype html><title>x</title>", encoding="utf-8")
    (tmp_path / "webui" / "static" / "app.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "webui" / "static" / "secret.py").write_text("nope", encoding="utf-8")
    (tmp_path / ".env").write_text("CHUNK_MAX_TOKENS=700\n", encoding="utf-8")
    return Api(
        root=tmp_path,
        manager=RunManager(tmp_path, python="PY"),
        runs=RunsStore(tmp_path / "logs" / "webui"),
        skills=SkillsStore(tmp_path / "skills"),
        glossaries=GlossaryStore(tmp_path / "glossaries"),
        config=ConfigStore(tmp_path, tmp_path / ".env"),
        port=8787,
    )


def call(api, method, path, **kw):
    from urllib.parse import parse_qs, urlsplit

    parts = urlsplit(path)
    return api.handle(method, parts.path, parse_qs(parts.query), kw.get("body", b""),
                      {"Origin": "http://127.0.0.1:8787", **kw.get("headers", {})})


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


def test_health(api):
    r = call(api, "GET", "/api/health")
    assert r.status == 200 and r.body == b'{"ok": true}'


# --- Task 8: origin, traversal, secrets, uploads ---------------------------


def test_mutations_require_a_loopback_origin(api):
    for headers in ({"Origin": "http://evil.example"}, {}):
        r = api.handle("PUT", "/api/inputs/a.md", {}, b"x", headers)
        assert r.status == 403, headers


def test_origin_may_be_either_loopback_name(api):
    """Both spellings of loopback are the same page for our purposes.

    Each upload uses its own file name on purpose: re-uploading the same name
    is refused by the overwrite rule, which `test_upload_does_not_overwrite…`
    covers. Asserting 200 twice for one name would contradict that rule.
    """
    pairs = (("http://127.0.0.1:8787", "a.md"), ("http://localhost:8787", "b.md"))
    for origin, name in pairs:
        r = call(api, "PUT", f"/api/inputs/{name}", body=b"x", headers={"Origin": origin})
        assert r.status == 200, origin


def test_reads_do_not_need_an_origin(api):
    """A plain GET of the page must work; only mutations are guarded."""
    assert call(api, "GET", "/api/inputs", headers={"Origin": "http://evil.example"}).status == 200


def test_path_traversal_is_rejected(api):
    for bad in ("../.env", "a/b", "..%2F.env"):
        assert call(api, "PUT", f"/api/inputs/{bad}", body=b"x").status == 403, bad


def test_traversal_variants_are_rejected(api, tmp_path):
    """Windows separators and encoded ones are refused just as firmly."""
    for bad in ("..\\..\\.env", "a%5Cb", "%2E%2E%2Fx", ""):
        r = call(api, "PUT", f"/api/inputs/{bad}", body=b"x")
        assert r.status == 403, bad
    assert not (tmp_path / ".env").read_bytes().endswith(b"x")  # .env untouched


def test_config_response_never_contains_a_secret_value(api, tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSLATOR_API_KEY", "sk-CANARY-LEAK-CHECK")
    (tmp_path / ".env").write_text(
        "CHUNK_MAX_TOKENS=700\nTRANSLATOR_API_KEY=sk-CANARY-LEAK-CHECK\n", encoding="utf-8")
    r = call(api, "GET", "/api/config")
    assert r.status == 200
    assert b"sk-CANARY-LEAK-CHECK" not in r.body
    assert b'"TRANSLATOR_API_KEY": "set"' in r.body


def test_config_ignores_secret_shaped_keys_the_probe_returns(api, monkeypatch):
    """The whitelist decides what the UI may see — not the probe's goodwill."""
    monkeypatch.setattr(
        type(api.manager),
        "probe",
        lambda self, args: {
            "values": {"CHUNK_MAX_TOKENS": 700, "TRANSLATOR_API_KEY": "sk-CANARY-LEAK-CHECK"},
            "sources": {"CHUNK_MAX_TOKENS": "dotenv", "TRANSLATOR_API_KEY": "env"},
        },
    )
    body = call(api, "GET", "/api/config").body
    data = json.loads(body)
    assert b"sk-CANARY-LEAK-CHECK" not in body
    assert "TRANSLATOR_API_KEY" not in data["values"]
    assert "TRANSLATOR_API_KEY" not in data["sources"]
    assert data["values"]["CHUNK_MAX_TOKENS"] == 700
    assert data["sources"]["CHUNK_MAX_TOKENS"] == "dotenv"
    assert set(data["editable"]) == set(EDITABLE_KEYS)
    assert set(data["readonly"]) == set(READONLY_KEYS)


def test_config_reports_unset_for_credentials_that_are_absent(api, monkeypatch):
    from webui.store import SECRET_KEYS

    for name in SECRET_KEYS:
        monkeypatch.delenv(name, raising=False)
    secrets = json.loads(call(api, "GET", "/api/config").body)["secrets"]
    assert set(SECRET_KEYS) <= set(secrets)
    assert set(secrets.values()) <= {"set", "unset"}
    assert secrets["TRANSLATOR_API_KEY"] == "unset"


def test_config_says_so_when_the_probe_cannot_run(api):
    """The fixture's interpreter is a stub, so this is the real degraded path."""
    data = json.loads(call(api, "GET", "/api/config").body)
    assert data["values"] == {} and data["probe_error"]


def test_upload_does_not_overwrite_without_consent(api, tmp_path):
    assert call(api, "PUT", "/api/inputs/a.md", body=b"first").status == 200
    assert call(api, "PUT", "/api/inputs/a.md", body=b"second").status == 409
    assert (tmp_path / "input" / "a.md").read_bytes() == b"first"
    assert call(api, "PUT", "/api/inputs/a.md?overwrite=1", body=b"second").status == 200
    assert (tmp_path / "input" / "a.md").read_bytes() == b"second"


def test_input_names_with_spaces_and_unicode_round_trip(api, tmp_path):
    """Review Focus 5: tên có dấu cách / ngoài ASCII vẫn tải lên và đọc lại được."""
    from urllib.parse import quote

    name = "Think Data dữ liệu.md"
    payload = "nội dung".encode()  # bytes literal phải là ASCII, nội dung thì không
    r = call(api, "PUT", f"/api/inputs/{quote(name)}", body=payload)
    assert r.status == 200
    assert (tmp_path / "input" / name).read_bytes() == payload

    names = [f["name"] for f in json.loads(call(api, "GET", "/api/inputs").body)["files"]]
    assert name in names


def test_inputs_listing_reports_size_and_tolerates_a_missing_dir(api, tmp_path):
    assert json.loads(call(api, "GET", "/api/inputs").body)["files"] == []

    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.md").write_bytes(b"12345")
    (tmp_path / "input" / "subdir.md").mkdir()
    files = json.loads(call(api, "GET", "/api/inputs").body)["files"]
    assert [(f["name"], f["size"]) for f in files] == [("a.md", 5)]  # thư mục bị bỏ qua
    assert files[0]["mtime"] > 0
