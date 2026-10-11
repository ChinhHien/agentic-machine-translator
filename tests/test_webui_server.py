"""Tests for webui.server: routing, static serving, and the API surface."""

from __future__ import annotations

import pytest

from webui.runs import RunManager
from webui.server import Api
from webui.store import ConfigStore, GlossaryStore, RunsStore, SkillsStore


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


def test_a_missing_index_is_reported_not_crashed(tmp_path, api):
    (tmp_path / "webui" / "static" / "index.html").unlink()
    r = call(api, "GET", "/")
    assert r.status == 404 and r.content_type.startswith("application/json")
