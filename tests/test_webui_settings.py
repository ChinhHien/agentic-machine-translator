"""Tests for webui.settings: the dialog's store, endpoint probing, env building,
and the routes plus launch injection that carry them."""

from __future__ import annotations

import json

import pytest

from webui.settings import (
    AGENT_ENV_PREFIXES,
    LAUNCH_TEST_TIMEOUT,
    Settings,
    SettingsError,
    SettingsStore,
    child_env,
)
from webui.settings import test_endpoint as _probe
from webui.store import RunsStore

# --------------------------------------------------------------------- store


def test_missing_file_loads_defaults(tmp_path):
    store = SettingsStore(tmp_path)
    settings = store.load()
    assert settings.use_source_group is False
    assert settings.max_concurrency == 1
    assert settings.provider_concurrency == 1
    assert settings.agents == {}


def test_corrupt_file_loads_defaults(tmp_path):
    path = tmp_path / "data" / "webui-settings.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert SettingsStore(tmp_path).load() == Settings()


def test_save_and_load_roundtrip(tmp_path):
    store = SettingsStore(tmp_path)
    store.apply_update(
        {
            "use_source_group": True,
            "max_concurrency": 4,
            "provider_concurrency": 6,
            "agents": {
                "PLANNER": {"base_url": "https://a.example/v1/", "model": "m", "api_key": "k1"},
                "SOURCE_GROUP": {"base_url": "https://b.example/v1", "model": "m2", "api_key": ""},
            },
        }
    )
    settings = store.load()
    assert settings.use_source_group is True
    assert settings.max_concurrency == 4
    assert settings.provider_concurrency == 6
    # The trailing slash of a base URL is normalised away.
    assert settings.agents["PLANNER"].base_url == "https://a.example/v1"
    assert settings.agents["SOURCE_GROUP"].api_key == ""


def test_empty_key_keeps_the_stored_key(tmp_path):
    store = SettingsStore(tmp_path)
    store.apply_update(
        {
            "agents": {
                "PLANNER": {"base_url": "https://a.example/v1", "model": "m", "api_key": "sk-real"}
            }
        }
    )
    # The dialog never echoes a key back, so a re-submission with an empty key
    # means "unchanged", not "removed".
    store.apply_update(
        {"agents": {"PLANNER": {"base_url": "https://a.example/v1", "model": "m2", "api_key": ""}}}
    )
    assert store.load().agents["PLANNER"].api_key == "sk-real"


def test_agents_left_out_of_an_update_are_dropped(tmp_path):
    store = SettingsStore(tmp_path)
    store.apply_update(
        {
            "agents": {
                "PLANNER": {"base_url": "https://a.example/v1", "model": "m", "api_key": ""},
                "VERIFIER": {"base_url": "https://b.example/v1", "model": "m", "api_key": ""},
            }
        }
    )
    # The dialog sends every card it shows; one it leaves out was emptied by
    # the reader, so the stored override goes with it.
    store.apply_update(
        {"agents": {"VERIFIER": {"base_url": "https://b.example/v1", "model": "m", "api_key": ""}}}
    )
    assert "PLANNER" not in store.load().agents
    assert "VERIFIER" in store.load().agents


def test_unknown_agent_and_bad_url_are_refused(tmp_path):
    store = SettingsStore(tmp_path)
    with pytest.raises(SettingsError):
        store.apply_update(
            {"agents": {"NOBODY": {"base_url": "https://a.example/v1", "model": "m"}}}
        )
    with pytest.raises(SettingsError):
        store.apply_update({"agents": {"PLANNER": {"base_url": "ftp://a.example", "model": "m"}}})
    with pytest.raises(SettingsError):
        store.apply_update({"max_concurrency": 99})


def test_delete_agent_removes_entry_and_test(tmp_path):
    store = SettingsStore(tmp_path)
    store.apply_update({"agents": {"PLANNER": {"base_url": "https://a.example/v1", "model": "m"}}})
    store.record_test("PLANNER", {"ok": True, "latency_ms": 5, "method": "models", "error": ""})
    assert store.delete_agent("PLANNER") is True
    assert store.load().agents == {}
    assert store.public()["tests"] == {}
    assert store.delete_agent("PLANNER") is False
    with pytest.raises(SettingsError):
        store.delete_agent("NOBODY")


def test_public_never_echoes_a_key(tmp_path):
    store = SettingsStore(tmp_path)
    store.apply_update(
        {
            "agents": {
                "PLANNER": {"base_url": "https://a.example/v1", "model": "m", "api_key": "sk-real"}
            }
        }
    )
    public = store.public()
    assert public["agents"]["PLANNER"] == {
        "base_url": "https://a.example/v1",
        "model": "m",
        "has_key": True,
    }
    assert "sk-real" not in json.dumps(public)


def test_record_test_survives_the_next_save(tmp_path):
    store = SettingsStore(tmp_path)
    store.apply_update({"agents": {"PLANNER": {"base_url": "https://a.example/v1", "model": "m"}}})
    store.record_test("PLANNER", {"ok": True, "latency_ms": 12, "method": "models", "error": ""})
    store.apply_update({"agents": {"PLANNER": {"base_url": "https://a.example/v1", "model": "m2"}}})
    tests = store.public()["tests"]
    assert tests["PLANNER"]["ok"] is True
    assert "checked_at" in tests["PLANNER"]


# --------------------------------------------------------------- test_endpoint


class _FakeResponse:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return b"{}"


class _FakeHTTPError(Exception):
    code = 404

    def read(self):
        return b"no such route"


def _urlopen(plan):
    """A urlopen stand-in driven by (method, path): "ok" or an exception."""

    def urlopen(request, timeout):
        url = request.get_full_url()
        outcome = None
        for pattern, value in plan:
            if pattern in url:
                outcome = value
                break
        if outcome is None:
            raise AssertionError(f"unexpected URL: {url}")
        if outcome == "ok":
            return _FakeResponse()
        if isinstance(outcome, Exception):
            raise outcome
        raise AssertionError(f"unknown outcome: {outcome}")

    return urlopen


def test_endpoint_ok_via_models(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "webui.settings.urllib.request.urlopen",
        _urlopen([("/models", "ok")]),
    )
    result = _probe("https://a.example/v1", "m", "sk")
    assert result["ok"] is True
    assert result["method"] == "models"


def test_endpoint_falls_back_to_chat(tmp_path, monkeypatch):
    # /models refuses; the one-token completion answers.
    monkeypatch.setattr(
        "webui.settings.urllib.request.urlopen",
        _urlopen([("/models", _FakeHTTPError()), ("/chat/completions", "ok")]),
    )
    result = _probe("https://a.example/v1", "m", "sk")
    assert result["ok"] is True
    assert result["method"] == "chat"


def test_endpoint_reports_the_last_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "webui.settings.urllib.request.urlopen",
        _urlopen(
            [
                ("/models", _FakeHTTPError()),
                ("/chat/completions", ConnectionError("refused")),
            ]
        ),
    )
    result = _probe("https://a.example/v1", "m", "sk")
    assert result["ok"] is False
    assert "refused" in result["error"]


def test_endpoint_needs_base_url_and_model(tmp_path):
    result = _probe("", "m", "")
    assert result["ok"] is False and "base_url" in result["error"]
    result = _probe("https://a.example/v1", " ", "")
    assert result["ok"] is False and "model" in result["error"]


# ------------------------------------------------------------------ child_env


def test_child_env_builds_prefixes_and_skips_failures():
    settings = Settings.model_validate(
        {
            "max_concurrency": 3,
            "provider_concurrency": 5,
            "agents": {
                "PLANNER": {"base_url": "https://a/v1", "model": "m1", "api_key": "k"},
                "VERIFIER": {"base_url": "https://b/v1", "model": "m2", "api_key": ""},
            },
        }
    )
    additions, removals = child_env(settings, {"PLANNER": True, "VERIFIER": False})
    assert additions["PLANNER_BASE_URL"] == "https://a/v1"
    assert additions["PLANNER_MODEL"] == "m1"
    assert additions["PLANNER_API_KEY"] == "k"
    # The failed endpoint is skipped entirely — .env keeps its place.
    assert not any(key.startswith("VERIFIER_") for key in additions)
    assert additions["MAX_CONCURRENCY"] == "3"
    # The worker pool is only useful when the provider queue lets calls through.
    assert additions["LLM_PROVIDER_CONCURRENCY"] == "5"
    assert removals == []


def test_child_env_group_pins_the_three_agents():
    settings = Settings.model_validate(
        {
            "use_source_group": True,
            "agents": {
                "SOURCE_GROUP": {"base_url": "https://g/v1", "model": "gm", "api_key": "gk"}
            },
        }
    )
    additions, removals = child_env(settings, {"SOURCE_GROUP": True})
    assert additions["SOURCE_GROUP_BASE_URL"] == "https://g/v1"
    for agent in ("TRANSLATOR", "CRITIC", "REFINER"):
        prefix = AGENT_ENV_PREFIXES[agent]
        assert additions[f"{prefix}BASE_URL"] == "https://g/v1"
        assert additions[f"{prefix}MODEL"] == "gm"
        assert additions[f"{prefix}API_KEY"] == "gk"
        assert f"{prefix}BASE_URL" in removals


def test_child_env_group_failing_leaves_env_alone():
    settings = Settings.model_validate(
        {
            "use_source_group": True,
            "agents": {"SOURCE_GROUP": {"base_url": "https://g/v1", "model": "gm"}},
        }
    )
    additions, removals = child_env(settings, {"SOURCE_GROUP": False})
    assert additions == {"MAX_CONCURRENCY": "1", "LLM_PROVIDER_CONCURRENCY": "1"}
    assert removals == []


# --------------------------------------------------------------------- routes


@pytest.fixture()
def api(tmp_path):
    from webui.server import Api

    (tmp_path / "webui" / "static").mkdir(parents=True)
    (tmp_path / "webui" / "static" / "index.html").write_text("<!doctype html>", encoding="utf-8")
    from webui.runs import RunManager

    return Api(
        root=tmp_path,
        manager=RunManager(tmp_path, python="PY"),
        runs=RunsStore(tmp_path / "logs" / "webui"),
        glossaries=None,
        port=8787,
    )


def _call(api, method, path, body=None):
    from urllib.parse import parse_qs, urlsplit

    parts = urlsplit(path)
    return api.handle(
        method,
        parts.path,
        parse_qs(parts.query),
        json.dumps(body).encode("utf-8") if body is not None else b"",
        {"Origin": "http://127.0.0.1:8787"},
    )


def test_settings_routes_roundtrip(api):
    body = {
        "use_source_group": True,
        "max_concurrency": 2,
        "provider_concurrency": 4,
        "agents": {
            "PLANNER": {"base_url": "https://a.example/v1", "model": "m", "api_key": "sk-1"}
        },
    }
    response = _call(api, "PUT", "/api/settings", body)
    assert response.status == 200
    saved = json.loads(response.body)
    assert saved["use_source_group"] is True
    assert saved["max_concurrency"] == 2
    assert saved["provider_concurrency"] == 4
    assert saved["agents"]["PLANNER"]["has_key"] is True
    assert "sk-1" not in response.body.decode("utf-8")

    fetched = json.loads(_call(api, "GET", "/api/settings").body)
    assert fetched == saved

    deleted = json.loads(_call(api, "DELETE", "/api/settings/PLANNER").body)
    assert deleted["deleted"] == "PLANNER"
    assert "PLANNER" not in deleted["settings"]["agents"]
    assert _call(api, "DELETE", "/api/settings/PLANNER").status == 404
    assert _call(api, "DELETE", "/api/settings/NOBODY").status == 404


def test_settings_rejects_a_malformed_body(api):
    assert (
        _call(api, "PUT", "/api/settings", {"agents": {"PLANNER": {"base_url": "nope"}}}).status
        == 422
    )
    assert _call(api, "PUT", "/api/settings", {"max_concurrency": 0}).status == 422
    assert _call(api, "PUT", "/api/settings", {"provider_concurrency": 0}).status == 422


def test_settings_test_uses_the_stored_key_when_field_is_empty(api, monkeypatch):
    seen = {}

    def fake_test(base_url, model, api_key, timeout=10.0):
        seen.update(base_url=base_url, model=model, api_key=api_key)
        return {"ok": True, "latency_ms": 1, "method": "models", "error": ""}

    monkeypatch.setattr("webui.server.test_endpoint", fake_test)
    _call(
        api,
        "PUT",
        "/api/settings",
        {"agents": {"PLANNER": {"base_url": "https://a/v1", "model": "m", "api_key": "sk-stored"}}},
    )
    result = json.loads(
        _call(
            api,
            "POST",
            "/api/settings/test",
            {"agent": "PLANNER", "base_url": "https://a/v1", "model": "m"},
        ).body
    )
    assert result["ok"] is True
    assert seen["api_key"] == "sk-stored"
    # The verdict is remembered for the next time the dialog opens.
    assert json.loads(_call(api, "GET", "/api/settings").body)["tests"]["PLANNER"]["ok"] is True


# ------------------------------------------------------------- launch inject


class _FakeProc:
    def __init__(self, argv, **kw):
        self.pid = 4242
        self.returncode = None

    def poll(self):
        return self.returncode


def test_launch_injects_passed_overrides_only(tmp_path, monkeypatch):
    """A launch turns the dialog's settings into child env — tested ones only."""
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "doc.pdf").write_bytes(b"%PDF-1.4 stub")

    from webui.runs import RunManager
    from webui.settings import EndpointOverride

    store = SettingsStore(tmp_path)
    store.save(
        Settings(
            max_concurrency=2,
            agents={
                "PLANNER": EndpointOverride(base_url="https://a/v1", model="m1", api_key="k"),
                "VERIFIER": EndpointOverride(base_url="https://b/v1", model="m2"),
            },
        )
    )
    verdicts = {"PLANNER": True, "VERIFIER": False}
    # `load_dotenv()` runs at polyglot_lc.config import time and pushes the
    # repo's real .env values into this process's os.environ, so a run that
    # imports it earlier in the session leaks ambient VERIFIER_* here. The
    # assertion below is about the launch's own env surgery, so start clean.
    for suffix in ("BASE_URL", "MODEL", "API_KEY", "TEMPERATURE"):
        monkeypatch.delenv(f"VERIFIER_{suffix}", raising=False)
    monkeypatch.setattr(
        "webui.runs.test_endpoint",
        lambda base_url, model, key, timeout=LAUNCH_TEST_TIMEOUT: {
            "ok": verdicts[{"https://a/v1": "PLANNER", "https://b/v1": "VERIFIER"}[base_url]]
        },
    )
    monkeypatch.setattr(
        RunManager,
        "plan",
        lambda self, file, max_tokens: {"chunks": 1, "output_path": "/out/x.md", "title": "d"},
    )
    made = []

    def popen(argv, **kw):
        made.append(kw)
        return _FakeProc(argv, **kw)

    manager = RunManager(tmp_path, python="PY", popen=popen)
    manager.start(file="doc.pdf", src="en", tgt="vi", glossary=None, max_tokens=None, mock=False)


    env = made[0]["env"]
    assert env["PLANNER_BASE_URL"] == "https://a/v1"
    assert env["PLANNER_MODEL"] == "m1"
    assert env["MAX_CONCURRENCY"] == "2"
    # The failed endpoint keeps the pipeline on its .env value. Only the three
    # endpoint fields are the override's; a global tuning key such as
    # VERIFIER_MAX_TOKENS is inherited from .env like MAX_CONCURRENCY is, and is
    # not a leak of the endpoint the reader left untested.
    endpoint_fields = {"BASE_URL", "MODEL", "API_KEY"}
    offenders = [
        key
        for key in env
        if key.startswith("VERIFIER_") and key.split("_", 1)[1] in endpoint_fields
    ]
    assert not offenders, f"leaked: {offenders}"
