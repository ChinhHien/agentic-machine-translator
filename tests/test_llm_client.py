"""Tests for the LLM client with a mocked HTTP server (respx)."""

from __future__ import annotations

import httpx
import pytest
import respx

from polyglot_lc.config import AgentConfig, EmbedConfig
from polyglot_lc.llm.client import LLMClient, LLMError, TokenUsage, parse_json_reply

BASE = "http://mockapi.test/v1"


def make_agent_config(**kw) -> AgentConfig:
    return AgentConfig(name="TRANSLATOR", base_url=BASE, model="test-model",
                       api_key="sk-test", temperature=0.1, **kw)


CHAT_URL = BASE + "/chat/completions"
EMBED_URL = BASE + "/embeddings"


def chat_payload(text: str) -> dict:
    return {
        "id": "c1", "object": "chat.completion", "created": 1, "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


@respx.mock
def test_chat_success():
    respx.post(CHAT_URL).respond(json=chat_payload("bonjour"))
    client = LLMClient(make_agent_config())
    resp = client.chat([{"role": "user", "content": "hello"}])
    assert resp.text == "bonjour"
    assert resp.prompt_tokens == 5
    assert resp.completion_tokens == 7


@respx.mock
def test_chat_json_mode_sends_response_format():
    route = respx.post(CHAT_URL).respond(json=chat_payload('{"a": 1}'))
    client = LLMClient(make_agent_config())
    client.chat([{"role": "user", "content": "x"}], json_mode=True)
    body = route.calls.last.request.read()
    assert b"json_object" in body


@pytest.fixture(autouse=True)
def _fast_resilience(monkeypatch):
    """Never sleep during retry tests (the resilience layer owns retries)."""
    monkeypatch.setattr("polyglot_lc.llm.resilience._sleep", lambda s: None)
    yield


@respx.mock
def test_chat_retries_on_500_then_succeeds():
    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(200, json=chat_payload("ok")),
    ]
    client = LLMClient(make_agent_config())
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.text == "ok"
    assert route.call_count == 2


@respx.mock
def test_chat_fails_after_retries():
    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(500, json={"error": "boom"}),
    ]
    client = LLMClient(make_agent_config())
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])


@respx.mock
def test_embed_success():
    respx.post(EMBED_URL).respond(json={
        "object": "list", "model": "embed-1",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
        ],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    })
    client = LLMClient(EmbedConfig(base_url=BASE, model="embed-1", api_key="k"))
    resp = client.embed(["a", "b"])
    assert resp.vectors == [[0.1, 0.2], [0.3, 0.4]]


@respx.mock
def test_embed_error_normalized():
    respx.post(EMBED_URL).respond(status_code=503)
    client = LLMClient(EmbedConfig(base_url=BASE, model="embed-1", api_key="k"))
    with pytest.raises(LLMError):
        client.embed(["a"])


def test_parse_json_reply_variants():
    assert parse_json_reply('{"a": 1}') == {"a": 1}
    assert parse_json_reply('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json_reply('junk before {"a": 3} after') == {"a": 3}
    with pytest.raises(LLMError):
        parse_json_reply("no json here")


def test_mock_mode_short_circuits(mock_env):
    cfg = make_agent_config()
    client = LLMClient(cfg)
    resp = client.chat([{"role": "user", "content": "hello"}])
    assert resp.text  # deterministic mock text, no network


def test_token_usage_totals_and_per_agent():
    TokenUsage.reset()
    TokenUsage.add(100, 50, agent="PLANNER")
    TokenUsage.add(200, 25, agent="TRANSLATOR", cached_tokens=120, model="m",
                   provider="http://p/v1")
    TokenUsage.add(10, 5, agent="PLANNER")
    TokenUsage.add(7, 3)  # unattributed call still hits the totals
    snap = TokenUsage.snapshot()
    assert snap["sent"] == 317
    assert snap["received"] == 83
    assert snap["calls"] == 4
    assert snap["cached"] == 120
    by_agent = TokenUsage.snapshot_by_agent()
    assert by_agent["PLANNER"] == {"sent": 110, "received": 55, "cached": 0, "calls": 2}
    assert by_agent["TRANSLATOR"] == {
        "sent": 200, "received": 25, "cached": 120, "calls": 1
    }
    by_model = TokenUsage.snapshot_by_model()
    assert by_model["http://p/v1 / m"] == {
        "sent": 200, "received": 25, "cached": 120, "calls": 1
    }
    TokenUsage.reset()
    assert TokenUsage.snapshot()["sent"] == 0
    assert TokenUsage.snapshot_by_agent() == {}
    assert TokenUsage.snapshot_by_model() == {}


def test_token_cache_cannot_exceed_prompt():
    TokenUsage.reset()
    TokenUsage.add(50, 10, cached_tokens=999)
    assert TokenUsage.snapshot()["cached"] == 50  # clamped to input tokens
    TokenUsage.reset()


@respx.mock
def test_chat_parses_openai_cached_tokens():
    payload = chat_payload("ok")
    payload["usage"]["prompt_tokens"] = 100
    payload["usage"]["prompt_tokens_details"] = {"cached_tokens": 80}
    respx.post(CHAT_URL).respond(json=payload)
    TokenUsage.reset()
    client = LLMClient(make_agent_config())
    resp = client.chat([{"role": "user", "content": "x"}])
    assert resp.cached_tokens == 80
    assert TokenUsage.snapshot()["cached"] == 80
    TokenUsage.reset()


@respx.mock
def test_chat_parses_deepseek_cache_hit_tokens():
    payload = chat_payload("ok")
    payload["usage"] = {
        "prompt_tokens": 100, "completion_tokens": 7,
        "prompt_cache_hit_tokens": 70, "prompt_cache_miss_tokens": 30,
    }
    respx.post(CHAT_URL).respond(json=payload)
    client = LLMClient(make_agent_config())
    resp = client.chat([{"role": "user", "content": "x"}])
    assert resp.prompt_tokens == 100
    assert resp.cached_tokens == 70


@respx.mock
def test_chat_parses_anthropic_style_cache_usage():
    payload = chat_payload("ok")
    payload["usage"] = {
        "input_tokens": 10, "cache_read_input_tokens": 90,
        "cache_creation_input_tokens": 0, "completion_tokens": 7,
    }
    respx.post(CHAT_URL).respond(json=payload)
    client = LLMClient(make_agent_config())
    resp = client.chat([{"role": "user", "content": "x"}])
    assert resp.prompt_tokens == 100  # uncached + cache read
    assert resp.cached_tokens == 90


@respx.mock
def test_cache_status_hit_with_metadata():
    payload = chat_payload("ok")
    payload["usage"]["prompt_tokens"] = 100
    payload["usage"]["prompt_tokens_details"] = {"cached_tokens": 80}
    respx.post(CHAT_URL).respond(json=payload)
    TokenUsage.reset()
    resp = LLMClient(make_agent_config()).chat([{"role": "user", "content": "x"}])
    assert resp.cache_status == "HIT"
    snap = TokenUsage.snapshot()
    assert snap["cache_hits"] == 1 and snap["cache_misses"] == 0
    assert snap["cached"] == 80
    TokenUsage.reset()


@respx.mock
def test_cache_status_miss_with_metadata():
    payload = chat_payload("ok")
    payload["usage"]["prompt_tokens"] = 100
    payload["usage"]["prompt_tokens_details"] = {"cached_tokens": 0}
    respx.post(CHAT_URL).respond(json=payload)
    TokenUsage.reset()
    resp = LLMClient(make_agent_config()).chat([{"role": "user", "content": "x"}])
    assert resp.cache_status == "MISS"
    snap = TokenUsage.snapshot()
    assert snap["cache_misses"] == 1 and snap["cache_hits"] == 0
    TokenUsage.reset()


@respx.mock
def test_cache_status_unknown_without_metadata():
    # A plain payload has no cache fields: absence must NOT be treated as a miss.
    respx.post(CHAT_URL).respond(json=chat_payload("ok"))
    TokenUsage.reset()
    resp = LLMClient(make_agent_config()).chat([{"role": "user", "content": "x"}])
    assert resp.cache_status == "UNKNOWN"
    snap = TokenUsage.snapshot()
    assert snap["cache_unknown"] == 1
    assert snap["cache_hits"] == 0 and snap["cache_misses"] == 0
    TokenUsage.reset()


@respx.mock
def test_explicit_cache_control_marker(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE", "explicit")
    route = respx.post(CHAT_URL).respond(json=chat_payload("ok"))
    client = LLMClient(make_agent_config())
    client.chat(
        [{"role": "system", "content": "stable prefix"},
         {"role": "user", "content": "dynamic"}],
        cache_prefix=True,
    )
    body = route.calls.last.request.read()
    assert b"cache_control" in body


@respx.mock
def test_auto_cache_mode_sends_no_marker(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE", "auto")
    route = respx.post(CHAT_URL).respond(json=chat_payload("ok"))
    client = LLMClient(make_agent_config())
    client.chat(
        [{"role": "system", "content": "stable prefix"},
         {"role": "user", "content": "dynamic"}],
        cache_prefix=True,
    )
    body = route.calls.last.request.read()
    assert b"cache_control" not in body


def test_get_cache_config(monkeypatch):
    from polyglot_lc.config import ConfigError, get_cache_config

    monkeypatch.delenv("PROMPT_CACHE", raising=False)
    assert get_cache_config().mode == "auto"
    monkeypatch.setenv("PROMPT_CACHE", "on")
    assert get_cache_config().mode == "explicit"
    assert get_cache_config().send_markers is True
    monkeypatch.setenv("PROMPT_CACHE", "off")
    assert get_cache_config().enabled is False
    monkeypatch.setenv("PROMPT_CACHE", "nonsense")
    with pytest.raises(ConfigError):
        get_cache_config()


def test_mock_chat_records_agent_tokens(mock_env):
    TokenUsage.reset()
    client = LLMClient(make_agent_config())
    client.chat([{"role": "user", "content": "hello"}])
    by_agent = TokenUsage.snapshot_by_agent()
    assert by_agent["TRANSLATOR"]["calls"] == 1
    assert by_agent["TRANSLATOR"]["sent"] > 0
    assert by_agent["TRANSLATOR"]["received"] > 0
    assert TokenUsage.snapshot()["estimated_calls"] == 1
    TokenUsage.reset()
