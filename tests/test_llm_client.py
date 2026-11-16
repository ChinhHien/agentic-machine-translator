"""Tests for the LLM client with a mocked HTTP server (respx)."""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
import respx

from polyglot_lc.config import AgentConfig, EmbedConfig
from polyglot_lc.llm.client import (
    LLMClient,
    LLMError,
    TokenUsage,
    active_api_calls,
    parse_json_reply,
    reset_provider_gates,
)
from polyglot_lc.llm.resilience import TruncatedError


BASE = "http://mockapi.test/v1"


def make_agent_config(**kw) -> AgentConfig:
    return AgentConfig(
        name="TRANSLATOR",
        base_url=BASE,
        model="test-model",
        api_key="sk-test",
        temperature=0.1,
        **kw,
    )


CHAT_URL = BASE + "/chat/completions"
EMBED_URL = BASE + "/embeddings"


def chat_payload(text: str) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
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
def test_concurrent_calls_never_overlap_on_a_single_request_provider(monkeypatch):
    """The queue is what stops a 1-concurrency provider rejecting our calls.

    A free tier answers the second simultaneous request with 429
    ``concurrency_limit_exceeded``; a chunk lost to that 429 is how a whole
    section disappears from the output.
    """
    monkeypatch.delenv("LLM_PROVIDER_CONCURRENCY", raising=False)
    reset_provider_gates()
    state = {"active": 0, "peak": 0, "rejected": 0}
    lock = threading.Lock()

    def handler(_request: httpx.Request) -> httpx.Response:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            overlap = state["active"] > 1
        time.sleep(0.03)
        with lock:
            state["active"] -= 1
        if overlap:
            state["rejected"] += 1
            return httpx.Response(
                429,
                json={
                    "error": {
                        "code": "concurrency_limit_exceeded",
                        "message": "Too many concurrent requests (limit: 1)",
                    }
                },
            )
        return httpx.Response(200, json=chat_payload("ok"))

    route = respx.post(CHAT_URL)
    route.side_effect = handler
    client = LLMClient(make_agent_config())
    errors: list[BaseException] = []
    results: list[str] = []

    def call() -> None:
        try:
            results.append(client.chat([{"role": "user", "content": "hi"}]).text)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)

    assert not errors, errors
    assert results == ["ok"] * 4
    assert state["peak"] == 1  # the provider never saw two at once
    assert state["rejected"] == 0  # so it never had to reject one


@respx.mock
def test_a_call_in_flight_is_counted_while_it_is_open(monkeypatch):
    """The Report tab's number is this counter, so it has to move around a real
    request: counted from inside the provider while the request is still open,
    back to zero when that request returns, and a peak that remembers the
    high-water mark once nothing is in flight any more.
    """
    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY", "2")
    reset_provider_gates()  # also clears the counter, so the run starts at zero
    seen: dict[str, dict[str, int]] = {}

    def handler(_request: httpx.Request) -> httpx.Response:
        seen["inside"] = active_api_calls()
        return httpx.Response(200, json=chat_payload("ok"))

    route = respx.post(CHAT_URL)
    route.side_effect = handler
    client = LLMClient(make_agent_config())

    assert active_api_calls() == {"active": 0, "peak": 0}
    client.chat([{"role": "user", "content": "hi"}])
    assert seen["inside"] == {"active": 1, "peak": 1}
    # Released with the call, while the peak survives it: a finished run still
    # has its high-water mark to report.
    assert active_api_calls() == {"active": 0, "peak": 1}


@respx.mock
def test_a_concurrency_429_is_survived(monkeypatch):
    """When the queue is widened past the provider's limit, the 429 is retried."""
    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY", "4")
    monkeypatch.setenv("LLM_RETRY_CONCURRENCY_WAIT", "6")
    reset_provider_gates()
    waits: list[float] = []
    monkeypatch.setattr("polyglot_lc.llm.resilience._sleep", waits.append)
    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(
            429,
            json={
                "error": {
                    "code": "concurrency_limit_exceeded",
                    "message": "Too many concurrent requests (limit: 1)",
                }
            },
        ),
        httpx.Response(200, json=chat_payload("ok")),
    ]
    client = LLMClient(make_agent_config())

    assert client.chat([{"role": "user", "content": "hi"}]).text == "ok"
    # Waited for the slot to clear instead of the ~2s exponential value. Jitter
    # is added above the floor (production default 0.2), never below it.
    assert waits and 6.0 <= waits[0] <= 7.2
    assert route.call_count == 2


def _slot_429() -> httpx.Response:
    return httpx.Response(
        429,
        json={
            "error": {
                "code": "concurrency_limit_exceeded",
                "message": "Too many concurrent requests (limit: 1)",
                "limit": 1,
                "role": "free",
            }
        },
    )


@respx.mock
def test_slot_streak_rebuilds_the_http_client(monkeypatch):
    """A connection pool that keeps colliding with a held slot is swapped out.

    A stalled request abandoned mid-flight can leave the provider counting it
    against its one-request limit, so every retry through the same keep-alive
    connection answers 429 ``concurrency_limit_exceeded``. After the
    configured streak, the SDK client is closed (its httpx pool goes with it)
    and the next attempt runs on a fresh one.
    """
    monkeypatch.setenv("LLM_RETRY_SLOT_REBUILD_AFTER", "3")
    route = respx.post(CHAT_URL)
    route.side_effect = [
        _slot_429(),
        _slot_429(),
        _slot_429(),
        httpx.Response(200, json=chat_payload("ok")),
    ]
    client = LLMClient(make_agent_config())
    first_sdk = client._sdk()

    assert client.chat([{"role": "user", "content": "hi"}]).text == "ok"
    # Three rejections in a row hit the threshold on the third, so the fourth
    # request ran on a new client — and the success reset the streak.
    assert client._sdk() is not first_sdk
    assert client._slot_rejects == 0
    assert route.call_count == 4


@respx.mock
def test_slot_streak_below_threshold_keeps_the_client(monkeypatch):
    """One or two collisions are ordinary retries: no pool churn."""
    monkeypatch.setenv("LLM_RETRY_SLOT_REBUILD_AFTER", "5")
    route = respx.post(CHAT_URL)
    route.side_effect = [_slot_429(), httpx.Response(200, json=chat_payload("ok"))]
    client = LLMClient(make_agent_config())
    first_sdk = client._sdk()

    assert client.chat([{"role": "user", "content": "hi"}]).text == "ok"
    assert client._sdk() is first_sdk
    assert client._slot_rejects == 0


@respx.mock
def test_a_success_resets_the_slot_streak(monkeypatch):
    """The streak is *consecutive*: serving one call clears it."""
    monkeypatch.setenv("LLM_RETRY_SLOT_REBUILD_AFTER", "2")
    route = respx.post(CHAT_URL)
    route.side_effect = [
        _slot_429(),
        httpx.Response(200, json=chat_payload("ok")),
        httpx.Response(200, json=chat_payload("ok again")),
    ]
    client = LLMClient(make_agent_config())
    first_sdk = client._sdk()

    client.chat([{"role": "user", "content": "hi"}])
    assert client._slot_rejects == 0
    client.chat([{"role": "user", "content": "hi again"}])
    assert client._sdk() is first_sdk


@respx.mock
def test_non_slot_retries_do_not_count_toward_the_rebuild(monkeypatch):
    """A 500 is not a held slot: the streak only tracks concurrency 429s."""
    monkeypatch.setenv("LLM_RETRY_SLOT_REBUILD_AFTER", "2")
    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(200, json=chat_payload("ok")),
    ]
    client = LLMClient(make_agent_config())
    first_sdk = client._sdk()

    assert client.chat([{"role": "user", "content": "hi"}]).text == "ok"
    assert client._slot_rejects == 0
    assert client._sdk() is first_sdk


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
    respx.post(EMBED_URL).respond(
        json={
            "object": "list",
            "model": "embed-1",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
            ],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
    )
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
    TokenUsage.add(
        200, 25, agent="TRANSLATOR", cached_tokens=120, model="m", provider="http://p/v1"
    )
    TokenUsage.add(10, 5, agent="PLANNER")
    TokenUsage.add(7, 3)  # unattributed call still hits the totals
    snap = TokenUsage.snapshot()
    assert snap["sent"] == 317
    assert snap["received"] == 83
    assert snap["calls"] == 4
    assert snap["cached"] == 120
    by_agent = TokenUsage.snapshot_by_agent()
    assert by_agent["PLANNER"] == {
        "sent": 110,
        "received": 55,
        "cached": 0,
        "thinking": 0,
        "answer": 55,
        "calls": 2,
    }
    assert by_agent["TRANSLATOR"] == {
        "sent": 200,
        "received": 25,
        "cached": 120,
        "thinking": 0,
        "answer": 25,
        "calls": 1,
    }
    by_model = TokenUsage.snapshot_by_model()
    assert by_model["http://p/v1 / m"] == {
        "sent": 200,
        "received": 25,
        "cached": 120,
        "thinking": 0,
        "answer": 25,
        "calls": 1,
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
        "prompt_tokens": 100,
        "completion_tokens": 7,
        "prompt_cache_hit_tokens": 70,
        "prompt_cache_miss_tokens": 30,
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
        "input_tokens": 10,
        "cache_read_input_tokens": 90,
        "cache_creation_input_tokens": 0,
        "completion_tokens": 7,
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
        [{"role": "system", "content": "stable prefix"}, {"role": "user", "content": "dynamic"}],
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
        [{"role": "system", "content": "stable prefix"}, {"role": "user", "content": "dynamic"}],
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

def reasoning_payload(
    text: str, reasoning: str, *, completion: int = 7, reasoning_tokens: int | None = None
) -> dict:
    """A reply whose thinking came back beside the answer."""
    payload = chat_payload(text)
    payload["choices"][0]["message"]["reasoning_content"] = reasoning
    payload["usage"]["completion_tokens"] = completion
    if reasoning_tokens is not None:
        payload["usage"]["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return payload


@respx.mock
def test_a_reasoning_reply_splits_the_budget_into_answer_and_thinking():
    """The provider's own reasoning_tokens is the split, per agent and model.

    A verifier cut at its cap with a full thinking budget is only visible when
    the two shares are counted apart; the totals carry both.
    """
    TokenUsage.reset()
    respx.post(CHAT_URL).respond(
        json=reasoning_payload(
            '{"pass": true, "reasons": []}', "weighing the checks", completion=60, reasoning_tokens=50
        )
    )
    client = LLMClient(make_agent_config())
    resp = client.chat([{"role": "user", "content": "hello"}])

    assert resp.reasoning_text == "weighing the checks"
    assert resp.reasoning_tokens == 50
    assert resp.answer_tokens == 10
    snap = TokenUsage.snapshot()
    assert snap["thinking"] == 50
    assert snap["answer"] == 10
    assert snap["received"] == 60
    assert TokenUsage.snapshot_by_agent()["TRANSLATOR"]["thinking"] == 50
    TokenUsage.reset()


@respx.mock
def test_a_provider_that_reports_no_reasoning_usage_gets_an_estimate():
    """Without reasoning_tokens the thinking is estimated from its own text."""
    TokenUsage.reset()
    respx.post(CHAT_URL).respond(
        json=reasoning_payload("bonjour", "abcd", completion=7)
    )
    client = LLMClient(make_agent_config())
    resp = client.chat([{"role": "user", "content": "hello"}])

    assert resp.reasoning_text == "abcd"
    assert resp.reasoning_tokens == 1  # ~chars/4 of the thinking text
    assert resp.answer_tokens == 6
    TokenUsage.reset()


@respx.mock
def test_a_cut_after_a_transient_blip_is_still_a_truncation(monkeypatch):
    """The retry loop must not turn a cut reply into an outage.

    A 503 on the first ask, then a reply the provider cut at the cap: the
    truncation is what the caller can fix (by raising its cap), so it has to
    arrive as ``TruncatedError`` — not wrapped as "no backup to switch to".
    """
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY", "0.01")
    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(503, json={"error": {"message": "upstream busy"}}),
        httpx.Response(200, json=truncated_payload("cutoff prefix")),
    ]
    client = LLMClient(make_agent_config())
    with pytest.raises(TruncatedError) as excinfo:
        client.chat([{"role": "user", "content": "hello"}])
    assert excinfo.value.partial_text == "cutoff prefix"
    assert route.call_count == 2


def truncated_payload(text: str) -> dict:
    """A reply the provider cut at its output-token limit."""
    payload = chat_payload(text)
    payload["choices"][0]["finish_reason"] = "length"
    return payload


@respx.mock
def test_a_length_cut_is_a_truncation_not_a_draft():
    """finish_reason == "length" must raise, never ship as a draft.

    The reply is a real prefix of the chunk, so it reads like a draft and
    no text heuristic can prove the tail is missing; the provider's own
    verdict is the only honest signal."""
    respx.post(CHAT_URL).respond(json=truncated_payload("cutoff prefix"))
    client = LLMClient(make_agent_config())
    with pytest.raises(TruncatedError) as excinfo:
        client.chat([{"role": "user", "content": "hello"}])
    assert excinfo.value.completion_tokens == 7  # from the mocked usage
    # The endpoint that cut it travels with the error, and so does the text that
    # did arrive — a caller may find a complete answer inside it.
    assert excinfo.value.provider == BASE
    assert excinfo.value.model == "test-model"
    assert excinfo.value.partial_text == "cutoff prefix"
    assert excinfo.value.answer_tokens == 7
    assert excinfo.value.reasoning_tokens == 0


@respx.mock
def test_the_translator_re_asks_a_cut_reply_with_a_raised_cap():
    """A cut reply is re-requested with a bigger output budget, once.

    The retry passes max_tokens explicitly, which is what lifts the cap
    the provider cut the first reply on; the recovered draft is returned
    normally (and lands in the translation memory).
    """
    from polyglot_lc.agents.translator import DEFAULT_TRANSLATOR_OUTPUT_CAP, TranslatorAgent

    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(200, json=truncated_payload("đoạn bị cắt")),
        httpx.Response(200, json=chat_payload("bản dịch đầy đủ")),
    ]
    agent = TranslatorAgent(make_agent_config())
    prompt = agent.build_prompt("Chunk text.", "en", "vi", "translate")
    out = agent.translate(prompt, chunk_id="chunk-000")

    assert out == "bản dịch đầy đủ"
    assert route.call_count == 2
    first = json.loads(route.calls[0].request.content)
    assert "max_tokens" not in first, "the first ask must not set a cap"
    second = json.loads(route.calls[1].request.content)
    assert second.get("max_tokens") == DEFAULT_TRANSLATOR_OUTPUT_CAP


def reasoning_truncated_payload(text: str, *, thinking: int, answer: int) -> dict:
    """A reasoning model's reply cut at the cap, with the budget split reported.

    This is the shape a real run produced: the gateway counts thinking against
    the same completion budget as the answer, so a model that thinks 3070 tokens
    and writes 2 is cut with essentially no translation in hand.
    """
    payload = truncated_payload(text)
    total = thinking + answer
    payload["usage"] = {
        "prompt_tokens": 5,
        "completion_tokens": total,
        "total_tokens": 5 + total,
        "completion_tokens_details": {"reasoning_tokens": thinking},
    }
    return payload


@respx.mock
def test_a_reply_cut_with_no_answer_is_re_asked_with_room_to_think():
    """The recovery budget must fit thinking AND answer, not just the answer.

    A real run fell back to a second endpoint whose default cap was 3072; the
    model spent 3070 of it thinking and emitted 2 tokens of translation. The
    recovery ask then clamped the primary endpoint — which normally runs
    uncapped to a median of ~9000 completion tokens — down to 4096, so it cut
    again and the chunk was reported as a failure it had not earned.
    """
    from polyglot_lc.agents.translator import DEFAULT_TRANSLATOR_OUTPUT_CAP, TranslatorAgent

    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(200, json=reasoning_truncated_payload("d", thinking=3070, answer=2)),
        httpx.Response(200, json=chat_payload("bản dịch đầy đủ")),
    ]
    agent = TranslatorAgent(make_agent_config())
    out = agent.translate(agent.build_prompt("Chunk text.", "en", "vi", "translate"))

    assert out == "bản dịch đầy đủ"
    assert route.call_count == 2
    first = json.loads(route.calls[0].request.content)
    assert "max_tokens" not in first, "the first ask must not set a cap"
    cap = json.loads(route.calls[1].request.content)["max_tokens"]
    assert cap > 3072, "a recovery below the budget just cut cannot succeed"
    assert cap >= DEFAULT_TRANSLATOR_OUTPUT_CAP


@respx.mock
def test_the_translator_recovery_cap_is_configurable(monkeypatch):
    """An endpoint with a different ceiling is tuned without editing code."""
    from polyglot_lc.agents.translator import TranslatorAgent

    monkeypatch.setenv("TRANSLATOR_MAX_TOKENS", "8192")
    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(200, json=truncated_payload("đoạn bị cắt")),
        httpx.Response(200, json=chat_payload("bản dịch đầy đủ")),
    ]
    agent = TranslatorAgent(make_agent_config())
    agent.translate(agent.build_prompt("Chunk text.", "en", "vi", "translate"))

    assert json.loads(route.calls[1].request.content)["max_tokens"] == 8192


@respx.mock
def test_a_configured_cap_below_the_cut_yields_to_the_evidence(monkeypatch):
    """A tuned cap smaller than the budget just cut is a guaranteed failure.

    The provider said how much it consumed before stopping; asking for less
    than that cannot produce a longer reply, so the observed cut wins.
    """
    from polyglot_lc.agents.translator import TranslatorAgent

    monkeypatch.setenv("TRANSLATOR_MAX_TOKENS", "2048")
    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(200, json=reasoning_truncated_payload("d", thinking=3070, answer=2)),
        httpx.Response(200, json=chat_payload("bản dịch đầy đủ")),
    ]
    agent = TranslatorAgent(make_agent_config())
    agent.translate(agent.build_prompt("Chunk text.", "en", "vi", "translate"))

    assert json.loads(route.calls[1].request.content)["max_tokens"] >= 6144


@respx.mock
def test_a_second_cut_escalates_once_more_then_reports_the_truncation():
    """The ladder is bounded: two recovery asks, then the caller is told.

    Each rung doubles the budget that was just cut, so a reply that overran one
    budget gets room instead of the same question again. At the ceiling there is
    nothing left to raise and the error propagates to the chunk layer, which
    salvages by splitting rather than spinning.
    """
    from polyglot_lc.agents.translator import (
        DEFAULT_TRANSLATOR_OUTPUT_CAP,
        TRANSLATOR_MAX_TOKENS_CEILING,
        TranslatorAgent,
    )

    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(200, json=reasoning_truncated_payload("a", thinking=3070, answer=2)),
        httpx.Response(
            200,
            json=reasoning_truncated_payload(
                "b", thinking=15000, answer=DEFAULT_TRANSLATOR_OUTPUT_CAP - 15000
            ),
        ),
        httpx.Response(
            200,
            json=reasoning_truncated_payload(
                "c", thinking=30000, answer=TRANSLATOR_MAX_TOKENS_CEILING - 30000
            ),
        ),
    ]
    agent = TranslatorAgent(make_agent_config())
    with pytest.raises(TruncatedError):
        agent.translate(agent.build_prompt("Chunk text.", "en", "vi", "translate"))

    assert route.call_count == 3
    caps = [json.loads(c.request.content).get("max_tokens") for c in route.calls]
    assert caps == [None, DEFAULT_TRANSLATOR_OUTPUT_CAP, TRANSLATOR_MAX_TOKENS_CEILING]


@respx.mock
def test_the_escalation_never_asks_above_the_ceiling(monkeypatch):
    """A gateway rejects a max_tokens over its own limit outright, so the
    ladder stops at the ceiling instead of doubling past it."""
    from polyglot_lc.agents.translator import TRANSLATOR_MAX_TOKENS_CEILING, TranslatorAgent

    monkeypatch.setenv("TRANSLATOR_MAX_TOKENS", str(TRANSLATOR_MAX_TOKENS_CEILING))
    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(
            200,
            json=reasoning_truncated_payload(
                "a", thinking=30000, answer=TRANSLATOR_MAX_TOKENS_CEILING - 30000
            ),
        ),
        httpx.Response(
            200,
            json=reasoning_truncated_payload(
                "b", thinking=30000, answer=TRANSLATOR_MAX_TOKENS_CEILING - 30000
            ),
        ),
    ]
    agent = TranslatorAgent(make_agent_config())
    with pytest.raises(TruncatedError):
        agent.translate(agent.build_prompt("Chunk text.", "en", "vi", "translate"))

    caps = [json.loads(c.request.content).get("max_tokens") for c in route.calls]
    assert caps == [None, TRANSLATOR_MAX_TOKENS_CEILING]
    assert route.call_count == 2, "at the ceiling a second recovery ask is the same question"
