"""Tests for the resilience layer: rate limits, backoff, backup fallback."""

from __future__ import annotations

import httpx
import pytest
import respx
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError as SDKRateLimitError,
)

from polyglot_lc.config import AgentConfig, ConfigError, RetryConfig, get_retry_config
from polyglot_lc.llm.client import LLMClient
from polyglot_lc.llm.resilience import (
    LLMError,
    PermanentError,
    RateLimitError,
    RetryStats,
    TransientError,
    backoff_delay,
    call_with_resilience,
    jittered_delay,
    normalize_provider_error,
    parse_retry_seconds,
)

BASE = "http://mockapi.test/v1"
CHAT_URL = BASE + "/chat/completions"
BACKUP_URL = "http://backup.test/v1/chat/completions"


def make_agent_config(**kw) -> AgentConfig:
    return AgentConfig(name="TRANSLATOR", base_url=BASE, model="test-model",
                       api_key="sk-test", temperature=0.1, **kw)


def chat_payload(text: str, model: str = "test-model") -> dict:
    return {
        "id": "c1", "object": "chat.completion", "created": 1, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def ratelimit_response(hint: str = "try again in 1s") -> httpx.Response:
    return httpx.Response(
        429, json={"error": {"message": f"rate_limit_exceeded: Please {hint}"}},
        request=httpx.Request("POST", CHAT_URL),
    )


@pytest.fixture(autouse=True)
def _fast_resilience(monkeypatch):
    """Never actually sleep in tests; capture waits via _waits when needed."""
    waits: list[float] = []
    monkeypatch.setattr("polyglot_lc.llm.resilience._sleep", waits.append)
    RetryStats.reset()
    yield waits
    RetryStats.reset()


# ---------------------------------------------------------------------------
# parse_retry_seconds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Please try again in 47.4s", 47.4),
        ("rate limit exceeded, try again in 3s", 3.0),
        ("Try again in 900ms", 0.9),
        ("please try again in 2 mins", 120.0),
        ("Retry-After: 12", 12.0),
        ("retry_after=1.5", 1.5),
        ("no hint here", None),
        ("", None),
    ],
)
def test_parse_retry_seconds(text, expected):
    assert parse_retry_seconds(text) == expected


# ---------------------------------------------------------------------------
# normalize_provider_error
# ---------------------------------------------------------------------------
def test_normalize_sdk_ratelimit_error():
    exc = SDKRateLimitError(
        "Please try again in 47.4s",
        response=httpx.Response(429, request=httpx.Request("POST", CHAT_URL)),
        body=None,
    )
    err = normalize_provider_error(exc)
    assert isinstance(err, RateLimitError)
    assert err.retry_after == 47.4
    assert err.status == 429


def test_normalize_timeout_and_connection():
    assert isinstance(normalize_provider_error(APITimeoutError(
        httpx.Request("POST", CHAT_URL))), TransientError)
    assert isinstance(normalize_provider_error(APIConnectionError(
        request=httpx.Request("POST", CHAT_URL))), TransientError)


def test_normalize_status_codes():
    req = httpx.Request("POST", CHAT_URL)
    e500 = normalize_provider_error(httpx.HTTPStatusError(
        "server error", request=req, response=httpx.Response(500, request=req)))
    assert isinstance(e500, TransientError) and e500.status == 500
    e429 = normalize_provider_error(RuntimeError("HTTP 429 too many requests"))
    assert isinstance(e429, RateLimitError)
    e401 = normalize_provider_error(RuntimeError("status 401 unauthorized"))
    assert isinstance(e401, PermanentError)


def test_permanent_error_fallback_class():
    err = normalize_provider_error(ValueError("bad request"))
    assert isinstance(err, PermanentError)


# ---------------------------------------------------------------------------
# backoff + config
# ---------------------------------------------------------------------------
def test_backoff_sequence():
    cfg = RetryConfig(max_retries=5, base_delay=2.0, max_delay=60.0)
    assert [backoff_delay(cfg, i) for i in range(5)] == [2.0, 4.0, 8.0, 16.0, 32.0]
    capped = RetryConfig(base_delay=40.0, max_delay=60.0)
    assert backoff_delay(capped, 1) == 60.0


def test_backoff_jitter_bounds():
    # jitter=0 -> exact exponential (deterministic)
    cfg0 = RetryConfig(base_delay=2.0, max_delay=60.0, jitter=0.0)
    assert jittered_delay(cfg0, 1) == 4.0
    # jitter=0.5 -> wait stays within [0.5*backoff, backoff]
    cfg = RetryConfig(base_delay=2.0, max_delay=60.0, jitter=0.5)
    for _ in range(50):
        w = jittered_delay(cfg, 2)  # plain backoff = 8.0
        assert 4.0 <= w <= 8.0


def test_get_retry_config_defaults(mock_env):
    cfg = get_retry_config()
    assert cfg.max_retries == 3
    assert cfg.base_delay == 2.0
    assert cfg.max_delay == 60.0
    assert cfg.buffer_seconds == 2.0
    assert cfg.jitter == 0.2  # production default; jitter bounded retries


def test_get_retry_config_overrides(mock_env, monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY", "1")
    monkeypatch.setenv("LLM_RETRY_MAX_DELAY", "30")
    monkeypatch.setenv("LLM_RETRY_BUFFER_SECONDS", "1.5")
    cfg = get_retry_config()
    assert (cfg.max_retries, cfg.base_delay, cfg.max_delay,
            cfg.buffer_seconds) == (5, 1.0, 30.0, 1.5)


def test_get_retry_config_bad_values(mock_env, monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "many")
    with pytest.raises(ConfigError, match="LLM_MAX_RETRIES"):
        get_retry_config()
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY", "99")
    with pytest.raises(ConfigError, match="LLM_RETRY_BASE_DELAY"):
        get_retry_config()


# ---------------------------------------------------------------------------
# call_with_resilience
# ---------------------------------------------------------------------------
def test_resilience_success_first_try():
    calls = []
    assert call_with_resilience(lambda *a, **k: calls.append(1) or "ok", None,
                                retry_cfg=RetryConfig()) == "ok"
    assert len(calls) == 1


def test_resilience_ratelimit_hint_plus_buffer(_fast_resilience):
    def flaky():
        raise RateLimitError("try again in 47.4s", retry_after=47.4, status=429)

    cfg = RetryConfig(max_retries=1, base_delay=2.0, max_delay=60.0,
                      buffer_seconds=2.0)
    with pytest.raises(RateLimitError):
        call_with_resilience(flaky, None, retry_cfg=cfg)
    # One wait: provider hint 47.4s + 2.0s buffer.
    assert _fast_resilience == [49.4]
    assert RetryStats.snapshot()["rate_limit_retries"] == 1


def test_resilience_ratelimit_hint_capped_at_max_delay(_fast_resilience):
    def always():
        raise RateLimitError("try again in 900s", retry_after=900.0, status=429)

    cfg = RetryConfig(max_retries=2, base_delay=2.0, max_delay=60.0,
                      buffer_seconds=2.0)
    with pytest.raises(RateLimitError):
        call_with_resilience(always, None, retry_cfg=cfg)
    assert _fast_resilience == [60.0, 60.0]
    assert RetryStats.snapshot()["rate_limit_retries"] == 2


def test_resilience_ratelimit_exponential_when_no_hint(_fast_resilience):
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] <= 2:
            raise RateLimitError("rate_limit_exceeded", status=429)
        return "ok"

    cfg = RetryConfig(max_retries=5, base_delay=2.0, max_delay=60.0)
    assert call_with_resilience(flaky, None, retry_cfg=cfg) == "ok"
    assert _fast_resilience == [2.0, 4.0]


def test_resilience_ratelimit_refreshes_hint_between_retries(_fast_resilience):
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        hint = 5.0 if state["n"] == 1 else 8.0
        if state["n"] <= 2:
            raise RateLimitError(f"try again in {hint}s", retry_after=hint,
                                 status=429)
        return "ok"

    cfg = RetryConfig(max_retries=3, buffer_seconds=1.0)
    assert call_with_resilience(flaky, None, retry_cfg=cfg) == "ok"
    assert _fast_resilience == [6.0, 9.0]  # hint + 1.0s buffer each time


def test_resilience_transient_retries_then_success(_fast_resilience):
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] <= 2:
            raise TransientError("connection reset", status=503)
        return "ok"

    cfg = RetryConfig(max_retries=4, base_delay=2.0, max_delay=60.0)
    assert call_with_resilience(flaky, None, retry_cfg=cfg) == "ok"
    assert _fast_resilience == [2.0, 4.0]
    assert RetryStats.snapshot()["transient_retries"] == 2


def test_resilience_permanent_never_retries_or_falls_back(_fast_resilience):
    calls = []

    def bad():
        calls.append(1)
        raise PermanentError("invalid api key", status=401)

    with pytest.raises(PermanentError):
        call_with_resilience(bad, lambda *a, **k: "backup",
                             retry_cfg=RetryConfig(max_retries=3))
    assert len(calls) == 1
    assert _fast_resilience == []
    assert RetryStats.snapshot()["fallbacks"] == 0


def test_resilience_exhausted_falls_back_to_backup():
    def always_429():
        raise RateLimitError("try again in 5s", retry_after=5.0, status=429)

    out = call_with_resilience(always_429, lambda *a, **k: "from-backup",
                               retry_cfg=RetryConfig(max_retries=3))
    assert out == "from-backup"
    stats = RetryStats.snapshot()
    assert stats["rate_limit_retries"] == 3
    assert stats["fallbacks"] == 1


def test_resilience_timeout_exhausted_falls_back():
    def always_timeout():
        raise TransientError("request timed out")

    out = call_with_resilience(always_timeout, lambda *a, **k: "backup-ok",
                               retry_cfg=RetryConfig(max_retries=2))
    assert out == "backup-ok"
    assert RetryStats.snapshot()["fallbacks"] == 1


def test_resilience_no_backup_reraises_last_error():
    def always():
        raise TransientError("boom", status=503)

    with pytest.raises(TransientError):
        call_with_resilience(always, None, retry_cfg=RetryConfig(max_retries=1))


def test_resilience_backup_failure_propagates():
    def always():
        raise RateLimitError("429", status=429)

    def backup():
        raise PermanentError("backup also broken", status=401)

    with pytest.raises(PermanentError):
        call_with_resilience(always, backup, retry_cfg=RetryConfig(max_retries=1))


def test_zero_retries_falls_back_immediately(_fast_resilience):
    calls = []

    def primary():
        calls.append(1)
        raise TransientError("timeout")

    out = call_with_resilience(primary, lambda *a, **k: "backup",
                               retry_cfg=RetryConfig(max_retries=0))
    assert out == "backup"
    assert len(calls) == 1
    assert _fast_resilience == []


# ---------------------------------------------------------------------------
# LLMClient end-to-end (respx-mocked HTTP)
# ---------------------------------------------------------------------------
@respx.mock
def test_infra_transient_retry_counts_one_request(monkeypatch):
    from polyglot_lc.llm.client import TokenUsage

    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(503, json={"error": "unavailable"}),
        httpx.Response(200, json=chat_payload("ok")),
    ]
    TokenUsage.reset()
    resp = LLMClient(make_agent_config()).chat([{"role": "user", "content": "hi"}])
    assert resp.text == "ok"
    assert route.call_count == 2  # initial attempt + one infra retry
    assert TokenUsage.snapshot()["calls"] == 1  # retries are not extra requests
    assert RetryStats.snapshot()["transient_retries"] == 1
    TokenUsage.reset()


@respx.mock
def test_chat_retries_on_429_with_hint_then_succeeds(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    route = respx.post(CHAT_URL)
    route.side_effect = [
        ratelimit_response(),
        httpx.Response(200, json=chat_payload("ok")),
    ]
    client = LLMClient(make_agent_config())
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.text == "ok"
    assert resp.via_backup is False
    assert route.call_count == 2


@respx.mock
def test_chat_falls_back_to_backup_after_429_exhausted(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    primary = respx.post(CHAT_URL)
    primary.side_effect = [ratelimit_response(), ratelimit_response(),
                           ratelimit_response()]
    respx.post(BACKUP_URL).respond(json=chat_payload("from backup",
                                                     model="backup-model"))
    client = LLMClient(make_agent_config(
        backup_base_url="http://backup.test/v1", backup_model="backup-model"))
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.text == "from backup"
    assert resp.model == "backup-model"
    assert resp.via_backup is True
    assert primary.call_count == 3  # 1 initial + 2 retries
    assert RetryStats.snapshot()["fallbacks"] == 1


@respx.mock
def test_chat_permanent_401_does_not_retry_or_fallback(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    route = respx.post(CHAT_URL).respond(status_code=401,
                                         json={"error": {"message": "bad key"}})
    client = LLMClient(make_agent_config(
        backup_base_url="http://backup.test/v1", backup_model="backup-model"))
    with pytest.raises(LLMError) as exc_info:
        client.chat([{"role": "user", "content": "hi"}])
    assert isinstance(exc_info.value, PermanentError)
    assert route.call_count == 1
    assert RetryStats.snapshot()["fallbacks"] == 0


@respx.mock
def test_chat_without_backup_reraises_after_exhaustion(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    route = respx.post(CHAT_URL)
    route.side_effect = [ratelimit_response(), ratelimit_response()]
    client = LLMClient(make_agent_config())
    with pytest.raises(RateLimitError):
        client.chat([{"role": "user", "content": "hi"}])
    assert route.call_count == 2


# ---------------------------------------------------------------------------
# Backup config wiring
# ---------------------------------------------------------------------------
def test_backup_config_pair_validation():
    with pytest.raises(Exception, match="BACKUP"):
        make_agent_config(backup_base_url="http://backup.test/v1")


def test_has_backup_property():
    assert make_agent_config().has_backup is False
    assert make_agent_config(backup_base_url="http://b.test/v1",
                             backup_model="m").has_backup is True


def test_backup_client_uses_backup_config():
    client = LLMClient(make_agent_config(
        backup_base_url="http://b.test/v1", backup_model="m",
        backup_api_key="sk-b"))
    backup = client._backup_client()
    assert backup is not None
    assert backup.config.base_url == "http://b.test/v1"
    assert backup.config.model == "m"
    assert backup.config.api_key == "sk-b"
    # Backups never nest: the backup client has no backup of its own.
    assert backup._backup_client() is None
