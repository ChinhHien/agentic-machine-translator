"""The hard per-request deadline that stops a wedged provider stalling a run."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from polyglot_lc.config import AgentConfig
from polyglot_lc.llm import client as client_mod
from polyglot_lc.llm.client import LLMClient
from polyglot_lc.llm.resilience import TransientError

BASE = "http://mockapi.test/v1"


def _hanging_client(monkeypatch, wait_s: float = 5.0, **cfg) -> LLMClient:
    """A client whose provider never answers."""
    monkeypatch.setattr(client_mod, "_DEADLINE_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(client_mod, "_DEADLINE_DRAIN_SECONDS", 0.05)
    config = AgentConfig(
        name="CRITIC",
        base_url=BASE,
        model="m",
        api_key="sk-test",
        temperature=0.0,
        timeout=0.05,
        **cfg,
    )
    client = LLMClient(config)
    release = threading.Event()

    def create(**_kw):
        release.wait(wait_s)
        return SimpleNamespace(choices=[])

    monkeypatch.setattr(
        client,
        "_sdk",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    return client


def test_wedged_request_raises_a_retryable_timeout(monkeypatch):
    """A provider that never answers must not stall the run indefinitely."""
    client = _hanging_client(monkeypatch)
    started = time.monotonic()
    with pytest.raises(TransientError):
        client._raw_chat([{"role": "user", "content": "hi"}])
    assert time.monotonic() - started < 3.0


def test_successful_request_is_returned_unchanged(monkeypatch):
    """The deadline wrapper is transparent when the call answers in time."""
    monkeypatch.setattr(client_mod, "_DEADLINE_GRACE_SECONDS", 5.0)
    config = AgentConfig(
        name="CRITIC",
        base_url=BASE,
        model="m",
        api_key="sk-test",
        temperature=0.0,
        timeout=1.0,
    )
    client = LLMClient(config)
    payload = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=1,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )
    monkeypatch.setattr(
        client,
        "_sdk",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kw: payload))
        ),
    )
    assert client._raw_chat([{"role": "user", "content": "hi"}]).text == "ok"


def test_slow_request_finishing_within_the_drain_is_not_wasted(monkeypatch):
    """A request that overruns the deadline but lands in the drain is used.

    Retrying while the old request is still in flight is what makes a
    concurrency-limited provider answer 429, so the drain prefers the late
    answer over a retry.
    """
    released = threading.Event()
    monkeypatch.setattr(client_mod, "_DEADLINE_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(client_mod, "_DEADLINE_DRAIN_SECONDS", 5.0)
    config = AgentConfig(
        name="CRITIC",
        base_url=BASE,
        model="m",
        api_key="sk-test",
        temperature=0.0,
        timeout=0.05,
    )
    client = LLMClient(config)
    payload = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="late"))],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )

    def create(**_kw):
        released.wait(1.0)
        return payload

    threading.Timer(0.5, released.set).start()
    monkeypatch.setattr(
        client,
        "_sdk",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    assert client._raw_chat([{"role": "user", "content": "hi"}]).text == "late"
