"""The per-provider request queue.

A free tier that serves one request at a time answers the second one with 429
``concurrency_limit_exceeded``. Retrying that is worse than queueing — the
request it collided with is still running — so calls wait their turn instead,
and a queued call must not have that waiting charged against its own deadline.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from polyglot_lc.config import AgentConfig
from polyglot_lc.llm import client as client_mod
from polyglot_lc.llm.client import LLMClient, reset_provider_gates
from polyglot_lc.llm.resilience import RateLimitError

BASE = "http://provider.test/v1"
OTHER = "http://other.test/v1"


@pytest.fixture(autouse=True)
def _real_path(monkeypatch):
    """Leave the real call path (the mock path never touches the queue)."""
    monkeypatch.setenv("LLM_MOCK", "0")
    monkeypatch.delenv("LLM_PROVIDER_CONCURRENCY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_CONCURRENCY_MAX", raising=False)
    monkeypatch.delenv("LLM_RETRY_CONCURRENCY_WAIT", raising=False)
    reset_provider_gates()
    yield
    reset_provider_gates()


def _client(
    base_url: str, *, timeout: float = 5.0, model: str = "m", name: str = "TRANSLATOR"
) -> LLMClient:
    return LLMClient(
        AgentConfig(
            name=name,
            base_url=base_url,
            model=model,
            api_key="sk-test",
            temperature=0.0,
            timeout=timeout,
        )
    )


class _Provider:
    """A fake SDK whose ``create`` records how many calls overlap."""

    def __init__(
        self, work: float = 0.05, barrier: threading.Barrier | None = None, fail_first: bool = False
    ) -> None:
        self.work = work
        self.barrier = barrier
        self.fail_first = fail_first
        self.active = 0
        self.peak = 0
        self.calls = 0
        self._lock = threading.Lock()

    def create(self, **_kw):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
            first = self.calls == 1
        try:
            if self.fail_first and first:
                raise RuntimeError("429 concurrency_limit_exceeded: limit: 1")
            if self.barrier is not None:
                self.barrier.wait(timeout=2.0)
            else:
                time.sleep(self.work)
        finally:
            with self._lock:
                self.active -= 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )


def _install(monkeypatch, client: LLMClient, provider: _Provider) -> None:
    monkeypatch.setattr(
        client,
        "_sdk",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=provider.create))
        ),
    )


def _call(client: LLMClient) -> None:
    client._raw_chat([{"role": "user", "content": "hi"}])


def _run(times: int, fn) -> list[BaseException]:
    """Run *fn* on *times* threads; return whatever they raised."""
    errors: list[BaseException] = []

    def target() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=target) for _ in range(times)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)
    return errors


# ------------------------------- the queue ----------------------------------
def test_a_single_concurrency_provider_serves_one_request_at_a_time(monkeypatch):
    provider = _Provider(work=0.05)
    client = _client(BASE)
    _install(monkeypatch, client, provider)

    assert not _run(4, lambda: _call(client))
    assert provider.calls == 4
    assert provider.peak == 1  # the queue serialised them


def test_the_queue_admits_up_to_the_configured_limit(monkeypatch):
    """Three workers must be able to be inside the provider together."""
    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY", "3")
    reset_provider_gates()
    provider = _Provider(barrier=threading.Barrier(3))
    client = _client(BASE)
    _install(monkeypatch, client, provider)

    errors = _run(3, lambda: _call(client))
    assert not errors, errors
    assert not provider.barrier.broken  # all three met inside the provider
    assert provider.peak == 3


def test_models_behind_one_provider_share_its_queue(monkeypatch):
    """A configured limit is for the provider, not for each model name.

    Translator, Critic and Verifier often use different model ids on the same
    gateway. Keying the queue by model let each of them admit the full limit, so
    the provider saw two or three times the configured concurrency.
    """
    provider = _Provider(work=0.2)
    one, two = _client(BASE, model="model-a"), _client(BASE, model="model-b")
    _install(monkeypatch, one, provider)
    _install(monkeypatch, two, provider)

    errors: list[BaseException] = []

    def call(client: LLMClient) -> None:
        try:
            _call(client)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(c,)) for c in (one, two)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)
    assert not errors, errors
    assert provider.peak == 1


def test_planner_calls_are_serialised(monkeypatch):
    """Planner runs one call at a time even when the provider allows more."""
    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY", "3")
    reset_provider_gates()
    provider = _Provider(work=0.15)
    one = _client(BASE, name="PLANNER")
    two = _client(BASE, name="PLANNER")
    _install(monkeypatch, one, provider)
    _install(monkeypatch, two, provider)

    errors: list[BaseException] = []

    def call(client: LLMClient) -> None:
        try:
            _call(client)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(c,)) for c in (one, two)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)
    assert not errors, errors
    assert provider.peak == 1


def test_the_queue_is_per_provider(monkeypatch):
    """Two endpoints must not queue behind each other.

    Both providers hold one call at a time, and the barrier only releases once
    two calls are inside it at the same moment — which is impossible if the two
    endpoints share a queue.
    """
    barrier = threading.Barrier(2)
    one, two = _client(BASE), _client(OTHER)
    _install(monkeypatch, one, _Provider(barrier=barrier))
    _install(monkeypatch, two, _Provider(barrier=barrier))

    errors: list[BaseException] = []

    def call(client: LLMClient) -> None:
        try:
            _call(client)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(c,)) for c in (one, two)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)
    assert not errors, errors
    assert not barrier.broken, "different providers shared one queue"


def test_a_queued_call_is_not_killed_by_its_own_deadline(monkeypatch):
    """Waiting a turn is queue time, not request time.

    The waiting call's deadline is far shorter than the wait, so a deadline
    started before the queue would abort it with a timeout it never earned.
    """
    monkeypatch.setattr(client_mod, "_DEADLINE_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(client_mod, "_DEADLINE_DRAIN_SECONDS", 0.05)
    slow = _Provider(work=0.25)
    fast = _Provider(work=0.0)
    holder = _client(BASE, timeout=5.0)
    waiter = _client(BASE, timeout=0.05)  # deadline ~0.1s, far below the wait
    _install(monkeypatch, holder, slow)
    _install(monkeypatch, waiter, fast)

    started = threading.Thread(target=lambda: _call(holder))
    started.start()
    deadline = time.monotonic() + 2.0
    while slow.calls == 0 and time.monotonic() < deadline:
        time.sleep(0.005)

    began = time.monotonic()
    _call(waiter)  # must queue, then run — not time out while waiting
    waited = time.monotonic() - began
    started.join(5.0)
    assert waited >= 0.2
    assert fast.calls == 1


def test_a_failed_request_releases_the_queue(monkeypatch):
    """The slot must come back when the provider rejects the call."""
    provider = _Provider(work=0.0, fail_first=True)
    client = _client(BASE)
    _install(monkeypatch, client, provider)

    with pytest.raises(RateLimitError):
        _call(client)

    done = threading.Event()

    def second() -> None:
        _call(client)
        done.set()

    thread = threading.Thread(target=second, daemon=True)
    thread.start()
    assert done.wait(3.0), "the queue was never released"
    assert provider.calls == 2


# --------------------------- adaptive concurrency ---------------------------
def test_without_a_ceiling_the_queue_never_moves():
    """The default stays exactly what the operator configured."""
    gate = client_mod._ProviderGate(2, 0)
    for _ in range(50):
        gate.observe_success()
        gate.observe_throttle()
    state = gate.snapshot()
    assert not state["adaptive"]
    assert state["limit"] == 2
    assert state["grows"] == 0 and state["throttles"] == 0


def test_consecutive_successes_widen_the_queue_to_its_ceiling():
    gate = client_mod._ProviderGate(1, 3)
    for _ in range(client_mod._GROW_AFTER):
        gate.observe_success()
    assert gate.snapshot()["limit"] == 2
    for _ in range(2 * client_mod._GROW_AFTER):
        gate.observe_success()
    assert gate.snapshot()["limit"] == 3
    for _ in range(10 * client_mod._GROW_AFTER):
        gate.observe_success()
    state = gate.snapshot()
    assert state["limit"] == 3  # the ceiling holds
    assert state["grows"] == 2


def test_a_slot_rejection_halves_the_queue_and_stops_at_one():
    gate = client_mod._ProviderGate(4, 8)
    gate.observe_throttle()
    assert gate.snapshot()["limit"] == 2
    gate.observe_throttle()
    assert gate.snapshot()["limit"] == 1
    gate.observe_throttle()
    state = gate.snapshot()
    assert state["limit"] == 1
    assert state["throttles"] == 3


def test_a_rejection_restarts_the_success_run():
    """Five successes before a rejection must not count toward the next slot."""
    gate = client_mod._ProviderGate(1, 4)
    for _ in range(client_mod._GROW_AFTER - 1):
        gate.observe_success()
    gate.observe_throttle()
    gate.observe_success()
    assert gate.snapshot()["limit"] == 1


def test_the_queue_counts_its_peak_in_flight():
    gate = client_mod._ProviderGate(2, 2)
    with gate, gate:
        assert gate.snapshot()["in_flight"] == 2
    state = gate.snapshot()
    assert state["in_flight"] == 0
    assert state["peak"] == 2


def test_a_served_request_widens_a_queue_that_has_a_ceiling(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY", "1")
    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY_MAX", "4")
    reset_provider_gates()
    provider = _Provider(work=0.0)
    client = _client(BASE)
    _install(monkeypatch, client, provider)

    for _ in range(client_mod._GROW_AFTER):
        _call(client)

    state = client_mod.provider_gate_stats()[BASE]
    assert state["adaptive"]
    assert state["limit"] == 2


def test_a_slot_rejection_from_the_provider_halves_the_queue(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY", "4")
    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY_MAX", "8")
    reset_provider_gates()
    client = _client(BASE)

    client._note_retry("429 concurrency_limit_exceeded: another request is running")

    state = client_mod.provider_gate_stats()[BASE]
    assert state["start"] == 4
    assert state["limit"] == 2
    assert state["throttles"] == 1


def test_a_ceiling_below_the_configured_limit_is_refused(monkeypatch):
    from polyglot_lc.config import ConfigError, get_retry_config

    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY", "4")
    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY_MAX", "2")
    with pytest.raises(ConfigError, match="LLM_PROVIDER_CONCURRENCY_MAX"):
        get_retry_config()
