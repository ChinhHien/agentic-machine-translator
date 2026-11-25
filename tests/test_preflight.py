"""The preflight canary: one short call per agent before a document is paid for."""

from __future__ import annotations

from polyglot_lc.config import AgentConfig
from polyglot_lc.llm.client import LLMClient
from polyglot_lc.pipeline.preflight import run_preflight


def _config(name: str = "TRANSLATOR") -> AgentConfig:
    return AgentConfig(
        name=name,
        base_url="http://mock.local/v1",
        model="mock-model",
        api_key="sk-mock",
    )


def test_the_mock_harness_is_skipped(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")
    report = run_preflight({"TRANSLATOR": _config()})
    assert report.ok
    # One check per agent, all skipped: the mock proves nothing about a provider.
    assert {check.detail for check in report.checks} == {"mock"}
    assert len(report.checks) == 3


def test_a_failing_agent_is_named_in_the_summary(monkeypatch):
    monkeypatch.delenv("LLM_MOCK", raising=False)

    def boom(self, *args, **kwargs):
        raise RuntimeError("401 invalid api key")

    monkeypatch.setattr(LLMClient, "chat", boom)
    report = run_preflight({"TRANSLATOR": _config(), "VERIFIER": _config("VERIFIER")})
    assert not report.ok
    assert "TRANSLATOR" in report.summary()
    assert "401" in report.summary()


def test_a_working_agent_passes(monkeypatch):
    monkeypatch.delenv("LLM_MOCK", raising=False)

    class Reply:
        text = "ok"

    monkeypatch.setattr(LLMClient, "chat", lambda self, *a, **k: Reply())
    report = run_preflight({"PLANNER": _config("PLANNER")})
    assert report.ok
    assert report.as_dicts()[0]["agent"] == "PLANNER"
