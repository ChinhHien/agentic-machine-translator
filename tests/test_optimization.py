"""Tests for token/request optimizations: source-group routing, cacheable
shared prefix, reduced summary-tree requests, planner-once, verifier scope and
the CLI token report.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from polyglot_lc.llm.client import LLMResponse, TokenUsage


# ------------------------- hybrid source-group routing ----------------------
def test_source_group_provides_shared_provider(monkeypatch):
    for name in ("TRANSLATOR", "CRITIC", "REFINER"):
        for suffix in ("BASE_URL", "MODEL", "API_KEY", "TEMPERATURE"):
            monkeypatch.delenv(f"{name}_{suffix}", raising=False)
    monkeypatch.setenv("LLM_MOCK", "0")
    monkeypatch.setenv("SOURCE_GROUP_BASE_URL", "http://shared.local/v1")
    monkeypatch.setenv("SOURCE_GROUP_MODEL", "shared-model")
    monkeypatch.setenv("SOURCE_GROUP_API_KEY", "sk-shared")

    from polyglot_lc.config import get_agent_config

    for name in ("TRANSLATOR", "CRITIC", "REFINER"):
        cfg = get_agent_config(name)
        assert cfg.base_url == "http://shared.local/v1"
        assert cfg.model == "shared-model"
        assert cfg.api_key == "sk-shared"


def test_per_agent_override_beats_source_group(monkeypatch):
    for suffix in ("BASE_URL", "MODEL", "API_KEY", "TEMPERATURE"):
        monkeypatch.delenv(f"CRITIC_{suffix}", raising=False)
    monkeypatch.setenv("LLM_MOCK", "0")
    monkeypatch.setenv("SOURCE_GROUP_BASE_URL", "http://shared.local/v1")
    monkeypatch.setenv("SOURCE_GROUP_MODEL", "shared-model")
    monkeypatch.setenv("SOURCE_GROUP_API_KEY", "sk-shared")
    monkeypatch.setenv("TRANSLATOR_BASE_URL", "http://own.local/v1")
    monkeypatch.setenv("TRANSLATOR_MODEL", "own-model")
    monkeypatch.setenv("TRANSLATOR_API_KEY", "sk-own")

    from polyglot_lc.config import get_agent_config

    assert get_agent_config("TRANSLATOR").model == "own-model"
    # Critic has no own override -> falls back to the shared group.
    assert get_agent_config("CRITIC").model == "shared-model"


def test_source_group_missing_reports_both_options(monkeypatch):
    for name in ("TRANSLATOR", "CRITIC", "REFINER"):
        for suffix in ("BASE_URL", "MODEL", "API_KEY"):
            monkeypatch.delenv(f"{name}_{suffix}", raising=False)
    for suffix in ("BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"SOURCE_GROUP_{suffix}", raising=False)
    monkeypatch.setenv("LLM_MOCK", "0")
    from polyglot_lc.config import ConfigError, get_agent_config

    with pytest.raises(ConfigError, match="SOURCE_GROUP_BASE_URL"):
        get_agent_config("TRANSLATOR")


# ------------------------- shared cacheable prefix --------------------------
def test_read_source_group_shares_identical_prefix(mock_env):
    from polyglot_lc.agents import CriticAgent, RefinerAgent, TranslatorAgent
    from polyglot_lc.config import get_agent_config

    translator = TranslatorAgent(get_agent_config("TRANSLATOR"))
    critic = CriticAgent(get_agent_config("CRITIC"))
    refiner = RefinerAgent(get_agent_config("REFINER"))

    translator_msgs = translator.build_prompt(
        "Source chunk text.", "en", "vi", "literal",
        pitfalls=["avoid literalism"], glossary_md="- NMT (x2)",
    )

    captured: dict[str, list] = {}

    def fake_json(messages, cache_prefix=False):
        captured["critic"] = messages
        return {"score": 9.0, "issues": [], "suggestions": []}

    def fake_chat(messages, **kw):
        captured["refiner"] = messages
        return LLMResponse(text="refined", model="m")

    critic.chat_json = fake_json  # type: ignore[assignment]
    refiner.chat = fake_chat  # type: ignore[assignment]

    critic.review(
        "Source chunk text.", "draft", "en", "vi",
        glossary_md="- NMT (x2)", strategy="literal", pitfalls=["avoid literalism"],
    )
    refiner.refine(
        "Source chunk text.", "draft", "Score: 3.0/10", "en", "vi",
        strategy="literal", pitfalls=["avoid literalism"], glossary_md="- NMT (x2)",
    )

    prefix = translator_msgs[0]["content"]
    assert prefix == captured["critic"][0]["content"]
    assert prefix == captured["refiner"][0]["content"]
    assert "Source chunk text." in prefix


def test_read_source_prefix_marks_cache_when_explicit(mock_env, monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE", "explicit")
    from polyglot_lc.agents import TranslatorAgent
    from polyglot_lc.config import get_agent_config
    from polyglot_lc.llm.client import _apply_cache_prefix

    translator = TranslatorAgent(get_agent_config("TRANSLATOR"))
    msgs = translator.build_prompt("Hello.", "en", "vi", "literal")
    marked = _apply_cache_prefix(msgs)
    block = marked[0]["content"]
    assert isinstance(block, list)
    assert block[0]["cache_control"] == {"type": "ephemeral"}


# ------------------------- summary tree request count -----------------------
class _CountingPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **kw):  # noqa: ANN001, ANN003
        self.calls += 1
        return SimpleNamespace(text="summary")


def test_summary_tree_skips_per_chunk_requests(mock_env, sample_doc):
    from polyglot_lc.segment.chunker import chunk_document
    from polyglot_lc.segment.summary_tree import build_sections, summarize_with_planner

    chunks = chunk_document(sample_doc, max_tokens=40, overlap_tokens=0)
    sections = build_sections(chunks)

    lean = _CountingPlanner()
    summarize_with_planner(chunks, lean, chunk_summaries=False)
    # one request per section + one for the document; none per chunk
    assert lean.calls == len(sections) + 1

    full = _CountingPlanner()
    summarize_with_planner(chunks, full, chunk_summaries=True)
    assert full.calls == len(chunks) + len(sections) + 1
    assert full.calls > lean.calls


# ------------------------- verifier scope -----------------------------------
def test_verifier_truncates_source_to_chunk_context(mock_env):
    from polyglot_lc.agents import VerifierAgent
    from polyglot_lc.config import get_agent_config

    verifier = VerifierAgent(get_agent_config("VERIFIER"))
    captured: dict[str, str] = {}

    def fake_chat(messages, **kw):
        captured["text"] = "\n".join(str(m["content"]) for m in messages)
        return LLMResponse(text='{"pass": true, "reasons": []}', model="m")

    verifier.chat = fake_chat  # type: ignore[assignment]
    long_source = "X" * 5000
    result = verifier.verify(long_source, "short translation", "en", "vi")
    assert result.ok is True
    # Only up to 2000 chars of the source chunk reach the verifier, never the
    # whole document.
    assert captured["text"].count("X") <= 2000


# ------------------------- planner runs once --------------------------------
def test_planner_runs_once_per_run(mock_env, sample_md_file, monkeypatch):
    from polyglot_lc.agents.planner import PlannerAgent

    calls = {"n": 0}
    original = PlannerAgent.plan

    def wrapped(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PlannerAgent, "plan", wrapped)
    from polyglot_lc.pipeline import TranslationPipeline

    pipeline = TranslationPipeline(quiet=True)
    pipeline.run(sample_md_file, src="en", tgt="vi")
    assert calls["n"] == 1


# ------------------------- CLI token report ---------------------------------
def test_cli_token_report_shows_cache_and_models():
    from polyglot_lc.cli import _token_report

    TokenUsage.reset()
    TokenUsage.add(1000, 200, agent="TRANSLATOR", cached_tokens=600,
                   model="m1", provider="http://a/v1")
    TokenUsage.add(500, 100, agent="VERIFIER",
                   model="m2", provider="http://b/v1")
    report = _token_report()
    assert "Cached input:   600" in report
    assert "Uncached input: 900" in report
    assert "Requests: 2" in report
    assert "By provider / model:" in report
    assert "http://a/v1 / m1" in report
    TokenUsage.reset()
