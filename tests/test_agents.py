"""Tests for agents (mock mode): prompts, parsing, hard checks, TM."""

from __future__ import annotations

import pytest

from polyglot_lc.agents import (
    CriticAgent,
    PlannerAgent,
    RefinerAgent,
    TranslatorAgent,
    VerifierAgent,
    hard_checks,
)
from polyglot_lc.agents.critic import Critique
from polyglot_lc.agents.verifier import VerifyResult
from polyglot_lc.memory.tm import TranslationMemory


@pytest.fixture()
def planner(mock_env):
    return PlannerAgent(get_agent_config := __import__(
        "polyglot_lc.config", fromlist=["get_agent_config"]).get_agent_config("PLANNER")
    )


@pytest.fixture()
def critic(mock_env):
    from polyglot_lc.config import get_agent_config

    return CriticAgent(get_agent_config("CRITIC"))


def test_planner_returns_plan(mock_env, planner):
    plan = planner.plan(doc_summary="A tech doc", src_lang="en", tgt_lang="vi",
                        domain="tech", glossary_md="- NMT (x3)")
    assert plan["strategy"]
    assert isinstance(plan["few_shot"], list)
    assert isinstance(plan["pitfalls"], list)


def test_critic_parses_score(mock_env, critic):
    c = critic.review("hello", "xin chào", "en", "vi")
    assert isinstance(c, Critique)
    assert 0 <= c.score <= 10
    assert "Score" in c.feedback_text


def test_verifier_hard_checks_placeholders(mock_env):
    problems = hard_checks("text with ⟦CODE_1⟧ inside", "text without codes")
    assert any("placeholder" in p for p in problems)


def test_verifier_hard_checks_headings(mock_env):
    problems = hard_checks("# Title\ntext", "## Different\ntext")
    assert problems  # heading level changed


def test_verifier_llm_pass(mock_env):
    from polyglot_lc.config import get_agent_config

    v = VerifierAgent(get_agent_config("VERIFIER"))
    result = v.verify("hello", "xin chào", "en", "vi")
    assert isinstance(result, VerifyResult)
    assert result.ok is True
    assert bool(result) is True


def test_translator_prompt_contains_glossary(mock_env):
    from polyglot_lc.config import get_agent_config

    t = TranslatorAgent(get_agent_config("TRANSLATOR"))
    prompt = t.build_prompt(
        chunk_text="NMT is great.", src_lang="en", tgt_lang="vi",
        strategy="literal", few_shot=["Hi -> Chào"], pitfalls=["drop terms"],
        glossary_md="- NMT (x2)",
    )
    joined = "\n".join(m["content"] for m in prompt)
    assert "NMT" in joined
    assert "Hi -> Chào" in joined
    assert "literal" in joined
    assert prompt[0]["role"] == "system"


def test_translator_stores_tm(mock_env):
    from polyglot_lc.config import get_agent_config

    tm = TranslationMemory()
    t = TranslatorAgent(get_agent_config("TRANSLATOR"), tm=tm)
    prompt = t.build_prompt("Unique source text.", "en", "vi", "literal")
    out = t.translate(prompt)
    assert out
    assert len(tm) >= 1


def test_refiner_returns_text(mock_env):
    from polyglot_lc.config import get_agent_config

    r = RefinerAgent(get_agent_config("REFINER"))
    out = r.refine("source", "draft", "score 3: bad", "en", "vi")
    assert isinstance(out, str) and out
