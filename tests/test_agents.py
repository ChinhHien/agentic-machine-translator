"""Tests for agents (mock mode): prompts, parsing, hard checks, TM."""

from __future__ import annotations

import pytest

from polyglot_lc.agents import (
    CriticAgent,
    RefinerAgent,
    TranslatorAgent,
    VerifierAgent,
    hard_checks,
)
from polyglot_lc.agents.critic import Critique
from polyglot_lc.agents.verifier import VerifyResult
from polyglot_lc.llm.client import LLMResponse, _mock_json_reply
from polyglot_lc.memory.tm import TranslationMemory


@pytest.fixture()
def critic(mock_env):
    from polyglot_lc.config import get_agent_config

    return CriticAgent(get_agent_config("CRITIC"))


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
        chunk_text="NMT is great.",
        src_lang="en",
        tgt_lang="vi",
        strategy="literal",
        few_shot=["Hi -> Chào"],
        pitfalls=["drop terms"],
        glossary_md="- NMT (x2)",
    )
    joined = "\n".join(m["content"] for m in prompt)
    assert "NMT" in joined
    assert "Hi -> Chào" in joined
    assert "literal" in joined
    assert prompt[0]["role"] == "system"


def test_translator_prompt_preserves_paragraph_structure(mock_env):
    from polyglot_lc.config import get_agent_config

    t = TranslatorAgent(get_agent_config("TRANSLATOR"))
    prompt = t.build_prompt("Para one.\n\nPara two.", "en", "vi", "literal")
    joined = "\n".join(m["content"] for m in prompt)
    assert "Preserve the document structure" in joined
    assert "never merge two paragraphs into one" in joined


def test_translator_prompt_keeps_markdown_tables(mock_env):
    """A flattened table row is how a license matrix loses its columns."""
    from polyglot_lc.config import get_agent_config

    t = TranslatorAgent(get_agent_config("TRANSLATOR"))
    table = "| Feature | LS | ST |\n| --- | --- | --- |\n| Servers | 5 | Not limited |"
    prompt = t.build_prompt(table, "en", "vi", "literal")
    joined = "\n".join(m["content"] for m in prompt)
    assert "Keep Markdown tables as tables" in joined
    assert "never rewrite a table as prose" in joined


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


# --------------------------- prompt contract --------------------------------
def _capture_critic_prompt(critic) -> str:
    """Run one review and return the dynamic (non-prefix) prompt it sent."""
    captured: dict[str, str] = {}

    def fake_json(messages, cache_prefix=False):
        captured["prompt"] = messages[-1]["content"]
        return {"score": 8.5, "issues": [], "suggestions": []}

    critic.chat_json = fake_json  # type: ignore[assignment]
    critic.review("hello", "xin chào", "en", "vi")
    return captured["prompt"]


def _capture_verifier_prompt(verifier) -> str:
    captured: dict[str, str] = {}

    def fake_chat(messages, **kw):
        captured["prompt"] = messages[-1]["content"]
        return LLMResponse(text='{"pass": true, "reasons": []}', model="m")

    verifier.chat = fake_chat  # type: ignore[assignment]
    verifier.verify("hello", "xin chào", "en", "vi")
    return captured["prompt"]


def test_critic_prompt_carries_a_rubric(mock_env, critic):
    """CRITIC_THRESHOLD=7.0 only means something if 7 is defined for the model."""
    prompt = _capture_critic_prompt(critic)
    assert "7-8: good" in prompt
    assert "Below 7, every issue must name the exact text" in prompt


def test_verifier_prompt_lists_the_checks_it_runs(mock_env):
    from polyglot_lc.config import get_agent_config

    v = VerifierAgent(get_agent_config("VERIFIER"))
    prompt = _capture_verifier_prompt(v)
    assert "Completeness" in prompt
    assert "No additions" in prompt
    # The reasons feed the single bounded retranslate, so they must be usable.
    assert "costs one re-translation" in prompt


def test_retranslate_prompt_names_the_rejection(mock_env):
    """The retranslate hint is a verifier verdict, not a critic's critique."""
    from polyglot_lc.config import get_agent_config

    t = TranslatorAgent(get_agent_config("TRANSLATOR"))
    prompt = t.build_prompt(
        "source text",
        "en",
        "vi",
        "literal",
        extra_critique="completeness: the final clause is missing",
    )
    joined = "\n".join(m["content"] for m in prompt)
    assert "rejected for these reasons" in joined
    assert "the final clause is missing" in joined


def test_mock_dispatcher_still_recognises_the_live_prompts(mock_env):
    """The inline mock keys on prompt substrings.

    A prompt edit that stops matching would silently downgrade every mock run
    to the generic ``{"note": "mock reply"}``, so this pins the coupling.
    """
    from polyglot_lc.analysis.unified_planner import _planner_prompt
    from polyglot_lc.config import get_agent_config

    planner = _planner_prompt(
        doc_title="Doc",
        doc_text="Body.",
        doc_summary="",
        outline={"Intro": "preview"},
        src_lang="en",
        tgt_lang="vi",
        domain="tech",
        glossary_candidates="(none)",
        user_glossary_md="(none)",
        trace_lines=[],
    )
    assert "glossary_terms" in _mock_json_reply(planner)

    critic = CriticAgent(get_agent_config("CRITIC"))
    assert "score" in _mock_json_reply(_capture_critic_prompt(critic))

    verifier = VerifierAgent(get_agent_config("VERIFIER"))
    assert "pass" in _mock_json_reply(_capture_verifier_prompt(verifier))
