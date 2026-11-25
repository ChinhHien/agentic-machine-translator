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
from polyglot_lc.agents.verifier import (
    VerifyResult,
    _filter_profile_defects,
    glossary_defects,
    heading_notes,
)
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


def test_verifier_forgives_the_source_pseudo_heading_pair(mock_env):
    """The extractor can split one sentence across two oversized lines; the
    translation re-joins them and must not burn the retry budget for it."""
    source = (
        "## Executive Summary\n\n"
        "## • Separating control selection from the controls, thereby allowing the\n\n"
        "## content to NIST SP 800-53B, Control Baselines;\n\n"
        "## Prologue\n"
    )
    translation = (
        "## Tóm tắt điều hành\n\n"
        "## • Tách biệt việc lựa chọn khỏi các biện pháp kiểm soát, cho phép nội dung mở rộng sang NIST SP 800-53B;\n\n"
        "## Phần mở đầu\n"
    )
    assert hard_checks(source, translation) == []
    assert heading_notes(source, translation)


def test_verifier_still_fails_a_dropped_real_heading(mock_env):
    problems = hard_checks(
        "## Joint Task Force\n\n## Patent Disclosure Notice\n\n## Table of Contents\n",
        "## Nhóm công tác liên hợp\n\n## Mục lục\n",
    )
    assert problems == ["heading structure changed"]
    assert heading_notes(
        "## Joint Task Force\n\n## Patent Disclosure Notice\n\n## Table of Contents\n",
        "## Nhóm công tác liên hợp\n\n## Mục lục\n",
    ) == []


def test_glossary_check_is_deterministic_and_names_the_exact_pair(mock_env):
    defects = glossary_defects(
        "Disinfect with iodine and chlorine.",
        "Khử khuẩn bằng i-ốt và clo.",
        {"iodine": "iốt", "chlorine": "clo"},
    )
    assert len(defects) == 1
    assert defects[0]["term"] == "iodine"
    assert defects[0]["expected"] == "iốt"
    assert defects[0]["severity"] == "major"


def test_a_multi_form_glossary_pair_accepts_either_rendering(mock_env):
    """A real run rejected 28 chunks for a pair the translator could not satisfy.

    The planner wrote ``security -> "bảo mật / an toàn thông tin"``; the check
    required that literal slash-joined string, so a translation using either
    correct Vietnamese form was reported as a missing term.
    """
    defects = glossary_defects(
        "Security controls protect the system.",
        "Các biện pháp bảo mật bảo vệ hệ thống.",
        {"security": "bảo mật / an toàn thông tin"},
    )
    assert defects == []


def test_a_multi_form_glossary_pair_still_fails_when_no_form_appears(mock_env):
    defects = glossary_defects(
        "Security controls protect the system.",
        "Các biện pháp an ninh bảo vệ hệ thống.",
        {"security": "bảo mật / an toàn thông tin"},
    )
    assert len(defects) == 1
    assert defects[0]["term"] == "security"
    assert defects[0]["expected"] == "bảo mật / an toàn thông tin"


def test_reference_profile_drops_language_and_terminology_notes(mock_env):
    """A bibliography may keep English article titles and chemical names."""
    kept = _filter_profile_defects(
        [
            {"type": "language", "severity": "major", "message": "English title"},
            {"type": "terminology", "severity": "major", "message": "chemical name"},
            {"type": "completeness", "severity": "major", "message": "missing entry"},
        ],
        "references",
    )
    assert [d["type"] for d in kept] == ["completeness"]


def test_major_term_defect_is_retryable_but_source_defect_is_not(mock_env):
    major = VerifyResult(
        False,
        ["term missing"],
        defects=[{"severity": "major", "type": "terminology", "term": "a", "expected": "b"}],
    )
    source = VerifyResult(
        False,
        ["heading structure changed"],
        defects=[{"severity": "fatal", "type": "structure", "message": "heading"}],
        source_defect=True,
    )
    assert major.severity == "major" and major.retryable is True
    assert source.severity == "fatal" and source.retryable is False


def test_a_verifier_that_cannot_run_is_not_a_pass(mock_env):
    from polyglot_lc.config import get_agent_config

    verifier = VerifierAgent(get_agent_config("VERIFIER"))

    def unavailable(*args, **kwargs):
        raise RuntimeError("provider down")

    verifier.chat_json = unavailable
    with pytest.raises(RuntimeError, match="verification unavailable"):
        verifier.verify("NMT is fast.", "Dịch máy neural rất nhanh.", "en", "vi")


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"pass": "false"},
        {"pass": "true"},
        {"pass": 1},
        {"pass": None},
        {"pass": True, "reasons": "unchecked"},
        {"pass": True, "reasons": [1]},
        [],
    ],
)
def test_verifier_rejects_invalid_verdicts(mock_env, monkeypatch, reply):
    verifier = VerifierAgent()
    monkeypatch.setattr(verifier, "chat_json", lambda *args, **kwargs: reply)
    with pytest.raises(RuntimeError, match="invalid verifier verdict"):
        verifier.verify("hello", "xin chào", "en", "vi")


def test_the_verifier_re_asks_a_cut_verdict_with_a_raised_cap(mock_env, monkeypatch):
    """A verdict cut at the provider's cap must not abort the run.

    The gateway defaults to 2048 completion tokens; a verbose verdict hit that
    limit, the verifier was declared unavailable and a 609-chunk run stopped
    with nothing to show.
    """
    from polyglot_lc.llm.resilience import RAISED_OUTPUT_CAP, TruncatedError

    verifier = VerifierAgent()
    caps: list[int | None] = []

    def chat_json(messages, **kw):
        caps.append(kw.get("max_tokens"))
        if len(caps) == 1:
            raise TruncatedError("cut at 2048", completion_tokens=2048)
        return {"pass": True, "reasons": []}

    monkeypatch.setattr(verifier, "chat_json", chat_json)
    result = verifier.verify("NMT is fast.", "Dịch máy neural rất nhanh.", "en", "vi")

    assert result.ok
    assert len(caps) == 2
    # The cap is asked for up front; the retry then doubles it instead of
    # repeating a budget the reply has just overrun.
    assert caps == [RAISED_OUTPUT_CAP, RAISED_OUTPUT_CAP * 2]


def test_a_cap_above_the_default_still_grows_to_the_ceiling(mock_env, monkeypatch):
    """A verifier already tuned to the ceiling does not shrink on the retry.

    ``VERIFIER_MAX_TOKENS`` at the ceiling is the largest budget this project
    asks for; the raised cap stays there rather than doubling past what a
    gateway accepts.
    """
    from polyglot_lc.agents.verifier import VERIFIER_MAX_TOKENS_CEILING
    from polyglot_lc.llm.resilience import TruncatedError

    monkeypatch.setenv("VERIFIER_MAX_TOKENS", str(VERIFIER_MAX_TOKENS_CEILING))
    verifier = VerifierAgent()
    caps: list[int | None] = []

    def chat_json(messages, **kw):
        caps.append(kw.get("max_tokens"))
        if len(caps) == 1:
            raise TruncatedError(
                "cut at the ceiling", completion_tokens=VERIFIER_MAX_TOKENS_CEILING
            )
        return {"pass": True, "reasons": []}

    monkeypatch.setattr(verifier, "chat_json", chat_json)
    result = verifier.verify("NMT is fast.", "Dịch máy neural rất nhanh.", "en", "vi")

    assert result.ok
    assert caps == [VERIFIER_MAX_TOKENS_CEILING, VERIFIER_MAX_TOKENS_CEILING]


def test_the_verifier_ceiling_fits_thinking_plus_a_full_verdict(mock_env):
    """The ceiling was 8192, and a real run cut three verdicts at exactly that.

    Each one then fell back to asking for a *minimal* verdict, which checks less
    than the full one — a quiet quality loss caused by a budget, not by the
    translation. The same endpoint served translator replies of ~17000
    completion tokens with no cap set, so it demonstrably allows more than 8192.
    """
    from polyglot_lc.agents.verifier import VERIFIER_MAX_TOKENS_CEILING

    assert VERIFIER_MAX_TOKENS_CEILING >= 32768


@pytest.mark.parametrize("cap, expected", [(4096, 8192), (16384, 32768), (32768, 32768)])
def test_the_recovery_retry_never_asks_above_the_ceiling(mock_env, monkeypatch, cap, expected):
    """Doubling is bounded: a gateway rejects a ``max_tokens`` over its limit."""
    from polyglot_lc.agents.verifier import VERIFIER_MAX_TOKENS_CEILING
    from polyglot_lc.llm.resilience import TruncatedError

    assert expected <= VERIFIER_MAX_TOKENS_CEILING
    monkeypatch.setenv("VERIFIER_MAX_TOKENS", str(cap))
    verifier = VerifierAgent()
    caps: list[int | None] = []

    def chat_json(messages, **kw):
        caps.append(kw.get("max_tokens"))
        if len(caps) == 1:
            raise TruncatedError(f"cut at {cap}", completion_tokens=cap)
        return {"pass": True, "reasons": []}

    monkeypatch.setattr(verifier, "chat_json", chat_json)
    verifier.verify("NMT is fast.", "Dịch máy neural rất nhanh.", "en", "vi")

    # Clamped to the ceiling, and never lowered below the cap that was used.
    assert caps == [cap, expected]


def test_a_verifier_that_keeps_cutting_is_still_unavailable(mock_env, monkeypatch):
    """A raised cap, then a minimal verdict, then the fatal policy applies."""
    from polyglot_lc.llm.resilience import TruncatedError

    verifier = VerifierAgent()
    calls: list[str] = []

    def chat_json(messages, **kw):
        calls.append("asked")
        raise TruncatedError("cut at 2048", completion_tokens=2048)

    monkeypatch.setattr(verifier, "chat_json", chat_json)
    with pytest.raises(RuntimeError, match="verification unavailable"):
        verifier.verify("NMT is fast.", "Dịch máy neural rất nhanh.", "en", "vi")
    assert len(calls) == 3


def test_a_thinking_heavy_model_gets_a_minimal_verdict_ask(mock_env, monkeypatch):
    """Both full verdicts cut at the ceiling: ask for pass/reasons only.

    This is the real failure: a reasoning model spent 7359 of 8192 tokens
    thinking, leaving 833 for a full defect list, and the run died as
    "verification unavailable". The same thinking leaves ample room for the
    two-field answer.
    """
    from polyglot_lc.llm.resilience import TruncatedError

    verifier = VerifierAgent()
    prompts: list[str] = []
    caps: list[int | None] = []

    def chat_json(messages, **kw):
        prompts.append(messages[-1]["content"])
        caps.append(kw.get("max_tokens"))
        if len(prompts) < 3:
            raise TruncatedError(
                "cut with the answer unfinished",
                completion_tokens=8192,
                reasoning_tokens=7359,
                answer_tokens=833,
            )
        assert '"pass": true|false' in prompts[-1]
        return {"pass": True, "reasons": []}

    monkeypatch.setattr(verifier, "chat_json", chat_json)
    result = verifier.verify("NMT is fast.", "Dịch máy neural rất nhanh.", "en", "vi")

    assert result.ok
    assert len(prompts) == 3
    assert "Answer immediately" in prompts[-1]
    assert '"defects"' in prompts[0]  # the ordinary ask still names defects
    assert caps[-1] == caps[-2]  # minimal answer, same ceiling


def test_a_minimal_verdict_cut_after_its_json_is_still_used(mock_env, monkeypatch):
    """The last ask is a normal call: a whole answer beside the cut is used."""
    from polyglot_lc.llm.resilience import TruncatedError

    verifier = VerifierAgent()
    calls: list[int] = []

    def chat_json(messages, **kw):
        calls.append(1)
        if len(calls) < 3:
            raise TruncatedError(
                "cut with the answer unfinished",
                completion_tokens=8192,
                reasoning_tokens=8100,
                answer_tokens=92,
            )
        raise TruncatedError(
            "cut after the answer finished",
            completion_tokens=8192,
            partial_text='{"pass": true, "reasons": []}',
            reasoning_tokens=8100,
            answer_tokens=20,
        )

    monkeypatch.setattr(verifier, "chat_json", chat_json)
    assert verifier.verify("NMT is fast.", "Dịch máy neural rất nhanh.", "en", "vi").ok
    assert len(calls) == 3


def test_a_complete_verdict_survives_a_reply_flagged_as_cut(mock_env, monkeypatch):
    """The answer can finish before the cap does: use it instead of retrying.

    A reasoning model keeps thinking after it has written the JSON, so
    ``finish_reason`` says ``length`` while the content is whole. Re-asking for
    a verdict the reply already holds would only spend the budget twice.
    """
    from polyglot_lc.llm.resilience import TruncatedError

    verifier = VerifierAgent()
    calls: list[int | None] = []

    def chat_json(messages, **kw):
        calls.append(kw.get("max_tokens"))
        raise TruncatedError(
            "cut after the answer finished",
            completion_tokens=4096,
            partial_text='{"pass": true, "reasons": []}',
            reasoning_tokens=4090,
            answer_tokens=6,
        )

    monkeypatch.setattr(verifier, "chat_json", chat_json)
    result = verifier.verify("NMT is fast.", "Dịch máy neural rất nhanh.", "en", "vi")

    assert result.ok
    assert len(calls) == 1


def test_verifier_preserves_a_negative_verdict(mock_env, monkeypatch):
    verifier = VerifierAgent()
    monkeypatch.setattr(
        verifier, "chat_json", lambda *args, **kwargs: {"pass": False, "reasons": ["wrong meaning"]}
    )
    result = verifier.verify("slow", "nhanh", "en", "vi")
    assert not result
    assert result.reasons == ["wrong meaning"]


def test_verifier_downgrades_a_self_contradicting_verdict(mock_env, monkeypatch):
    """A fail whose reasons claim every check passes and that names no defect
    gives a re-translation nothing to fix: it is a pass, not a rejection."""
    verifier = VerifierAgent()
    monkeypatch.setattr(
        verifier,
        "chat_json",
        lambda *args, **kwargs: {
            "pass": False,
            "defects": [],
            "reasons": [
                (
                    "All applicable checks pass: the translation is complete, adds"
                    " no unsupported content, reads naturally in Vietnamese, and"
                    " preserves headings and list structure."
                )
            ],
        },
    )
    result = verifier.verify("hello", "xin chào", "en", "vi")
    assert result


def test_verifier_keeps_a_fail_that_names_a_defect(mock_env, monkeypatch):
    """The contradiction downgrade never rescues a verdict that points at
    something a re-translation must fix."""
    verifier = VerifierAgent()
    monkeypatch.setattr(
        verifier,
        "chat_json",
        lambda *args, **kwargs: {
            "pass": False,
            "defects": [
                {"severity": "major", "type": "language", "message": "checks pass but tone drifts"}
            ],
            "reasons": ["All applicable checks pass."],
        },
    )
    result = verifier.verify("slow", "nhanh", "en", "vi")
    assert not result


def test_verifier_pass_with_praise_reasons_is_not_rejected(mock_env, monkeypatch):
    """A pass that names no defect must not have its praise turned into a
    defect: the reasons fallback is for fails, not for passes."""
    verifier = VerifierAgent()
    monkeypatch.setattr(
        verifier,
        "chat_json",
        lambda *args, **kwargs: {
            "pass": True,
            "defects": [],
            "reasons": [
                (
                    "All checks passed: complete, no additions, natural Vietnamese, "
                    "structure and placeholders intact, terminology consistent."
                )
            ],
        },
    )
    result = verifier.verify("hello", "xin chào", "en", "vi")
    assert result
    assert result.defects == []


def test_verifier_pass_is_still_overridden_by_a_named_defect(mock_env, monkeypatch):
    verifier = VerifierAgent()
    monkeypatch.setattr(
        verifier,
        "chat_json",
        lambda *args, **kwargs: {
            "pass": True,
            "defects": [
                {"severity": "fatal", "type": "completeness", "message": "tail missing"}
            ],
            "reasons": ["looks fine"],
        },
    )
    result = verifier.verify("hello world", "xin chào", "en", "vi")
    assert not result


def test_verifier_pass_still_checks_the_glossary(mock_env, monkeypatch):
    verifier = VerifierAgent()
    monkeypatch.setattr(
        verifier,
        "chat_json",
        lambda *args, **kwargs: {"pass": True, "defects": [], "reasons": ["complete"]},
    )
    result = verifier.verify(
        "Disinfection is key.",
        "Khử khuẩn là chính.",
        "en",
        "vi",
        glossary_pairs={"Disinfection": "khử trùng"},
    )
    assert not result
    assert any(d["type"] == "terminology" for d in result.defects)


def test_verifier_fail_without_defects_still_uses_reasons(mock_env, monkeypatch):
    verifier = VerifierAgent()
    monkeypatch.setattr(
        verifier,
        "chat_json",
        lambda *args, **kwargs: {"pass": False, "defects": [], "reasons": ["wrong meaning"]},
    )
    result = verifier.verify("slow", "nhanh", "en", "vi")
    assert not result
    assert any(d["severity"] == "major" for d in result.defects)


@pytest.mark.parametrize(
    "source, translation, term, expected",
    [
        ("The values below are listed.", "Các giá trị dưới đây được liệt kê.", "low", "thấp"),
        ("Allow the flow to continue.", "Cho phép luồng tiếp tục.", "low", "thấp"),
        ("Please update the candidate list.", "Vui lòng cập nhật danh sách.", "date", "ngày"),
        ("A higher value follows.", "Một giá trị cao hơn đi sau.", "high", "cao"),
    ],
)
def test_a_term_inside_a_longer_word_is_not_a_glossary_defect(
    mock_env, source, translation, term, expected
):
    """A real run rejected 27 chunks demanding 'thấp' for the term 'low'.

    The source only ever said 'below'/'allow'/'flow'. No retranslation can
    satisfy a term that is not there, so the chunk burned its retry budget and
    was logged as a verification failure for nothing.
    """
    assert glossary_defects(source, translation, {term: expected}) == []


@pytest.mark.parametrize(
    "source",
    ["The impact level is LOW.", "A low-impact system.", "low", "Use low memory."],
)
def test_a_term_present_as_a_whole_word_is_still_checked(mock_env, source):
    defects = glossary_defects(source, "Mức tác động là CAO.", {"low": "thấp"})
    assert len(defects) == 1
    assert defects[0]["term"] == "low" and defects[0]["expected"] == "thấp"


def test_a_multiword_term_matches_across_line_wrapping(mock_env):
    defects = glossary_defects(
        "The privacy\ncontrol baseline applies.", "Áp dụng mức cơ sở.", {"privacy control": "kiểm soát quyền riêng tư"}
    )
    assert len(defects) == 1


def test_a_glossary_override_names_the_term_first_not_the_praise(mock_env, monkeypatch):
    """The log used to read 'verification failed: all source segments are present'.

    ``reasons`` carried the LLM's praise first and the actual defect last, and
    the graph quotes only the first three — so a rejected chunk's error message
    described a translation that was fine.
    """
    verifier = VerifierAgent()
    monkeypatch.setattr(
        verifier,
        "chat_json",
        lambda *a, **k: {
            "pass": True,
            "defects": [],
            "reasons": [
                "Completeness: all source sentences are represented.",
                "No additions: no invented content detected.",
                "Language: reads naturally.",
            ],
        },
    )
    result = verifier.verify(
        "Disinfection is key.",
        "Khử khuẩn là chính.",
        "en",
        "vi",
        glossary_pairs={"Disinfection": "khử trùng"},
    )
    assert not result
    assert "khử trùng" in result.reasons[0]


def test_a_glossary_override_of_an_llm_pass_is_logged(mock_env, monkeypatch):
    from polyglot_lc.agents import verifier as verifier_module

    seen: list[tuple] = []
    monkeypatch.setattr(verifier_module.logger, "warning", lambda *a, **k: seen.append(a))
    verifier = VerifierAgent()
    monkeypatch.setattr(
        verifier,
        "chat_json",
        lambda *a, **k: {"pass": True, "defects": [], "reasons": ["complete"]},
    )
    verifier.verify(
        "Disinfection is key.",
        "Khử khuẩn là chính.",
        "en",
        "vi",
        glossary_pairs={"Disinfection": "khử trùng"},
    )
    assert any("glossary override" in str(entry[0]) for entry in seen), seen


def test_a_fail_whose_defects_are_all_exempt_for_the_genre_is_not_a_rejection(
    mock_env, monkeypatch
):
    """A reference list may keep English titles; that is not a shippable defect."""
    verifier = VerifierAgent()
    monkeypatch.setattr(
        verifier,
        "chat_json",
        lambda *a, **k: {
            "pass": False,
            "defects": [
                {"severity": "major", "type": "language", "message": "English title left as is"}
            ],
            "reasons": ["Article titles stay in English."],
        },
    )
    result = verifier.verify(
        "Smith, J. (2020). A Study of Things.",
        "Smith, J. (2020). A Study of Things.",
        "en",
        "vi",
        profile="references",
    )
    assert result.ok
    assert result.defects == []


@pytest.mark.parametrize("role", ["critic", "refiner"])
def test_review_and_refinement_receive_the_complete_draft(mock_env, monkeypatch, role):
    draft = "Một đoạn bản dịch dài. " * 250 + " END_OF_DRAFT"
    feedback = "Fix terminology. " * 150 + " END_OF_FEEDBACK"
    captured = []

    def record(messages, **kwargs):
        captured.extend(message["content"] for message in messages)
        return LLMResponse(text=draft, model="mock-model")

    if role == "refiner":
        agent = RefinerAgent()
        monkeypatch.setattr(agent, "chat", record)
        agent.refine("source", draft, feedback, "en", "vi")
        assert feedback in "\n".join(captured)
    else:
        agent = CriticAgent()

        def review(messages, **kwargs):
            record(messages, **kwargs)
            return {"score": 8, "issues": [], "suggestions": []}

        monkeypatch.setattr(agent, "chat_json", review)
        agent.review("source", draft, "en", "vi")
    assert draft in "\n".join(captured)


def test_patch_verification_compares_source_original_and_candidate(mock_env):
    verifier = VerifierAgent()
    original = "NMT chậm. " * 400
    candidate = original.replace("NMT", "Dịch máy neural")
    source = "NMT is slow. " * 400
    prompt = _capture_verifier_prompt(
        verifier,
        source,
        candidate,
        previous_translation=original,
        glossary_md="- NMT -> Dịch máy neural",
    )
    assert source in prompt
    assert original in prompt
    assert candidate in prompt
    assert "Before terminology patch" in prompt
    assert "negation" in prompt
    assert "numbers" in prompt


def test_the_consistency_patch_sees_the_whole_translation(mock_env):
    """The editor must be shown the whole text it is asked to correct.

    The prompt used to slice the translation at 2500 characters, and the
    model answered with the fragment it was shown — a fragment that still
    carried the wanted term, so the checker accepted it in place of the
    complete translation.
    """
    from types import SimpleNamespace

    from polyglot_lc.agents.consistency import ConsistencyAgent
    from polyglot_lc.config import get_agent_config

    seen: dict[str, str] = {}

    def record(messages, **kwargs):
        seen["prompt"] = "\n".join(m["content"] for m in messages)
        return SimpleNamespace(text="bản dịch đã chỉnh thuật ngữ")

    agent = ConsistencyAgent(get_agent_config("REFINER"))
    agent.chat = record  # type: ignore[method-assign]
    translation = "Một câu đủ dài của bản dịch. " * 200  # well over 2500 characters
    out = agent.patch(
        source="A source paragraph.",
        translation=translation,
        fixes=[("NMT", "Dịch máy neural")],
        src_lang="en",
        tgt_lang="vi",
    )
    assert out == "bản dịch đã chỉnh thuật ngữ"
    assert translation in seen["prompt"]  # the WHOLE text, not a 2500-char prefix


def test_verifier_llm_pass(mock_env):
    from polyglot_lc.config import get_agent_config

    v = VerifierAgent(get_agent_config("VERIFIER"))
    result = v.verify("hello", "xin chào", "en", "vi")
    assert isinstance(result, VerifyResult)
    assert result.ok is True
    assert bool(result) is True


def test_mock_translation_preserves_source_headings(mock_env):
    from polyglot_lc.llm.client import clear_mock_replies

    clear_mock_replies()
    source = "# Title\n\nA paragraph.\n\n## Section\n\nAnother paragraph."
    translator = TranslatorAgent()
    translated = translator.translate(translator.build_prompt(source, "en", "vi", "faithful"))
    assert hard_checks(source, translated) == []


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


def _capture_verifier_prompt(
    verifier, source: str = "hello", translation: str = "xin chào", **kwargs
) -> str:
    captured: dict[str, str] = {}

    def fake_chat(messages, **kw):
        captured["prompt"] = messages[-1]["content"]
        return LLMResponse(text='{"pass": true, "reasons": []}', model="m")

    verifier.chat = fake_chat  # type: ignore[assignment]
    verifier.verify(source, translation, "en", "vi", **kwargs)
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


def test_verifier_prompt_carries_the_glossary(mock_env):
    """A term the glossary fixes is a defect only if the verifier can see it."""
    from polyglot_lc.config import get_agent_config

    v = VerifierAgent(get_agent_config("VERIFIER"))
    prompt = _capture_verifier_prompt(v, glossary_md="- NMT -> dịch máy nơ-ron")
    assert "Terminology" in prompt
    assert "dịch máy nơ-ron" in prompt


def test_verifier_prompt_skips_an_empty_glossary(mock_env):
    """``(none)`` is the planner's "no glossary", not a term to match."""
    from polyglot_lc.config import get_agent_config

    v = VerifierAgent(get_agent_config("VERIFIER"))
    assert "Glossary" not in _capture_verifier_prompt(v, glossary_md="(none)")
    assert "Glossary" not in _capture_verifier_prompt(v)


def test_verifier_sees_the_whole_chunk(mock_env):
    """A truncation would leave the tail of a long chunk unverified."""
    from polyglot_lc.config import get_agent_config

    tail = "The final clause is only here."
    source = "".join(f"Sentence number {i} of the chunk. " for i in range(120)) + tail
    assert len(source) > 2000
    v = VerifierAgent(get_agent_config("VERIFIER"))
    prompt = _capture_verifier_prompt(v, source=source, translation="xin chào")
    assert tail in prompt
    assert source in prompt


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
        glossary_candidates="(none)",
        core_glossary_md="(none)",
        trace_lines=[],
    )
    assert "glossary_terms" in _mock_json_reply(planner)

    critic = CriticAgent(get_agent_config("CRITIC"))
    assert "score" in _mock_json_reply(_capture_critic_prompt(critic))

    verifier = VerifierAgent(get_agent_config("VERIFIER"))
    assert "pass" in _mock_json_reply(_capture_verifier_prompt(verifier))


def test_the_translator_prompt_describes_the_emphasis_tokens(mock_env):
    """The tokens only survive if the model is told they are structure."""
    from polyglot_lc.config import get_agent_config

    t = TranslatorAgent(get_agent_config("TRANSLATOR"))
    prompt = t.build_prompt("A ⟦BOLD⟧B⟦/BOLD⟧ C.", "en", "vi", "literal")
    joined = "\n".join(m["content"] for m in prompt)
    assert "⟦BOLD⟧" in joined and "⟦/BOLD⟧" in joined
    assert "never translate or rename the tokens" in joined


def test_the_verifier_prompt_names_the_emphasis_tokens_as_structure(mock_env):
    """Otherwise "No additions" could report the tokens as invented content."""
    from polyglot_lc.config import get_agent_config

    v = VerifierAgent(get_agent_config("VERIFIER"))
    prompt = _capture_verifier_prompt(v)
    assert "⟦BOLD⟧" in prompt
