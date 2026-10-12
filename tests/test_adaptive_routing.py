"""Tests for the adaptive quality pipeline: routing gates and the fast verify."""

from __future__ import annotations

from polyglot_lc.config import get_app_config
from polyglot_lc.llm.client import TokenUsage, clear_mock_replies
from polyglot_lc.orchestrator.graph import should_take_fast_path
from polyglot_lc.orchestrator.router import after_translate

DOC = (
    "Neural machine translation splits documents into segments.\n\n"
    "Each segment is translated with a consistent glossary.\n"
)

#: A state that satisfies every fast-path condition.
_GOOD = {"complexity": 0.05, "draft_score": 10.0, "draft_issues": []}


# --------------------------------- routing ----------------------------------
def test_after_translate_reads_the_nodes_decision():
    assert after_translate({"fast_path": True}) == "verify"
    assert after_translate({"fast_path": False}) == "critic"
    assert after_translate({}) == "critic"  # unknown -> full quality path


def test_fast_path_gate_requires_a_clean_confident_simple_chunk(mock_env):
    app = get_app_config()
    assert should_take_fast_path(_GOOD, app) is True

    assert should_take_fast_path({**_GOOD, "complexity": None}, app) is False
    assert should_take_fast_path({**_GOOD, "complexity": 0.9}, app) is False
    assert should_take_fast_path({**_GOOD, "draft_score": 8.9}, app) is False
    assert (
        should_take_fast_path({**_GOOD, "draft_issues": ["heading structure changed"]}, app)
        is False
    )


def test_fast_path_verdict_names_the_comparison_that_decided_it(mock_env):
    """The gate's answer was always computed; now it also says why, in numbers."""
    from polyglot_lc.orchestrator.graph import fast_path_verdict

    app = get_app_config()
    taken, reasons = fast_path_verdict({**_GOOD, "complexity": 0.62}, app)
    assert taken is False
    assert reasons[0] == "complexity 0.62 > 0.35"  # decisive reason first

    taken, reasons = fast_path_verdict({**_GOOD, "draft_score": 8.9}, app)
    assert taken is False
    assert reasons[0] == "draft score 8.90 < 9.00"

    taken, reasons = fast_path_verdict(_GOOD, app)
    assert taken is True
    assert reasons == ["complexity 0.05 <= 0.35", "draft clean"]


def test_a_verdict_for_a_chunk_with_no_complexity_says_so_rather_than_guessing(mock_env):
    from polyglot_lc.orchestrator.graph import fast_path_verdict

    taken, reasons = fast_path_verdict({**_GOOD, "complexity": None}, get_app_config())
    assert taken is False and reasons == ["complexity unknown for this chunk"]


def test_fast_path_gate_is_off_when_adaptive_quality_is_off(mock_env, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_QUALITY", "0")
    assert should_take_fast_path(_GOOD, get_app_config()) is False


def test_gate_thresholds_follow_configuration(mock_env, monkeypatch):
    monkeypatch.setenv("FAST_PATH_MAX_COMPLEXITY", "0.9")
    monkeypatch.setenv("FAST_PATH_MIN_SCORE", "7.0")
    app = get_app_config()
    assert (
        should_take_fast_path({"complexity": 0.85, "draft_score": 7.5, "draft_issues": []}, app)
        is True
    )


# ------------------------------ end to end ----------------------------------
def _run(path, out, **kw):
    from polyglot_lc.pipeline import TranslationPipeline

    return TranslationPipeline(quiet=True).run(path, src="en", tgt="vi", out=str(out), **kw)


def _write(tmp_path, text: str = DOC):
    """Input is PDF-only, so a document has to be rendered into a real one."""
    from pdfdoc import write_pdf

    return write_pdf(tmp_path / "doc.pdf", text)


def test_simple_document_takes_the_fast_path(mock_env, tmp_path):
    clear_mock_replies()
    path = _write(tmp_path)
    TokenUsage.reset()
    result = _run(path, tmp_path / "out.md")

    assert result.fast_path_chunks == result.chunks >= 1
    assert result.standard_path_chunks == 0
    assert result.deep_path_chunks == 0
    # No Critic request was made for any chunk.
    calls = TokenUsage.snapshot_by_agent()
    assert "CRITIC" not in calls
    assert result.avg_critic > 0  # estimated from the draft signals
    assert result.avg_critic_estimate is True


def test_structured_document_takes_the_full_path(mock_env, tmp_path):
    clear_mock_replies()
    # The mock translation cannot preserve headings, so the deterministic draft
    # signals report a structural problem and the fast path must not fire. The
    # signal is structural, so the body is plain prose.
    path = _write(
        tmp_path,
        "# Title\n\nA first paragraph of the document body.\n\n## Section\n\n"
        "Another paragraph of the document body.\n",
    )
    TokenUsage.reset()
    result = _run(path, tmp_path / "out.md")

    assert result.fast_path_chunks == 0
    assert TokenUsage.snapshot_by_agent().get("CRITIC", {}).get("calls", 0) > 0
    assert result.avg_critic_estimate is False


def test_realistic_draft_takes_the_fast_path_on_chunks_larger_than_the_guard(
    mock_env, tmp_path, monkeypatch
):
    """The fast path must work on normal-size chunks, not just tiny ones.

    The mock translator returns a fixed short string, which the truncation guard
    correctly distrusts once a chunk is longer than a few hundred characters.
    A realistic target-length draft must therefore route to the fast path.
    """
    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.translator import TranslatorAgent

    unit = "Đây là bản dịch tiếng Việt đầy đủ cho đoạn văn bản nguồn rất dài này. "

    def fake_translate(self, prompt, chunk_id=""):
        source = extract_source(prompt)
        target_len = max(1, int(len(source) * 0.9))  # plausible long target
        return (unit * (target_len // len(unit) + 1))[:target_len]

    monkeypatch.setattr(TranslatorAgent, "translate", fake_translate)
    clear_mock_replies()
    path = _write(
        tmp_path,
        "\n\n".join(
            f"Paragraph {i} describes step {i} of the workflow in plain prose without any "
            f"heading at all during the whole process."
            for i in range(10)
        ),
    )
    TokenUsage.reset()
    result = _run(path, tmp_path / "out.md", max_tokens=100)

    assert result.chunks > 2
    assert result.fast_path_chunks == result.chunks
    assert TokenUsage.snapshot_by_agent().get("CRITIC") is None


def test_fast_verify_skips_the_paid_verifier_when_enabled(mock_env, tmp_path, monkeypatch):
    monkeypatch.setenv("FAST_PATH_SKIP_LLM_VERIFY", "1")
    clear_mock_replies()
    path = _write(tmp_path)
    TokenUsage.reset()
    result = _run(path, tmp_path / "out.md")

    assert result.fast_path_chunks == result.chunks >= 1
    assert TokenUsage.snapshot_by_agent().get("VERIFIER") is None
    # Hard checks still guard the output, and the document is still written.
    assert (tmp_path / "out.md").read_text(encoding="utf-8").strip()


def test_verifier_still_runs_on_the_fast_path_by_default(mock_env, tmp_path):
    clear_mock_replies()
    path = _write(tmp_path)
    TokenUsage.reset()
    result = _run(path, tmp_path / "out.md")

    by = TokenUsage.snapshot_by_agent()
    assert by.get("CRITIC") is None
    assert by.get("VERIFIER", {}).get("calls", 0) == result.chunks
