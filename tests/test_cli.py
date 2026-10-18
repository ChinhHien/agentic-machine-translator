"""Tests for what ``polyglot translate`` says when it is done.

The command's own words are the answer to "did this work?", so they are read
here from a real mock-mode run rather than from the report object.
"""

from __future__ import annotations

from typer.testing import CliRunner

from polyglot_lc.cli import app

runner = CliRunner()


def _doc(tmp_path):
    from pdfdoc import write_pdf

    return write_pdf(
        tmp_path / "doc.pdf",
        "Neural machine translation splits documents into segments.\n\n"
        "Each segment is translated with a consistent glossary.\n",
    )


def test_a_clean_run_says_completed(mock_env, tmp_path):
    out = tmp_path / "clean.md"
    result = runner.invoke(app, ["translate", "--file", str(_doc(tmp_path)), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Completed." in result.stdout
    assert out.is_file()


def test_a_provider_failure_the_run_waited_out_is_not_a_failed_translation(
    mock_env, tmp_path, monkeypatch
):
    """Waiting out an outage is not failing to translate.

    Every chunk is in the file when the command finishes, so the run is a
    completed one; what it had to wait for is listed under the totals.
    """
    from polyglot_lc.agents.translator import TranslatorAgent
    from polyglot_lc.llm.resilience import TransientError

    monkeypatch.setenv("CHUNK_RETRY_BASE_DELAY", "0.01")
    monkeypatch.setenv("CHUNK_RETRY_MAX_DELAY", "0.02")
    original = TranslatorAgent.translate
    calls = {"n": 0}

    def flaky_translate(self, prompt, chunk_id=""):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientError("connection reset by peer")
        return original(self, prompt, chunk_id=chunk_id)

    monkeypatch.setattr(TranslatorAgent, "translate", flaky_translate)
    out = tmp_path / "waited.md"
    result = runner.invoke(app, ["translate", "--file", str(_doc(tmp_path)), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert calls["n"] >= 2  # the chunk really was run again
    assert "Completed with errors." in result.stdout
    assert "Failed." not in result.stdout
    assert "chunk run again after errors" in result.stdout


def test_the_command_says_failed_when_a_chunk_has_no_translation(mock_env, tmp_path, monkeypatch):
    """``Failed`` is a verdict on the file, not on the road to it."""
    from polyglot_lc.pipeline import PipelineResult, TranslationPipeline

    def fake_run(self, *args, **kwargs):
        result = PipelineResult(chunks=3, chunks_failed=1, output_path=str(tmp_path / "out.md"))
        result.errors = ["chunk-001: RateLimitError: 429"]
        return result

    monkeypatch.setattr(TranslationPipeline, "run", fake_run)
    result = runner.invoke(
        app, ["translate", "--file", str(_doc(tmp_path)), "--out", str(tmp_path / "out.md")]
    )

    verdict = next(line for line in result.stdout.splitlines() if line.strip())
    assert verdict == "Failed."
    assert "chunk-001: RateLimitError: 429" in result.stdout
