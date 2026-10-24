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


def test_the_cli_reconfigures_a_cp1252_console_to_utf8(monkeypatch):
    """The final report carries Vietnamese; a cp1252 console must not crash it.

    Rich's legacy Windows renderer is picked from the environment at
    construction and writes through the stream's own codec, so the stream has
    to speak UTF-8 before anything is printed — otherwise the run dies after
    its files are written and the exit code lies.
    """
    import io
    import sys

    from polyglot_lc import cli

    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    cli.main(version=None)
    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"
    assert sys.stderr.encoding.lower().replace("-", "") == "utf8"


def test_a_clean_run_says_completed(mock_env, tmp_path):
    out = tmp_path / "clean.md"
    result = runner.invoke(app, ["translate", "--file", str(_doc(tmp_path)), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Completed." in result.stdout
    assert out.is_file()


def test_the_cli_says_how_much_drift_the_pass_left(mock_env, tmp_path, monkeypatch):
    """Báo cáo cuối phải nói phần *chưa* sửa, không chỉ số bản vá đã làm."""
    from polyglot_lc.pipeline import coordinator
    from polyglot_lc.pipeline.consistency import ConsistencyReport

    class _StubChecker:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, chunks, translations, pairs, *, context=None):
            report = ConsistencyReport(
                checked_terms=len(pairs),
                labels_checked=9,
                drifted_labels=3,
                inconsistencies=3,
                patch_budget=3,
                patched_chunks=2,
                skipped_chunks=4,
                stopped_early=True,
            )
            report.unresolved.append("chunk-001")
            return translations, report

    monkeypatch.setattr(coordinator, "ConsistencyChecker", _StubChecker)
    glossary = tmp_path / "glossary.csv"
    glossary.write_text("NMT,Dịch máy neural\n", encoding="utf-8")
    out = tmp_path / "drift.md"

    result = runner.invoke(
        app,
        [
            "translate",
            "--file",
            str(_doc(tmp_path)),
            "--out",
            str(out),
            "--glossary",
            str(glossary),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "consistency left 4 chunk(s) drifting" in result.output
    assert "patch rejected: 1" in result.output
    assert "3 label(s) drifted out of 9 compared" in result.output
    # The pass's cost ceiling is stated, not left to be inferred from the log.
    assert "consistency patch budget: 3 request(s), 2 used" in result.output
    assert "stopped after two patches in a row were refused" in result.output


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


def test_the_cli_says_when_there_was_no_glossary_term_to_compare(mock_env, tmp_path, monkeypatch):
    """Không lệch trên không thuật ngữ phải đọc ra là "chưa so thuật ngữ nào".

    The document has heuristic terminology (NMT, found twice) but the Planner
    returned no term *pair*, so there was nothing to enforce across chunks —
    which is not the same result as checking 175 terms and finding no drift.

    The pair's built-in terms are cleared: an en→vi run now always has them, so
    a run with no pair at all cannot otherwise be reached.
    """
    from pdfdoc import write_pdf

    from polyglot_lc.pipeline import coordinator

    monkeypatch.setattr(coordinator, "default_glossary", lambda src, tgt: {})

    path = write_pdf(
        tmp_path / "nmt.pdf",
        "NMT is used here. NMT is used again in this paragraph.\n",
    )
    result = runner.invoke(
        app,
        ["translate", "--file", str(path), "--out", str(tmp_path / "bare.md")],
    )

    assert result.exit_code == 0, result.output
    assert "no glossary term to compare" in result.output
    assert "glossary terms: 0" not in result.output


def test_the_cli_says_when_the_verifier_could_not_run(mock_env, tmp_path, monkeypatch):
    """A chunk nobody checked must not read like a chunk that passed."""
    from polyglot_lc.agents.verifier import VerifierAgent, VerifyResult

    monkeypatch.setattr(
        VerifierAgent,
        "verify",
        lambda self, source, translation, src_lang, tgt_lang, glossary_md="", **kw: VerifyResult(
            ok=True, reasons=["verifier unavailable; skipped"], skipped=True
        ),
    )
    result = runner.invoke(
        app,
        ["translate", "--file", str(_doc(tmp_path)), "--out", str(tmp_path / "v.md")],
    )

    assert result.exit_code == 0, result.output
    assert "verifier unavailable for" in result.output
    assert "shipped without a verdict" in result.output


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
