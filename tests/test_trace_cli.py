"""Tests for ``polyglot trace`` and for the two guarantees the trace makes.

These run against a real mock-mode pipeline run, so they cover the writer, the
reader and the command together — the way a person uses them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from polyglot_lc.cli import app
from polyglot_lc.pipeline import TranslationPipeline

runner = CliRunner()

#: A sentence the sample document does not contain, planted so a leak is provable.
PLANTED = "PLANTED-SENTENCE-THAT-MUST-NEVER-REACH-THE-TRACE"


@pytest.fixture()
def finished_run(mock_env, tmp_path):
    """One complete mock-mode run, with its log left on disk.

    ``mock_env`` turns logging off (most of the suite does not want log files),
    so it is turned back on here along with a run id the report can name.
    """
    import os

    from pdfdoc import write_pdf

    os.environ["LOG_JSONL"] = "1"
    source = write_pdf(
        tmp_path / "doc.pdf",
        "# Demo Doc\n\n## Intro\n\n"
        f"{PLANTED} Neural Machine Translation is widely used by the Berlin team.\n\n"
        "## Details\n\nThe pipeline splits documents. Quality depends on context.\n",
    )
    pipeline = TranslationPipeline(quiet=True, run_id="20261012_015104_dad6b2")
    result = pipeline.run(file_path=str(source), src="en", tgt="vi", output_format="md")
    assert result.log_path
    return Path(result.log_path)


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_view(path: Path) -> str:
    """The rendered trace, read the way the command reads it."""
    from polyglot_lc.ui.trace import read_trace

    return read_trace(path).as_text()


def test_a_finished_run_records_the_whole_vocabulary_and_every_line_parses(finished_run):
    records = _lines(finished_run)
    kinds = {record.get("type") for record in records}
    assert {"run_start", "step_start", "step_end", "decision", "run_end"} <= kinds
    assert records[0]["run_id"] == "20261012_015104_dad6b2"


def test_run_start_is_the_settings_and_run_end_is_the_totals(finished_run):
    records = _lines(finished_run)
    start = next(r for r in records if r.get("type") == "run_start")
    end = records[-1]
    assert start["file"] == "doc.pdf" and start["sha256"] and start["plan_hash"]
    assert start["chunks"] >= 1
    assert {agent["name"] for agent in start["agents"]} >= {"planner", "translator", "verifier"}
    assert all("http" not in agent["host"] for agent in start["agents"])
    assert end["type"] == "run_end" and end["status"] == "done" and end["exit_code"] == 0
    assert end["tokens"]["sent"] > 0 and end["chunks_done"] >= 1
    assert end["problems"] == sum(1 for r in records if r.get("type") == "error")


def test_every_routing_decision_carries_the_numbers_that_made_it(finished_run):
    """A decision with no reason is exactly the bug this feature exists to fix."""
    decisions = [r for r in _lines(finished_run) if r.get("type") == "decision"]
    assert decisions
    fast = [d for d in decisions if d["kind"] == "fast_path"]
    assert fast, "the gate always decides, even when the answer is no"
    for decision in fast:
        assert decision["reasons"], decision
        assert "complexity" in decision["numbers"]
        assert isinstance(decision["numbers"]["draft_score"], (int, float))
    for kind in {d["kind"] for d in decisions}:
        assert kind in {"fast_path", "refine_gate", "retranslate", "verify_mode"}


def test_the_run_id_given_on_the_command_line_names_the_trace(mock_env, tmp_path):
    """``--run-id`` is how a launcher ties its own record to this run's trace."""
    import os

    from pdfdoc import write_pdf

    os.environ["LOG_JSONL"] = "1"
    source = write_pdf(
        tmp_path / "doc.pdf",
        "# Demo Doc\n\n## Intro\n\nNeural Machine Translation is widely used here.\n",
    )
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "translate",
            "--file",
            str(source),
            "--out",
            str(out),
            "--run-id",
            "20301231_235959_abcdef",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    logs = list((tmp_path / "logs").glob("run_*.jsonl"))
    assert logs, result.output
    records = _lines(logs[0])
    assert {r["run_id"] for r in records} == {"20301231_235959_abcdef"}


def test_no_environment_value_and_no_document_text_reaches_the_file(finished_run, mock_env):
    """Acceptance criterion 4: metadata only, and no secrets."""
    text = finished_run.read_text(encoding="utf-8")
    assert PLANTED not in text
    assert "sk-mock" not in text
    assert "sk-" not in text.replace("sk-\u200b", "")
    assert mock_env is not None  # the fixture owns the planted env values


def test_trace_prints_settings_reasons_problems_and_totals(finished_run):
    result = runner.invoke(app, ["trace", str(finished_run)])
    assert result.exit_code == 0
    out = result.stdout
    assert "doc.pdf" in out and "chunk-000" in out
    assert "route=" in out and "why:" in out
    assert "Totals" in out


def test_trace_json_is_parseable_and_stable(finished_run):
    first = runner.invoke(app, ["trace", str(finished_run), "--json"]).stdout
    second = runner.invoke(app, ["trace", str(finished_run), "--json"]).stdout
    assert first == second
    payload = json.loads(first)
    assert payload["run_id"] == "20261012_015104_dad6b2"
    assert payload["started"]["file"] == "doc.pdf"
    assert payload["timeline"]


def test_trace_explains_a_run_that_never_started(tmp_path):
    empty = tmp_path / "run_empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["trace", str(empty), "--log-dir", str(tmp_path)])
    assert result.exit_code == 0  # reading a broken run is not a command failure
    assert "no records" in result.stdout.lower()


def test_trace_refuses_to_read_a_log_outside_the_log_root(tmp_path):
    """A path outside the root is not read, whatever it points at."""
    outside = tmp_path / "elsewhere" / "run_stolen.jsonl"
    outside.parent.mkdir()
    outside.write_text(json.dumps({"type": "run_start", "file": "secret.pdf"}), encoding="utf-8")
    root = tmp_path / "logs"
    root.mkdir()
    result = runner.invoke(app, ["trace", str(outside), "--log-dir", str(root)])
    assert result.exit_code == 1
    assert "outside the log root" in result.stdout
    assert "secret.pdf" not in result.stdout


def test_trace_output_survives_a_legacy_windows_console(finished_run):
    """A cp1252 console cannot encode the placeholder marker; it must still print.

    Ran for real on this machine: before the guard, ``polyglot trace`` died with
    ``UnicodeEncodeError`` on U+27E6 and printed a traceback instead of the run.
    """
    from polyglot_lc.cli import _console_safe

    out = _console_safe("dropped \u27e6CODE_1\u27e7 and ti\u1ebfng Vi\u1ec7t", encoding="cp1252")
    out.encode("cp1252")  # the whole point: this must not raise
    assert "CODE_1" in out  # and the rest of the line survives the replacement

    view = read_view(finished_run)
    assert view  # a real run's view is printable on that console as well
    _console_safe(view, encoding="cp1252").encode("cp1252")


def test_trace_says_so_when_there_is_no_run_at_all(tmp_path):
    result = runner.invoke(app, ["trace", "--log-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "no run to read" in result.stdout.lower()


def test_trace_finds_the_newest_run_under_the_log_root(mock_env, finished_run):
    """No argument: the newest run below the log directory is the one printed."""
    result = runner.invoke(app, ["trace", "--log-dir", str(finished_run.parent)])
    assert result.exit_code == 0
    assert "run_start" in result.stdout or "Settings" in result.stdout
