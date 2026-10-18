"""Tests for the UI layer: events, JSONL sink, simple step reporter."""

from __future__ import annotations

import json
import pathlib

import pytest
from rich.console import Console

from polyglot_lc.llm.client import TokenUsage
from polyglot_lc.ui.events import Event, EventSink
from polyglot_lc.ui.simple import SimpleUI


def test_event_json_shape():
    e = Event(step="translate", chunk_id="chunk-000", status="ok", latency=12.5)
    data = json.loads(e.to_json())
    assert {"ts", "step", "chunk_id", "status", "latency", "error"} <= set(data)


def test_sink_writes_jsonl(tmp_path):
    sink = EventSink(tmp_path, enabled=True)
    sink.info("ingest", message="loading")
    sink.emit(Event(step="translate", chunk_id="chunk-000", status="ok", latency=1.0))
    lines = [json.loads(l) for l in sink.path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[1]["step"] == "translate"


def test_sink_disabled_no_file(tmp_path):
    sink = EventSink(tmp_path, enabled=False)
    sink.info("x")
    assert sink.path is None


def test_sink_tail_and_counts(tmp_path):
    sink = EventSink(tmp_path, enabled=False)
    for i in range(5):
        sink.emit(Event(step=f"s{i}", status="ok"))
    sink.fail("boom", error="kaput")
    tail = list(sink.tail(2))
    assert tail[-1].status == "fail"
    counts = sink.counts()
    assert counts["ok"] == 5 and counts["fail"] == 1


def test_simpleui_prints_steps_and_captures_failure(tmp_path):
    sink = EventSink(tmp_path, enabled=False)
    console = Console(file=__import__("io").StringIO(), width=100, force_terminal=False)
    ui = SimpleUI(sink, console=console, quiet=False)
    with ui:
        with ui.step("ok-step"):
            pass
        with pytest.raises(ValueError), ui.step("bad-step", "chunk-000"):
            raise ValueError("kaput")
    assert any("bad-step" in e for e in ui.errors)
    rendered = console.file.getvalue()
    assert "Running ok-step" in rendered
    assert "Running bad-step" in rendered


def test_simpleui_quiet_suppresses_step_lines(tmp_path):
    import io

    sink = EventSink(tmp_path, enabled=False)
    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    ui = SimpleUI(sink, console=console, quiet=True)
    with ui, ui.step("translate"):
        pass
    assert console.file.getvalue().strip() == ""
    # events are still logged regardless of quiet mode
    assert next(iter(sink.tail(1))).step == "translate"


def test_a_waiting_chunk_is_visible_on_both_surfaces(tmp_path):
    """A run that waits minutes for a provider must not look like a hang.

    The wait prints on the console and is written as a ``retry`` event: that
    status is what keeps the chunk out of the web UI's "failed" state and
    captions the progress bar.
    """
    import io

    sink = EventSink(tmp_path, enabled=False)
    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    ui = SimpleUI(sink, console=console, quiet=False)
    ui.retry("verify", "chunk-014", "TransientError: timeout - waiting 40s, then attempt 3")

    rendered = console.file.getvalue()
    assert "Waiting: verify (chunk-014)" in rendered
    assert "waiting 40s" in rendered
    event = next(iter(sink.tail(1)))
    assert (event.status, event.step, event.chunk_id) == ("retry", "verify", "chunk-014")
    assert event.type == "retry_wait"
    # A wait counts as a problem, so the run's own count reconciles with the
    # ``retry`` lines a reader finds in its file.
    assert sink.problem_count == 1


def test_simpleui_set_stats_and_errors(tmp_path):
    sink = EventSink(tmp_path, enabled=False)
    ui = SimpleUI(sink, quiet=True)
    ui.set_stats(chunks=3, avg=8.0)
    ui.add_error("boom")
    ui.set_total_chunks(5)
    ui.advance()
    assert ui.stats == {"chunks": 3, "avg": 8.0}
    assert ui.errors == ["boom"]


def test_simpleui_failure_is_recorded_as_an_event(tmp_path):
    """The live path (SimpleUI.step) is the only step context there is."""
    sink = EventSink(tmp_path, enabled=False)
    ui = SimpleUI(sink, quiet=True)
    with pytest.raises(RuntimeError), ui.step("solo"):
        raise RuntimeError("x")
    assert next(iter(sink.tail(1))).status == "fail"
    assert any("solo" in e for e in ui.errors)


def test_step_end_carries_duration_and_the_token_delta_for_that_step(tmp_path):
    """Tokens per step are a delta against the global counter, not a total."""
    TokenUsage.reset()
    sink = EventSink(tmp_path, enabled=True)
    ui = SimpleUI(sink, quiet=True)
    with ui.step("translate", "chunk-000"):
        TokenUsage.add(120, 40, agent="TRANSLATOR", model="m", provider="http://x/v1")
    line = json.loads(sink.path.read_text(encoding="utf-8").splitlines()[-1])
    assert line["status"] == "ok"
    assert line["type"] == "step_end"
    assert line["duration_ms"] >= 0
    assert line["tokens"] == {"sent": 120, "received": 40, "cached": 0}


def test_a_failing_step_records_a_cause_from_the_closed_set(tmp_path):
    from polyglot_lc.llm.resilience import RateLimitError
    from polyglot_lc.ui.trace import CAUSES

    sink = EventSink(tmp_path, enabled=True)
    ui = SimpleUI(sink, quiet=True)
    with pytest.raises(RateLimitError), ui.step("translate", "chunk-000"):
        raise RateLimitError("429 rate limit exceeded")
    line = json.loads(sink.path.read_text(encoding="utf-8").splitlines()[-1])
    assert line["status"] == "fail"
    assert line["cause"] == "rate_limit" and line["cause"] in CAUSES


def test_every_record_carries_the_run_id_and_the_existing_keys_survive(tmp_path):
    """A reader written before the richer records existed must not need updating."""
    sink = EventSink(tmp_path, enabled=True, run_id="20261012_015104_dad6b2")
    sink.record(type="run_note", text="reasoning bank unavailable")
    line = json.loads(sink.path.read_text(encoding="utf-8").splitlines()[-1])
    assert line["run_id"] == "20261012_015104_dad6b2"
    assert line["type"] == "run_note" and line["ts"] > 0
    assert {"ts", "step", "chunk_id", "status", "latency", "error"} <= set(line)
    assert line["text"] == "reasoning bank unavailable"


def test_an_error_record_speaks_the_old_vocabulary_too(tmp_path):
    """The web UI filters on status, so an error record has to look like a fail."""
    sink = EventSink(tmp_path, enabled=True)
    sink.record(
        type="error", step="translate", chunk_id="chunk-000", cause="rate_limit", message="429"
    )
    line = json.loads(sink.path.read_text(encoding="utf-8").splitlines()[-1])
    assert line["status"] == "fail"
    assert line["error"] == "429" and line["cause"] == "rate_limit"
    assert sink.problem_count == 1


def test_the_problem_count_reconciles_with_the_failures_written(tmp_path):
    sink = EventSink(tmp_path, enabled=True)
    sink.fail("translate", "chunk-000", error="boom")
    sink.record(type="error", step="verify", cause="verify", message="nope")
    sink.emit(Event(step="translate", chunk_id="chunk-001", status="ok"))
    assert sink.problem_count == 2


def test_events_written_through_emit_carry_the_run_id(tmp_path):
    sink = EventSink(tmp_path, enabled=True, run_id="r1")
    sink.emit(Event(step="translate", chunk_id="chunk-000", status="ok"))
    line = json.loads(sink.path.read_text(encoding="utf-8").splitlines()[-1])
    assert line["run_id"] == "r1"
    assert {"ts", "step", "chunk_id", "status", "latency", "error"} <= set(line)


def test_record_is_a_silent_no_op_when_logging_is_off(tmp_path):
    sink = EventSink(tmp_path, enabled=False)
    sink.record(type="run_start", file="x.pdf")  # must not raise, must not write
    assert sink.path is None


def test_a_run_id_minted_here_matches_the_web_ui_shape(tmp_path):
    import re

    sink = EventSink(tmp_path, enabled=False)
    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", sink.run_id)


def test_a_resume_record_in_the_log_does_not_break_the_trace_reader(tmp_path):
    """The pipeline's early `resume` record is an unknown kind to the trace
    reader, which must treat it as a bare step and move on — not crash or
    swallow the surrounding records."""
    from polyglot_lc.ui.trace import read_trace

    log = tmp_path / "run_20261012_015104_abc123.jsonl"
    log.write_text(
        json.dumps({"type": "run_start", "file": "doc.pdf", "chunks": 4})
        + "\n"
        + json.dumps(
            {
                "type": "resume",
                "resume_requested": True,
                "resumed_chunks": 3,
                "checkpoint_usable": True,
            }
        )
        + "\n"
        + json.dumps({"type": "run_end", "status": "done", "exit_code": 0}),
        encoding="utf-8",
    )
    trace = read_trace(log)
    assert trace.run_id == "20261012_015104_abc123"
    assert trace.malformed == 0
    assert trace.ended is not None


def test_simpleui_run_with_cached_metrics_does_not_break(tmp_path):
    TokenUsage.reset()
    TokenUsage.add(
        100, 10, agent="TRANSLATOR", cached_tokens=60, model="mock-model", provider="http://x/v1"
    )
    snap = TokenUsage.snapshot()
    assert snap["cached"] == 60
    TokenUsage.reset()


def test_events_carry_api_call_concurrency_metrics(tmp_path):
    """Web UI reads active API call concurrency from events, so step/retry events must carry it."""
    from polyglot_lc.llm.client import _ActiveCalls

    _ActiveCalls.reset()
    TokenUsage.reset()
    sink = EventSink(tmp_path, enabled=True)
    ui = SimpleUI(sink, quiet=True)
    _ActiveCalls.enter()
    _ActiveCalls.enter()
    _ActiveCalls.exit()
    with ui.step("translate", "chunk-001"):
        ui.retry("translate", "chunk-002", "waiting")
    _ActiveCalls.exit()
    _ActiveCalls.reset()

    lines = [json.loads(l) for l in sink.path.read_text(encoding="utf-8").splitlines()]
    starts = [line for line in lines if line.get("type") == "step_start"]
    retries = [line for line in lines if line.get("type") == "retry_wait"]
    assert starts and retries
    for line in starts + retries:
        assert "active_api_calls" in line and "max_active_api_calls" in line
    assert starts[0]["active_api_calls"] == 1
    assert starts[0]["max_active_api_calls"] == 2
    assert retries[0]["active_api_calls"] == 1
    assert retries[0]["max_active_api_calls"] == 2


def test_webui_report_shows_api_call_concurrency():
    app_js = pathlib.Path("webui/static/app.js").read_text(encoding="utf-8")
    assert "Api calls" in app_js
    assert "active_api_calls" in app_js and "max_active_api_calls" in app_js
    assert "chưa đo được" in app_js
    assert 'callWords.push(`${formatCount(peak)} luồng`)' in app_js or "cao nhất" in app_js


def test_webui_log_lines_expand_and_collapse():
    """Each history line folds to two lines and opens onto its event's fields."""
    app_js = pathlib.Path("webui/static/app.js").read_text(encoding="utf-8")
    app_css = pathlib.Path("webui/static/app.css").read_text(encoding="utf-8")
    # A chevron per line, wired to aria-expanded, and the event kept on the entry.
    assert "log-toggle" in app_js
    assert 'toggle.setAttribute("aria-expanded"' in app_js
    assert "event: event" in app_js
    # The detail panel lists the event's own fields under Vietnamese labels.
    assert "log-detail" in app_js and "logDetailRows" in app_js
    assert "LOG_DETAIL_WORDS" in app_js
    # Collapsed, the message is clamped to two lines; open, the chevron turns.
    assert '-webkit-line-clamp: 2' in app_css
    assert '.log-line[data-expanded="true"] .log-toggle { transform: rotate(90deg)' in app_css
    # The narrow layout keeps the fold on the same row as the clock.
    assert (
        ".log-line { grid-template-columns: max-content max-content minmax(0, 1fr) max-content; }"
        in app_css
    )
