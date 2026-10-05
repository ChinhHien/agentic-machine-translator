"""Tests for the UI layer: events, JSONL sink, simple step reporter."""

from __future__ import annotations

import json

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
        with pytest.raises(ValueError):
            with ui.step("bad-step", "chunk-000"):
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
    with ui:
        with ui.step("translate"):
            pass
    assert console.file.getvalue().strip() == ""
    # events are still logged regardless of quiet mode
    assert list(sink.tail(1))[0].step == "translate"


def test_simpleui_set_stats_and_errors(tmp_path):
    sink = EventSink(tmp_path, enabled=False)
    ui = SimpleUI(sink, quiet=True)
    ui.set_stats(chunks=3, avg=8.0)
    ui.add_error("boom")
    ui.set_total_chunks(5)
    ui.advance()
    assert ui.stats == {"chunks": 3, "avg": 8.0}
    assert ui.errors == ["boom"]


def test_step_standalone_sink(tmp_path):
    from polyglot_lc.ui.step import step

    sink = EventSink(tmp_path, enabled=False)
    with step("solo", sink=sink):
        pass
    assert list(sink.tail(1))[0].status == "ok"


def test_step_standalone_failure(tmp_path):
    from polyglot_lc.ui.step import step

    sink = EventSink(tmp_path, enabled=False)
    with pytest.raises(RuntimeError):
        with step("solo", sink=sink):
            raise RuntimeError("x")
    assert list(sink.tail(1))[0].status == "fail"


def test_simpleui_run_with_cached_metrics_does_not_break(tmp_path):
    TokenUsage.reset()
    TokenUsage.add(100, 10, agent="TRANSLATOR", cached_tokens=60,
                   model="mock-model", provider="http://x/v1")
    snap = TokenUsage.snapshot()
    assert snap["cached"] == 60
    TokenUsage.reset()
