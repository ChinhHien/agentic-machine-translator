"""Tests for the structured trace: cause vocabulary, plan hash, reader, resolver."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from polyglot_lc.config import ConfigError
from polyglot_lc.llm.resilience import PermanentError, RateLimitError, TransientError
from polyglot_lc.segment.chunker import Chunk
from polyglot_lc.ui.trace import (
    CAUSES,
    cause_for,
    mint_run_id,
    plan_hash,
    read_trace,
    resolve_run,
)


# ---------------------------------------------------------------------------
# cause_for
# ---------------------------------------------------------------------------
def test_cause_for_maps_the_taxonomy_the_resilience_layer_already_computes():
    assert cause_for(RateLimitError("429 rate limit exceeded")) == "rate_limit"
    assert cause_for(RateLimitError("only one concurrent request allowed")) == "concurrency"
    assert cause_for(TransientError("read timeout")) == "timeout"
    assert cause_for(PermanentError("401 unauthorized", status=401)) == "auth"
    assert cause_for(PermanentError("403 forbidden", status=403)) == "auth"
    assert cause_for(PermanentError("422 unprocessable", status=422)) == "provider"


def test_cause_for_falls_back_to_unknown_rather_than_inventing_a_cause():
    assert cause_for(ValueError("something nobody classified")) == "unknown"


def test_every_cause_comes_from_the_closed_set():
    for exc in (
        RateLimitError("x"),
        TransientError("y"),
        PermanentError("z"),
        ConfigError("c"),
        ValueError("v"),
        TimeoutError("t"),
    ):
        assert cause_for(exc) in CAUSES
    # A closed set, and this number is part of what closing it means: it moved
    # from 11 to 12 because `emphasis` is a real cause a writer records (a chunk
    # whose sentinel pairs do not match the source is printed plain).
    assert len(CAUSES) == 12


def test_ingest_step_unclassified_failure_is_an_input_cause():
    """The step name is the only evidence that the file, not the provider, failed."""
    assert cause_for(ValueError("cannot open document"), step="ingest") == "input"
    assert cause_for(ValueError("cannot open document"), step="translate") == "unknown"


def test_a_malformed_json_reply_is_a_parse_cause():
    assert cause_for(json.JSONDecodeError("bad", "{}", 0)) == "parse"


# ---------------------------------------------------------------------------
# plan_hash / mint_run_id
# ---------------------------------------------------------------------------
def test_plan_hash_tracks_the_split_not_the_document():
    a = [Chunk(id="chunk-000", text="alpha", token_count=1, index=0)]
    same = [Chunk(id="chunk-000", text="alpha", token_count=1, index=0)]
    resplit = [
        Chunk(id="chunk-000", text="al", token_count=1, index=0),
        Chunk(id="chunk-001", text="pha", token_count=1, index=1),
    ]
    assert plan_hash(a) == plan_hash(same)
    assert plan_hash(a) != plan_hash(resplit)
    assert len(plan_hash(a)) == 12
    assert plan_hash([]) == ""


def test_mint_run_id_matches_the_shape_the_web_ui_mints():
    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", mint_run_id())


# ---------------------------------------------------------------------------
# read_trace
# ---------------------------------------------------------------------------
def _write(tmp_path: Path, lines: list[dict], name: str = "run_20261012_015104.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def test_reader_groups_records_into_settings_a_timeline_and_totals(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "ts": 1.0,
                "run_id": "r1",
                "type": "run_start",
                "file": "d.pdf",
                "sha256": "ab",
                "chunks": 2,
                "agents": [{"name": "translator", "model": "m", "host": "h", "temperature": 0.2}],
            },
            {
                "ts": 2.0,
                "run_id": "r1",
                "type": "step_end",
                "step": "translate",
                "chunk_id": "chunk-000",
                "status": "ok",
                "latency": 10.0,
                "duration_ms": 10,
                "tokens": {"sent": 5, "received": 2, "cached": 0},
            },
            {
                "ts": 3.0,
                "run_id": "r1",
                "type": "decision",
                "kind": "fast_path",
                "chunk_id": "chunk-000",
                "chosen": True,
                "reasons": ["complexity 0.10 <= 0.35"],
                "numbers": {"complexity": 0.1},
            },
            {
                "ts": 4.0,
                "run_id": "r1",
                "type": "error",
                "step": "translate",
                "chunk_id": "chunk-001",
                "cause": "rate_limit",
                "message": "429",
            },
            {"ts": 5.0, "run_id": "r1", "type": "run_end", "status": "done", "duration_ms": 4000},
        ],
    )
    trace = read_trace(path)
    assert trace.run_id == "r1"
    assert trace.started["file"] == "d.pdf"
    assert trace.ended["status"] == "done"
    assert [t.chunk_id for t in trace.timeline] == ["chunk-000", "chunk-001"]
    assert trace.timeline[0].steps[0].name == "translate"
    assert trace.timeline[0].steps[0].tokens["sent"] == 5
    assert trace.timeline[0].route == "fast"
    assert trace.timeline[0].reasons == ["complexity 0.10 <= 0.35"]
    assert [p["cause"] for p in trace.problems] == ["rate_limit"]
    assert trace.timeline[1].problems[0]["cause"] == "rate_limit"
    assert trace.malformed == 0
    assert trace.totals["status"] == "done"


def test_route_is_deep_only_when_a_refine_actually_happened(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "type": "decision",
                "kind": "fast_path",
                "chunk_id": "c0",
                "chosen": False,
                "reasons": [],
            },
            {
                "type": "decision",
                "kind": "refine_gate",
                "chunk_id": "c0",
                "chosen": True,
                "reasons": [],
            },
            {
                "type": "decision",
                "kind": "fast_path",
                "chunk_id": "c1",
                "chosen": False,
                "reasons": [],
            },
            {
                "type": "decision",
                "kind": "refine_gate",
                "chunk_id": "c1",
                "chosen": False,
                "reasons": [],
            },
        ],
    )
    routes = {t.chunk_id: t.route for t in read_trace(path).timeline}
    assert routes == {"c0": "deep", "c1": "standard"}


def test_a_truncated_final_line_is_counted_and_never_hides_earlier_records(tmp_path):
    path = tmp_path / "run_x.jsonl"
    path.write_text(
        json.dumps({"ts": 1.0, "type": "run_start", "file": "d.pdf"})
        + "\n"
        + '{"ts": 2.0, "type": "step_',
        encoding="utf-8",
    )
    trace = read_trace(path)
    assert trace.malformed == 1
    assert trace.started["file"] == "d.pdf"


def test_a_log_from_the_previous_version_reads_as_steps(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "ts": 1.0,
                "step": "translate",
                "chunk_id": "chunk-000",
                "status": "ok",
                "latency": 4.0,
                "error": "",
            },
        ],
    )
    trace = read_trace(path)
    assert trace.started == {} and trace.malformed == 0
    assert trace.timeline[0].chunk_id == "chunk-000"
    assert trace.timeline[0].steps[0].name == "translate"
    # The id falls back to the file name when no record carried one.
    assert trace.run_id == "20261012_015104"


def test_a_missing_file_reads_as_an_empty_trace_that_says_so(tmp_path):
    trace = read_trace(tmp_path / "nope.jsonl")
    assert trace.malformed == 0
    assert trace.timeline == []
    assert "no records" in trace.as_text().lower()
    assert trace.as_dict()["run_id"] == "nope"


def test_as_text_shows_settings_reasons_problems_and_totals(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "type": "run_start",
                "run_id": "r1",
                "file": "d.pdf",
                "sha256": "ab",
                "bytes": 10,
                "pages": 2,
                "chunks": 1,
                "plan_hash": "3ab1",
                "src": "en",
                "tgt": "vi",
                "agents": [{"name": "translator", "model": "m", "host": "h", "temperature": 0.2}],
                "max_tokens": 700,
            },
            {
                "type": "decision",
                "run_id": "r1",
                "kind": "fast_path",
                "chunk_id": "chunk-000",
                "chosen": False,
                "reasons": ["complexity 0.62 > 0.35"],
            },
            {
                "type": "error",
                "run_id": "r1",
                "step": "translate",
                "chunk_id": "chunk-000",
                "cause": "rate_limit",
                "message": "429 too many requests",
            },
            {
                "type": "run_end",
                "run_id": "r1",
                "status": "done",
                "duration_ms": 120650,
                "calls": 3,
            },
        ],
    )
    text = read_trace(path).as_text()
    assert "d.pdf" in text and "3ab1" in text
    assert "complexity 0.62 > 0.35" in text
    assert "[rate_limit]" in text
    assert "Totals" in text


def test_as_text_reports_the_drift_the_consistency_pass_left_behind(tmp_path):
    """Tổng kết phải nói cả phần *chưa* sửa: chunk bỏ sót và bản vá bị từ chối."""
    path = _write(
        tmp_path,
        [
            {"type": "run_start", "run_id": "r1", "file": "d.pdf"},
            {
                "type": "run_end",
                "run_id": "r1",
                "status": "done",
                "consistency_inconsistencies": 4,
                "consistency_patched": 2,
                "consistency_skipped": 5,
                "consistency_rejected": 1,
            },
        ],
    )
    text = read_trace(path).as_text()
    assert "consistency: 4 drifting term(s)" in text
    assert "patched: 2 chunk(s)" in text
    assert "skipped: 5" in text and "rejected: 1" in text


def test_as_text_keeps_quiet_when_the_chunks_agreed(tmp_path):
    path = _write(
        tmp_path,
        [
            {"type": "run_start", "run_id": "r1", "file": "d.pdf"},
            {"type": "run_end", "run_id": "r1", "status": "done"},
        ],
    )
    assert "consistency:" not in read_trace(path).as_text()


def test_as_dict_is_stable_between_two_reads(tmp_path):
    path = _write(tmp_path, [{"type": "run_start", "run_id": "r1", "file": "d.pdf"}])
    first = json.dumps(read_trace(path).as_dict(), sort_keys=True)
    second = json.dumps(read_trace(path).as_dict(), sort_keys=True)
    assert first == second


# ---------------------------------------------------------------------------
# resolve_run
# ---------------------------------------------------------------------------
def test_resolve_run_takes_an_explicit_path_or_finds_the_newest_recursively(tmp_path):
    nested = tmp_path / "webui" / "20261012_015104_dad6b2"
    nested.mkdir(parents=True)
    older = tmp_path / "run_20261012_010000.jsonl"
    newer = nested / "run_20261012_015104.jsonl"
    older.write_text("", encoding="utf-8")
    newer.write_text("", encoding="utf-8")

    assert resolve_run(str(newer), tmp_path).resolve() == newer.resolve()
    assert resolve_run("20261012_015104_dad6b2", tmp_path).resolve() == newer.resolve()
    assert resolve_run("run_20261012_015104.jsonl", tmp_path).resolve() == newer.resolve()
    assert resolve_run(None, tmp_path).resolve() == newer.resolve()


def test_resolve_run_refuses_a_path_outside_the_log_root(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    outside = tmp_path / "elsewhere.jsonl"
    outside.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        resolve_run(str(outside), root)


def test_resolve_run_says_which_directory_it_searched_when_it_finds_nothing(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        resolve_run(None, tmp_path)
    assert str(tmp_path) in str(err.value)
