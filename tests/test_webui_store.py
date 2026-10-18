"""Tests for webui.store: the run, skill and glossary stores.

There is no configuration store to test: the web application has no view of
`.env`, and `tests/test_webui_server.py` asserts that no route exposes one.
"""

from __future__ import annotations

import json

import pytest

from webui.store import GlossaryStore, RunRecord, RunsStore, SkillsStore


def _record(run_id: str = "r1", **over) -> RunRecord:
    """A record with every field filled, so a test overrides only what it means."""
    fields = {
        "run_id": run_id,
        "started": 1.0,
        "status": "done",
        "file": "f.pdf",
        "src": "en",
        "tgt": "vi",
        "domain": "tech",
        "chunks": 3,
        "exit_code": 0,
        "output_path": "o.md",
        "mock": False,
        "pid": None,
        "command": ["x"],
        "note": "",
    }
    fields.update(over)
    return RunRecord(**fields)


# --------------------------------- run store --------------------------------
def test_run_record_round_trip_and_order(tmp_path):
    s = RunsStore(tmp_path)
    for rid, ts in (("b", 2.0), ("a", 1.0)):
        s.write_record(_record(rid, started=ts))
    assert [r.run_id for r in s.list_records()] == ["b", "a"]
    assert s.read_record("a").chunks == 3
    assert s.read_record("missing") is None


def test_run_record_carries_its_own_automatic_continues(tmp_path):
    """The supervisor's budget lives on the record, not in the server's memory.

    A restart in the middle of a provider outage must not forget how many
    automatic continues a run already spent, and the run it is waiting on has to
    stay findable while the browser is closed for the whole wait.
    """
    s = RunsStore(tmp_path)
    s.write_record(_record("r1", auto_resumes=2, retry_at=1234.5))
    back = s.read_record("r1")
    assert back.auto_resumes == 2 and back.retry_at == 1234.5

    # A record written before these fields existed reads as "none yet" rather
    # than as a malformed record: every run in `logs/webui` predates them.
    s.write_record(_record("r2"))
    legacy = s.dir_for("r2") / "run.json"
    data = json.loads(legacy.read_text(encoding="utf-8"))
    for field in ("auto_resumes", "retry_at"):
        del data[field]
    legacy.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    old = s.read_record("r2")
    assert old is not None and old.auto_resumes == 0 and old.retry_at == 0.0


def test_last_run_end_reads_only_the_newest_attempt(tmp_path):
    """The supervisor asks this on a timer, so it reads one file, not all of them."""
    s = RunsStore(tmp_path)
    d = s.dir_for("r1")
    d.mkdir(parents=True)
    (d / "run_20260101_000000.jsonl").write_text(
        '{"type": "run_end", "chunks_failed": 9}\n', encoding="utf-8"
    )
    (d / "run_20260101_010000.jsonl").write_text(
        '{"type": "chunk"}\n{"type": "run_end", "chunks_failed": 1}\n', encoding="utf-8"
    )

    assert s.last_run_end("r1")["chunks_failed"] == 1
    assert s.last_run_end("r-missing") == {}


def test_read_events_reads_every_attempt_of_a_run(tmp_path):
    """Continuing a run appends a log file to the same directory.

    Reading only the newest one would wipe the run's earlier history off the
    page — its progress bar included — the moment the reader pressed "Tiếp tục".
    """
    s = RunsStore(tmp_path)
    d = s.dir_for("r1")
    d.mkdir(parents=True)
    (d / "run_20260101_000000.jsonl").write_text('{"type": "chunk", "n": 1}\n', encoding="utf-8")
    (d / "run_20260101_010000.jsonl").write_text('{"type": "resume"}\n', encoding="utf-8")

    events, nxt, malformed = s.read_events("r1", 0)
    assert [e["type"] for e in events] == ["chunk", "resume"]
    assert nxt == 2 and malformed == 0
    # A page polling with a cursor keeps its place across the second attempt.
    assert s.read_events("r1", 1) == ([{"type": "resume"}], 2, 0)
    assert s.log_path("r1").name == "run_20260101_010000.jsonl"
    assert [p.name for p in s.log_files("r1")] == [
        "run_20260101_000000.jsonl",
        "run_20260101_010000.jsonl",
    ]


def test_read_events_counts_malformed_lines(tmp_path):
    s = RunsStore(tmp_path)
    d = s.dir_for("r1")
    d.mkdir(parents=True)
    (d / "run_20260101_000000.jsonl").write_text(
        '{"step": "ingest", "status": "ok"}\nnot json\n{"step": "plan", "status": "ok"}\n',
        encoding="utf-8",
    )
    events, nxt, malformed = s.read_events("r1", 0)
    assert [e["step"] for e in events] == ["ingest", "plan"]
    assert nxt == 2 and malformed == 1
    assert s.read_events("r1", 2) == ([], 2, 1)


def test_run_without_a_log_has_no_events(tmp_path):
    s = RunsStore(tmp_path)
    s.write_record(_record("r2", status="running", exit_code=None, pid=1))
    assert s.log_path("r2") is None
    assert s.read_events("r2", 0) == ([], 0, 0)


def test_tail_output_prefers_stderr_and_keeps_the_last_lines(tmp_path):
    """A child that dies before its first event explains itself only here."""
    s = RunsStore(tmp_path)
    s.write_record(_record("r1", status="failed", exit_code=2))
    d = tmp_path / "r1"
    (d / "stderr.log").write_text("line 1\n\nline 2\nline 3\n", encoding="utf-8")
    (d / "stdout.log").write_text("noise\n", encoding="utf-8")
    assert s.tail_output("r1", limit=2) == "line 2\nline 3"


def test_tail_output_falls_back_to_stdout_and_tolerates_absence(tmp_path):
    s = RunsStore(tmp_path)
    s.write_record(_record("r1"))
    d = tmp_path / "r1"
    (d / "stderr.log").write_text("   \n", encoding="utf-8")  # nothing but blanks
    (d / "stdout.log").write_text("Configuration error: missing key\n", encoding="utf-8")
    assert s.tail_output("r1") == "Configuration error: missing key"
    assert s.tail_output("nope") == ""
    assert s.tail_output("../escape") == ""


# -------------------------------- skill store -------------------------------
def test_list_skills_skips_directories_and_bad_json(tmp_path):
    (tmp_path / "en-vi").mkdir(parents=True)
    (tmp_path / "en-vi" / "general.json").write_text(
        json.dumps(
            {
                "instruction": "i",
                "mean_score": 8.5,
                "domain": "general",
                "src_lang": "en",
                "tgt_lang": "vi",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "en-vi" / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "en-vi" / "session_abc.json").write_text("{}", encoding="utf-8")
    (tmp_path / "en-vi" / "a-directory.json").mkdir()

    st = SkillsStore(tmp_path)
    names = [(s["pair"], s["domain"]) for s in st.list_skills()]
    assert names == [("en-vi", "general")]
    assert st.get_skill("en-vi", "general")["mean_score"] == 8.5
    assert st.get_skill("en-vi", "broken") is None


def test_skill_history_is_read(tmp_path):
    (tmp_path / "en-vi").mkdir(parents=True)
    (tmp_path / "en-vi" / "general.json").write_text("{}", encoding="utf-8")
    (tmp_path / "en-vi" / "general.history.jsonl").write_text(
        '{"score": 7.0, "ts": 1}\n{"score": 9.0, "ts": 2}\n', encoding="utf-8"
    )
    assert [h["score"] for h in SkillsStore(tmp_path).history("en-vi", "general")] == [7.0, 9.0]


def test_skill_history_tolerates_a_missing_file(tmp_path):
    (tmp_path / "en-vi").mkdir(parents=True)
    assert SkillsStore(tmp_path).history("en-vi", "general") == []


# ------------------------------ glossary store ------------------------------
def test_glossary_round_trip_with_bom(tmp_path):
    (tmp_path / "g.csv").write_bytes("\ufeffsource,target\nNMT, dịch máy\n".encode("utf-8"))
    st = GlossaryStore(tmp_path)
    assert st.read("g.csv")[1] == ["NMT", "dịch máy"]
    st.write("h.csv", [["a", "b"]])
    assert st.read("h.csv") == [["a", "b"]]
    assert {f["name"] for f in st.list_files()} == {"g.csv", "h.csv"}


def test_glossary_rejects_a_name_that_escapes(tmp_path):
    st = GlossaryStore(tmp_path)
    with pytest.raises(ValueError):
        st.read("../secrets.csv")
