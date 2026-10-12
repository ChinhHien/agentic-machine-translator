"""Tests for webui.runs: launching and supervising one CLI run per document."""

from __future__ import annotations

from pathlib import Path

import pytest

from webui.runs import RunManager, RunPlanError


class FakeProc:
    """A stand-in for `subprocess.Popen`, recording how it was launched."""

    def __init__(self, argv, **kw):
        self.argv = argv
        self.kw = kw
        self.pid = 4242
        self.returncode = None

    def poll(self):
        return self.returncode


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A repository-shaped temp dir: just `input/`, enough to start a run."""
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "doc.pdf").write_bytes(b"%PDF-1.4 stub")
    monkeypatch.setattr("webui.runs.REPO_ROOT", tmp_path)
    return tmp_path


@pytest.fixture()
def planned(monkeypatch):
    """Replace the probe call so start-up logic is tested without a subprocess."""
    monkeypatch.setattr(
        RunManager,
        "plan",
        lambda self, file_path, max_tokens: {
            "chunks": 3,
            "output_path": "/out/x.en.pdf",
            "title": "doc.pdf",
        },
    )


def _manager(repo, made=None):
    """A manager whose children are fake, recording every spawn in `made`."""
    made = [] if made is None else made

    def popen(argv, **kw):
        made.append((argv, kw))
        return FakeProc(argv, **kw)

    return RunManager(repo, python="PY", popen=popen)


def test_start_builds_the_command_and_the_run_dir(repo, planned):
    made: list = []
    mgr = _manager(repo, made)

    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary="g.csv",
        max_tokens=555,
        mock=True,
    )

    argv, kw = made[0]
    assert argv[:3] == ["PY", "-m", "polyglot_lc.cli"]
    assert argv[3] == "translate"
    for flag, value in (
        ("--src", "vi"),
        ("--tgt", "en"),
        ("--domain", "tech"),
        ("--glossary", "g.csv"),
        ("--max-tokens", "555"),
    ):
        assert argv[argv.index(flag) + 1] == value, flag
    assert kw["cwd"] == repo
    assert kw["env"]["LOGS_DIR"] == str(repo / "logs" / "webui" / rec.run_id)
    assert kw["env"]["LOG_JSONL"] == "1"
    assert kw["env"]["LLM_MOCK"] == "1"
    assert kw["env"]["PYTHONIOENCODING"] == "utf-8"  # log tiếng Việt không làm chết con
    assert Path(rec.output_path).parent == repo / "output" / "webui" / rec.run_id
    assert (repo / "logs" / "webui" / rec.run_id / "run.json").is_file()
    assert rec.status == "running" and rec.chunks == 3 and rec.pid == 4242


def test_resume_adds_the_flag_and_leaves_the_rest_alone(repo, planned):
    """Continuing an interrupted document is a real capability of the CLI; the
    page offers it as "Tiếp tục", so the flag has to reach the child — and a
    fresh run must not carry it."""
    made: list = []
    mgr = _manager(repo, made)

    mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
        resume=True,
    )
    mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )

    assert "--resume" in made[0][0]
    assert "--resume" not in made[1][0]


def test_input_path_is_validated_against_the_input_directory(repo, planned):
    """The source pane asks for the run's own input file by name."""
    mgr = _manager(repo)
    assert mgr.input_path("doc.pdf") == repo / "input" / "doc.pdf"
    with pytest.raises(ValueError):
        mgr.input_path("../.env")


def test_two_runs_keep_separate_log_dirs(repo, planned):
    mgr = _manager(repo)
    a = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    b = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )

    assert a.run_id != b.run_id
    assert mgr.logs_dir(a.run_id) != mgr.logs_dir(b.run_id)
    assert mgr.env_for(a)["LOGS_DIR"] == str(mgr.logs_dir(a.run_id))
    assert "LLM_MOCK" not in mgr.env_for(a)  # không bật mock thì không đặt biến


def test_probe_failure_raises_and_starts_nothing(repo, monkeypatch):
    def boom(self, file_path, max_tokens):
        raise RunPlanError("probe failed: ValueError: không đọc được file")

    monkeypatch.setattr(RunManager, "plan", boom)
    made: list = []
    mgr = _manager(repo, made)

    with pytest.raises(RunPlanError):
        mgr.start(
            file="doc.pdf",
            src="vi",
            tgt="en",
            domain="tech",
            glossary=None,
            max_tokens=None,
            mock=False,
        )

    assert made == []
    # Chưa tạo gì cả: probe là bước đầu tiên, và nó đã thất bại.
    assert not (repo / "logs" / "webui").exists()
    assert not (repo / "output" / "webui").exists()


def test_input_file_outside_input_dir_is_rejected(repo, planned):
    mgr = _manager(repo)
    with pytest.raises(ValueError):
        mgr.start(
            file="../.env",
            src="vi",
            tgt="en",
            domain="tech",
            glossary=None,
            max_tokens=None,
            mock=False,
        )


def test_missing_input_file_is_a_plan_error(repo, planned):
    mgr = _manager(repo)
    with pytest.raises(RunPlanError):
        mgr.start(
            file="nope.pdf",
            src="vi",
            tgt="en",
            domain="tech",
            glossary=None,
            max_tokens=None,
            mock=False,
        )


def test_plan_asks_the_probe_and_reports_its_failure(repo, monkeypatch):
    """`plan` runs probe as a child process; a failure carries probe's stderr."""
    import subprocess

    calls: list = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom: bad file")

    monkeypatch.setattr("webui.runs.subprocess.run", fake_run)
    mgr = _manager(repo)

    with pytest.raises(RunPlanError) as err:
        mgr.plan(repo / "input" / "doc.pdf", None)

    argv = calls[0][0]
    assert argv[0] == "PY" and argv[1].endswith("probe.py") and argv[2] == "--plan"
    assert "boom: bad file" in str(err.value)


def test_plan_parses_probe_json(repo, monkeypatch):
    import json
    import subprocess

    payload = {"chunks": 7, "output_path": "x.en.md", "title": "x.md", "model": "m"}
    monkeypatch.setattr(
        "webui.runs.subprocess.run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(payload), stderr=""
        ),
    )
    assert _manager(repo).plan(repo / "input" / "doc.pdf", 100)["chunks"] == 7


# --- Task 6: status, stop, events -----------------------------------------


def test_status_reports_no_events_when_log_is_absent(repo, planned):
    """Review Focus 1: LOG_JSONL=0 ⇒ không có file log; phải nói ra, không đứng im."""
    mgr = _manager(repo)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )

    st = mgr.status(rec.run_id)
    assert mgr.has_events(rec.run_id) is False
    assert mgr.events(rec.run_id, 0) == ([], 0, 0)
    assert st.note  # có ghi chú giải thích vì sao không có sự kiện
    assert st.note == mgr.store.read_record(rec.run_id).note  # và nó được ghi xuống đĩa


def test_the_no_log_note_is_taken_back_once_the_log_appears(repo, planned):
    """The first poll of a healthy run beats the child's own log file.

    The note written then says the progress log is missing; the moment the file
    exists that sentence is false, and a run that is reporting progress
    normally must not keep carrying a warning about reporting none.
    """
    mgr = _manager(repo)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )

    assert mgr.status(rec.run_id).note  # chưa có log ⇒ có ghi chú

    (repo / "logs" / "webui" / rec.run_id / "run_20260101_000000.jsonl").write_text(
        '{"step": "plan", "status": "start"}\n', encoding="utf-8"
    )

    after = mgr.status(rec.run_id)
    assert after.note == ""
    assert mgr.store.read_record(rec.run_id).note == ""  # và nó được xoá trên đĩa


def test_a_legacy_no_log_note_is_cleaned_up_too(repo, planned):
    """Records written before this wording existed live on in `run.json`.

    The sentence they carry ("không có file sự kiện") is just as false once the
    run's own log is on disk, and leaving it would show a reader a warning that
    the step rail right above it visibly contradicts.
    """
    from webui.runs import RunRecord

    mgr = _manager(repo)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    (repo / "logs" / "webui" / rec.run_id / "run_20260101_000000.jsonl").write_text(
        '{"step": "plan", "status": "start"}\n', encoding="utf-8"
    )
    legacy = "lượt chạy không có file sự kiện (LOG_JSONL=0?): không có tiến độ để hiển thị"
    stored = mgr.store.read_record(rec.run_id)
    assert isinstance(stored, RunRecord)
    stored.note = f"đã dừng từ web UI (pid 12); {legacy}"
    mgr.store.write_record(stored)

    after = mgr.status(rec.run_id)
    assert "file sự kiện" not in after.note
    assert after.note == "đã dừng từ web UI (pid 12)"  # phần còn lại được giữ nguyên


def test_status_reconciles_a_dead_pid_after_a_server_restart(repo, planned, monkeypatch):
    mgr = _manager(repo)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    mgr.procs.clear()  # như thể server vừa khởi động lại
    monkeypatch.setattr("webui.runs.pid_alive", lambda pid: False)

    st = mgr.status(rec.run_id)
    assert st.status == "failed"
    assert "khởi động lại" in st.note.lower()


def test_status_folds_in_the_exit_code(repo, planned):
    mgr = _manager(repo)
    ok = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    bad = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    mgr.procs[ok.run_id].returncode = 0
    mgr.procs[bad.run_id].returncode = 1

    assert mgr.status(ok.run_id).status == "done"
    assert mgr.status(bad.run_id).status == "failed"
    assert mgr.status(bad.run_id).exit_code == 1
    assert ok.run_id not in mgr.procs  # đã chốt thì không giữ handle nữa


def test_status_of_an_unknown_run_raises(repo, planned):
    with pytest.raises(KeyError):
        _manager(repo).status("không-tồn-tại")


def test_events_are_read_through_the_store(repo, planned):
    mgr = _manager(repo)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    log = mgr.logs_dir(rec.run_id) / "run_20260101_000000.jsonl"
    log.write_text(
        '{"step": "ingest", "status": "ok"}\n{"step": "plan", "status": "ok"}\n', encoding="utf-8"
    )

    assert mgr.has_events(rec.run_id) is True
    assert [e["step"] for e in mgr.events(rec.run_id, 0)[0]] == ["ingest", "plan"]
    assert mgr.events(rec.run_id, 1) == ([{"step": "plan", "status": "ok"}], 2, 0)


def test_stop_marks_stopped_and_kills_the_tree(repo, planned, monkeypatch):
    killed = []
    monkeypatch.setattr("webui.runs.kill_tree", lambda pid: killed.append(pid))
    mgr = _manager(repo)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )

    st = mgr.stop(rec.run_id)
    assert killed == [4242]
    assert st.status == "stopped"
    assert st.exit_code is None
    assert mgr.store.read_record(rec.run_id).status == "stopped"
    assert rec.run_id not in mgr.procs

    with pytest.raises(RuntimeError):
        mgr.stop(rec.run_id)  # đã dừng rồi
    with pytest.raises(KeyError):
        mgr.stop("không-tồn-tại")


def test_delete_removes_the_runs_dirs_and_keeps_the_input(repo, planned, monkeypatch):
    """Erasing a run's history removes both its log dir and its output dir, but
    the input PDF a reader may translate again is left alone."""
    killed: list = []
    monkeypatch.setattr("webui.runs.kill_tree", lambda pid: killed.append(pid))
    mgr = _manager(repo)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        domain="tech",
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    # Give the run some real output to be erased.
    out = Path(rec.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# x", encoding="utf-8")
    assert (repo / "input" / "doc.pdf").is_file()

    mgr.delete(rec.run_id)

    assert not (repo / "logs" / "webui" / rec.run_id).exists()
    assert not (repo / "output" / "webui" / rec.run_id).exists()
    assert (repo / "input" / "doc.pdf").is_file()
    assert rec.run_id not in mgr.procs

    with pytest.raises(KeyError):
        mgr.delete(rec.run_id)  # đã xoá rồi
    with pytest.raises(KeyError):
        mgr.delete("không-tồn-tại")


def test_pid_alive_recognises_this_process_and_rejects_nonsense():
    """`pid_alive` is the only guard against signalling a recycled pid."""
    import os

    from webui.runs import pid_alive

    assert pid_alive(os.getpid()) is True
    assert pid_alive(-1) is False
    assert pid_alive(999_999_999) is False


def test_probe_command_uses_the_configured_interpreter(repo, monkeypatch):
    """The probe runs on our interpreter, and both ends speak UTF-8."""
    import subprocess
    import sys

    seen: list = []
    monkeypatch.setattr(
        "webui.runs.subprocess.run",
        lambda argv, **kw: (
            seen.append((argv, kw)),
            subprocess.CompletedProcess(argv, 0, stdout='{"chunks": 1}', stderr=""),
        )[1],
    )
    RunManager(repo, python=sys.executable).plan(repo / "input" / "doc.pdf", None)
    argv, kw = seen[0]
    assert argv[0] == sys.executable
    assert kw["encoding"] == "utf-8"
    assert kw["env"]["PYTHONIOENCODING"] == "utf-8"
    assert kw["cwd"] == repo
