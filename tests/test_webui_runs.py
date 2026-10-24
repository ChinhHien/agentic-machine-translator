"""Tests for webui.runs: launching and supervising one CLI run per document."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from webui.runs import AUTO_RESUME_DELAYS, RunManager, RunPlanError


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


def _stopped(mgr, rec):
    """Make *rec* look like a run whose last attempt is over.

    Its child has exited and the manager has reconciled the record on the next
    look (what the page's poll does), and the reader pressed "Dừng".
    """
    mgr.procs[rec.run_id].returncode = 1
    stopped = mgr.status(rec.run_id)
    stopped.status = "stopped"
    mgr.store.write_record(stopped)
    return stopped


def test_start_builds_the_command_and_the_run_dir(repo, planned):
    made: list = []
    mgr = _manager(repo, made)

    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
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


def test_resume_continues_the_run_in_place_with_its_own_command(repo, planned):
    """ "Tiếp tục" continues the run the reader is watching, not a new one.

    The chunks left to translate are the ones the stopped attempt wrote
    checkpoints for, and those checkpoints are keyed by the command that
    produced them — file, languages, glossary, token budget — plus the
    Translator model. So the resumed attempt is that exact command with
    ``--resume``, under the same run id, log directory and output path: the
    page keeps showing one run, in the place it already was.
    """
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        glossary="g.csv",
        max_tokens=555,
        mock=False,
    )
    original = list(rec.command)
    stopped = _stopped(mgr, rec)

    again = mgr.resume(stopped.run_id)

    argv, kw = made[1]
    assert argv == original + ["--resume"]  # nothing rebuilt from today's settings
    assert again.run_id == stopped.run_id
    assert again.started == stopped.started  # the same session, not a second one
    assert again.output_path == stopped.output_path
    assert again.status == "running" and again.exit_code is None and again.note == ""
    assert len(list((repo / "logs" / "webui").iterdir())) == 1  # no second run dir
    assert kw["env"]["LOGS_DIR"] == str(repo / "logs" / "webui" / stopped.run_id)


def test_resume_drops_an_option_the_cli_no_longer_takes(repo, planned):
    """A stored command is replayed without the options the CLI has dropped.

    A record keeps the command line of the day the run started, and the CLI
    has changed since: an option it no longer has would stop the child before
    it started, so the run could never continue. The option goes, and the
    value it carried goes with it; everything else stays, so the run still
    lands on the very same checkpoints.
    """
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        glossary="g.csv",
        max_tokens=555,
        mock=False,
    )
    stopped = _stopped(mgr, rec)
    stale = list(stopped.command) + ["--legacy-flag", "general"]
    mgr.store.write_record(replace(stopped, command=stale))

    mgr.resume(stopped.run_id)

    argv, _ = made[1]
    assert "--legacy-flag" not in argv
    assert "general" not in argv  # the value went with the option
    assert "--glossary" in argv and "g.csv" in argv  # the rest is untouched
    assert "--max-tokens" in argv and "555" in argv
    assert argv[-1] == "--resume"


def test_resume_refuses_a_run_that_is_still_running(repo, planned):
    """A run that never stopped keeps its own child: a second one would write
    the same event log and the same output file at the same time."""
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        glossary=None,
        max_tokens=None,
        mock=False,
    )

    with pytest.raises(RuntimeError):
        mgr.resume(rec.run_id)
    assert len(made) == 1  # nothing was spawned


def test_resume_of_an_unknown_run_is_a_key_error(repo, planned):
    with pytest.raises(KeyError):
        _manager(repo).resume("không-tồn-tại")


def test_resume_keeps_the_previous_attempt_s_console_output(repo, planned):
    """Append, not truncate: a reader asking "why did it stop?" still finds the
    previous attempt's own words beside the continued attempt's."""
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    stderr = repo / "logs" / "webui" / rec.run_id / "stderr.log"
    stderr.write_text("provider 500\n", encoding="utf-8")
    stopped = _stopped(mgr, rec)

    mgr.resume(stopped.run_id)

    assert stderr.read_text(encoding="utf-8") == "provider 500\n"


def test_resume_pins_the_model_the_run_itself_reported(repo, planned):
    """ "Tiếp tục" must keep the checkpoint key of the run it comes from.

    Checkpoints are keyed by the translator model (and the tokenizer it picks),
    so a resumed attempt that fell back to the server's current config would
    derive a different key and silently re-translate every chunk instead of
    reusing the ones the stopped attempt already wrote. The model comes from
    the run's own `run_start` record — the page sends no settings at all.
    """
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    (repo / "logs" / "webui" / rec.run_id / "run_20260101_000000.jsonl").write_text(
        json.dumps({"type": "run_start", "agents": [{"name": "translator", "model": "gpt-4o"}]})
        + "\n",
        encoding="utf-8",
    )
    stopped = _stopped(mgr, rec)

    mgr.resume(stopped.run_id)

    assert made[1][1]["env"]["TRANSLATOR_MODEL"] == "gpt-4o"

    # A fresh run never pins: it should track whatever config is in effect now.
    fresh = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    assert fresh.translator_model == ""
    assert "TRANSLATOR_MODEL" not in made[2][1]["env"]


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
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    b = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
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
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    bad = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
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


def test_delete_removes_the_runs_dirs_and_the_input_it_was_last_at(repo, planned, monkeypatch):
    """Erasing a run's history removes its log dir, its output dir, and the
    input PDF it came from — but only when no other run still points at that
    document: two runs may share one file, and deleting it under the other
    would break its resume."""
    killed: list = []
    monkeypatch.setattr("webui.runs.kill_tree", lambda pid: killed.append(pid))
    mgr = _manager(repo)
    rec = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
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
    assert not (repo / "input" / "doc.pdf").exists()
    assert rec.run_id not in mgr.procs

    with pytest.raises(KeyError):
        mgr.delete(rec.run_id)  # đã xoá rồi
    with pytest.raises(KeyError):
        mgr.delete("không-tồn-tại")


def test_delete_keeps_the_input_while_another_run_still_uses_it(repo, planned, monkeypatch):
    """Two runs translating the same document: deleting the first must not
    take the file out from under the second, whose resume still needs it."""
    killed: list = []
    monkeypatch.setattr("webui.runs.kill_tree", lambda pid: killed.append(pid))
    mgr = _manager(repo)
    first = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        glossary=None,
        max_tokens=None,
        mock=False,
    )
    second = mgr.start(
        file="doc.pdf",
        src="vi",
        tgt="en",
        glossary=None,
        max_tokens=None,
        mock=False,
    )

    mgr.delete(first.run_id)

    assert (repo / "input" / "doc.pdf").is_file()

    mgr.delete(second.run_id)

    assert not (repo / "input" / "doc.pdf").exists()


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


# --- Tự động tiếp tục khi nhà cung cấp lỗi tạm thời -------------------------


def _run(mgr, **over):
    """Start one run with the choices every test in this section uses."""
    fields = {
        "file": "doc.pdf",
        "src": "vi",
        "tgt": "en",
        "glossary": None,
        "max_tokens": None,
        "mock": False,
    }
    fields.update(over)
    return mgr.start(**fields)


def _died(mgr, rec, tail="Run failed: RateLimitError: 429 rate limit reached\n"):
    """Make *rec* look like an attempt whose child has just died saying *tail*."""
    mgr.procs[rec.run_id].returncode = 1
    (mgr.logs_dir(rec.run_id) / "stderr.log").write_text(tail, encoding="utf-8")
    return mgr.status(rec.run_id)


def test_sweep_schedules_a_wait_instead_of_relaunching_at_once(repo, planned):
    """A provider that refused everything needs time to clear.

    The pass spends an attempt and writes down when the next one is due; it must
    not relaunch inside the same outage, or all three attempts burn in one
    second and the run is dead anyway.
    """
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    failed = _died(mgr, rec)

    first = mgr.sweep(now=1000.0)

    assert [r.run_id for r in first] == [failed.run_id]
    stored = mgr.store.read_record(failed.run_id)
    assert stored.auto_resumes == 1
    assert stored.retry_at == 1000.0 + AUTO_RESUME_DELAYS[0]
    assert "lần 1/3" in stored.note and "30 giây" in stored.note
    assert len(made) == 1  # chưa relaunch: chờ là chủ ý

    # Chưa tới hạn thì vòng quét để yên.
    assert mgr.sweep(now=1029.0) == []
    assert len(made) == 1


def test_sweep_continues_the_run_itself_when_the_wait_is_over(repo, planned):
    """ "Tiếp tục" được bấm hộ: cùng run_id, cùng thư mục log, không sinh lượt mới."""
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    original = list(rec.command)
    _died(mgr, rec)
    mgr.sweep(now=1000.0)

    due = mgr.sweep(now=1000.0 + AUTO_RESUME_DELAYS[0])

    argv, kw = made[1]
    assert [r.run_id for r in due] == [rec.run_id]
    assert argv == original + ["--resume"]
    assert kw["env"]["LOGS_DIR"] == str(repo / "logs" / "webui" / rec.run_id)
    assert due[0].status == "running" and due[0].exit_code is None
    assert due[0].retry_at == 0.0
    assert due[0].note.startswith("đang tự tiếp tục lần 1/3")
    assert len(list((repo / "logs" / "webui").iterdir())) == 1


def test_each_further_attempt_waits_longer(repo, planned):
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    _died(mgr, rec)

    mgr.sweep(now=0.0)  # hẹn lần 1
    mgr.sweep(now=AUTO_RESUME_DELAYS[0])  # chạy lần 1
    mgr.procs[rec.run_id].returncode = 1  # và nó chết lại
    mgr.status(rec.run_id)

    mgr.sweep(now=100.0)

    stored = mgr.store.read_record(rec.run_id)
    assert stored.auto_resumes == 2
    assert stored.retry_at == 100.0 + AUTO_RESUME_DELAYS[1]
    assert "lần 2/3" in stored.note and "2 phút" in stored.note


def test_sweep_gives_up_once_the_budget_is_spent(repo, planned):
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    failed = _died(mgr, rec)
    spent = mgr.store.read_record(failed.run_id)
    spent.auto_resumes = len(AUTO_RESUME_DELAYS)
    mgr.store.write_record(spent)

    assert mgr.sweep(now=1000.0) == []
    assert len(made) == 1
    assert mgr.store.read_record(rec.run_id).retry_at == 0.0


def test_sweep_leaves_a_refused_request_alone(repo, planned):
    """A bad key or a bad request cannot be fixed by waiting; retrying it would
    spend ten minutes and bury the real reason under three identical failures."""
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    _died(mgr, rec, "Run failed: PermanentError: Error code: 401 - Incorrect API key\n")

    assert mgr.sweep(now=1000.0) == []
    assert len(made) == 1


def test_sweep_does_not_retry_a_failure_it_cannot_read(repo, planned):
    """A bug is not an outage: three attempts would only repeat the crash."""
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    _died(mgr, rec, "Run failed: ZeroDivisionError: division by zero\n")

    assert mgr.sweep(now=1000.0) == []
    assert len(made) == 1


def test_sweep_never_touches_a_run_the_reader_stopped(repo, planned):
    """ "Dừng" is a decision, and it outranks a retryable error."""
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    stopped = _stopped(mgr, rec)
    (mgr.logs_dir(rec.run_id) / "stderr.log").write_text(
        "Run failed: RateLimitError: 429\n", encoding="utf-8"
    )

    assert mgr.sweep(now=1_000_000.0) == []
    assert len(made) == 1
    assert mgr.store.read_record(stopped.run_id).auto_resumes == 0


def test_sweep_continues_a_finished_run_that_lost_a_chunk(repo, planned):
    """A chunk dropped by an exhausted rate limit does not kill the run.

    The process still exits 0 and the page says "xong", so nothing but the run's
    own ``run_end`` record says the work is missing — and that is the failure a
    reader most often has to press "Tiếp tục" for.
    """
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    log = mgr.logs_dir(rec.run_id) / "run_20260101_000000.jsonl"
    log.write_text(
        json.dumps(
            {
                "step": "translate",
                "status": "fail",
                "chunk_id": "c3",
                "error": "c3: RateLimitError: 429 rate limit exceeded",
            }
        )
        + "\n"
        + json.dumps({"type": "run_end", "status": "failed", "chunks_failed": 1})
        + "\n",
        encoding="utf-8",
    )
    mgr.procs[rec.run_id].returncode = 0
    done = mgr.status(rec.run_id)
    assert done.status == "done"  # mã thoát 0: trang coi như đã xong

    assert [r.run_id for r in mgr.sweep(now=500.0)] == [rec.run_id]
    assert mgr.store.read_record(rec.run_id).auto_resumes == 1


def test_sweep_leaves_a_run_that_finished_cleanly_alone(repo, planned):
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    (mgr.logs_dir(rec.run_id) / "run_20260101_000000.jsonl").write_text(
        json.dumps({"type": "run_end", "status": "done", "chunks_failed": 0}) + "\n",
        encoding="utf-8",
    )
    mgr.procs[rec.run_id].returncode = 0
    mgr.status(rec.run_id)

    assert mgr.sweep(now=500.0) == []


def test_sweep_notices_a_child_that_died_with_no_page_open(repo, planned, monkeypatch):
    """The record still says "running" over a dead pid after a server restart.

    The sweep is often the first thing to look, and it must not need a browser to
    notice — that would make the feature work only for a reader already watching.
    """
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    (mgr.logs_dir(rec.run_id) / "stderr.log").write_text(
        "Run failed: TransientError: connection reset by peer\n", encoding="utf-8"
    )
    mgr.procs.clear()  # như thể server vừa khởi động lại
    monkeypatch.setattr("webui.runs.pid_alive", lambda pid: False)

    out = mgr.sweep(now=1000.0)

    assert [r.run_id for r in out] == [rec.run_id]
    assert mgr.store.read_record(rec.run_id).auto_resumes == 1


def test_a_manual_resume_cancels_the_scheduled_attempt(repo, planned):
    """The reader pressing "Tiếp tục" takes over: the pending wait means nothing
    then, and the attempt it had spent is not handed back."""
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    _died(mgr, rec)
    mgr.sweep(now=1000.0)

    again = mgr.resume(rec.run_id)

    assert again.retry_at == 0.0
    assert again.auto_resumes == 1
    assert again.note == ""
    assert mgr.sweep(now=1_000_000.0) == []


def test_stop_calls_off_a_waiting_retry(repo, planned):
    """Between two attempts there is no child, so "Dừng" means "đừng thử lại nữa".

    Without this the reader watches "thử lại sau 8 phút" with no way to say no,
    which is exactly the feeling a supervisor acting on its own must not create.
    """
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    _died(mgr, rec)
    mgr.sweep(now=1000.0)  # hẹn lần 1

    stopped = mgr.stop(rec.run_id)

    assert stopped.retry_at == 0.0
    assert stopped.auto_resumes == len(AUTO_RESUME_DELAYS)
    assert "huỷ tự động tiếp tục" in stopped.note
    assert stopped.status == "failed"  # vẫn là lượt đã lỗi, không giả vờ là đã dừng
    assert mgr.sweep(now=1_000_000.0) == []
    assert len(made) == 1


def test_stop_still_refuses_a_run_with_nothing_to_stop(repo, planned):
    """A finished run with no retry pending has no attempt to call off either."""
    mgr = _manager(repo)
    rec = _run(mgr)
    mgr.procs[rec.run_id].returncode = 0
    mgr.status(rec.run_id)

    with pytest.raises(RuntimeError):
        mgr.stop(rec.run_id)


def test_the_sweeper_keeps_going_and_survives_a_bad_pass(repo, planned):
    """A supervisor that dies quietly looks like a run that never comes back."""
    import threading
    import time as clock

    from webui.runs import start_sweeper

    calls: list = []

    def sweep(now=None):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("store went away")
        return []

    mgr = _manager(repo)
    mgr.sweep = sweep
    stop = threading.Event()
    thread = start_sweeper(mgr, interval=0.01, stop=stop)

    deadline = clock.time() + 5
    while len(calls) < 3 and clock.time() < deadline:
        clock.sleep(0.01)
    stop.set()
    thread.join(timeout=5)

    assert len(calls) >= 3  # một lần hỏng không kết thúc vòng lặp
    assert not thread.is_alive()
