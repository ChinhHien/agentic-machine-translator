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
        lambda self, file_path: {
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
        mock=True,
    )

    argv, kw = made[0]
    assert argv[:3] == ["PY", "-m", "polyglot_lc.cli"]
    assert argv[3] == "translate"
    for flag in ("--glossary", "--max-tokens"):
        assert flag not in argv, flag
    assert "--src" not in argv and "--tgt" not in argv
    assert kw["cwd"] == repo
    assert kw["env"]["LOGS_DIR"] == str(repo / "logs" / "webui" / rec.run_id)
    assert kw["env"]["LOG_JSONL"] == "1"
    assert kw["env"]["LLM_MOCK"] == "1"
    assert kw["env"]["PYTHONIOENCODING"] == "utf-8"  # log tiếng Việt không làm chết con
    assert Path(rec.output_path).parent == repo / "output" / "webui" / rec.run_id
    assert (repo / "logs" / "webui" / rec.run_id / "run.json").is_file()
    assert rec.status == "running" and rec.chunks == 3 and rec.pid == 4242


def test_plan_probes_with_the_dialog_endpoint_overrides(repo, monkeypatch):
    """The plan must count chunks for the configuration the child will run in.

    Without the dialog's overrides the probe counted with the model in `.env`
    (or its own fallback), and the page disagreed with the run about the number
    of chunks in the very same document.
    """
    seen: dict = {}

    def probe(self, args, extra_env=None):
        seen["args"] = args
        seen["env"] = extra_env or {}
        return {"chunks": 5}

    monkeypatch.setattr(RunManager, "probe", probe)
    monkeypatch.setattr(
        RunManager,
        "_settings_env",
        lambda self: ({"TRANSLATOR_MODEL": "override-model"}, ["TRANSLATOR_MODEL"]),
    )

    plan = _manager(repo).plan(repo / "input" / "doc.pdf")

    assert plan == {"chunks": 5}
    assert seen["args"][0] == "--plan" and seen["args"][1].endswith("doc.pdf")
    assert seen["env"]["TRANSLATOR_MODEL"] == "override-model"


def test_chunks_follow_the_run_start_not_the_plan(repo, planned):
    """The page showed 78 chunks while the child was working on 76."""
    mgr = _manager(repo)
    rec = mgr.start(file="doc.pdf", mock=False)
    assert rec.chunks == 3  # the plan's estimate, written before the child ran

    log = mgr.logs_dir(rec.run_id) / "run_20260101_000000.jsonl"
    log.write_text(json.dumps({"type": "run_start", "chunks": 5}) + "\n", encoding="utf-8")

    assert mgr.status(rec.run_id).chunks == 5
    assert mgr.store.read_record(rec.run_id).chunks == 5


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


def test_finish_resumes_the_same_run_and_asks_for_both_formats(repo, planned):
    """An old Markdown-only run gets one action that finishes it and merges its PDF.

    The command is the run's own, plus ``--resume`` and ``--format both``:
    checkpointed chunks are reused and the missing ones are the only work.
    """
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(
        file="doc.pdf",
        mock=False,
        output_format="md",
    )
    stopped = _stopped(mgr, rec)

    again = mgr.finish(stopped.run_id)

    argv, _kw = made[1]
    assert argv[-1] == "--resume"
    assert argv[argv.index("--format") + 1] == "both"
    assert again.run_id == stopped.run_id
    assert again.status == "running" and again.exit_code is None
    assert len(list((repo / "logs" / "webui").iterdir())) == 1


def test_unfinished_chunks_falls_back_to_the_checkpoint_count(repo, planned):
    """Old runs have no completion verdict; what is on disk still says how much is left."""
    mgr = _manager(repo)
    rec = mgr.start(file="doc.pdf", mock=False)
    mgr.procs[rec.run_id].returncode = 0
    rec.status = "done"
    mgr.store.write_record(rec)
    assert mgr.unfinished_chunks(rec.run_id) == rec.chunks  # no checkpoints yet
    directory = repo / "data" / "checkpoints" / "doc"
    directory.mkdir(parents=True)
    (directory / "chunk-000.json").write_text("{}", encoding="utf-8")
    mgr._checkpoint_dir = lambda run_id: directory
    assert mgr.unfinished_chunks(rec.run_id) == rec.chunks - 1


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
        mock=False,
    )
    stopped = _stopped(mgr, rec)
    stale = list(stopped.command) + [
        "--glossary",
        "g.csv",
        "--max-tokens",
        "555",
        "--legacy-flag",
        "general",
    ]
    mgr.store.write_record(replace(stopped, command=stale))

    mgr.resume(stopped.run_id)

    argv, _ = made[1]
    # Both dropped options are gone with the values they carried: the CLI does
    # not take them any more, so replaying them would stop the child cold.
    assert "--glossary" not in argv and "g.csv" not in argv
    assert "--max-tokens" not in argv and "555" not in argv
    assert "--legacy-flag" not in argv
    assert "general" not in argv  # the value went with the option
    assert argv[-1] == "--resume"


def test_resume_refuses_a_run_that_is_still_running(repo, planned):
    """A run that never stopped keeps its own child: a second one would write
    the same event log and the same output file at the same time."""
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(
        file="doc.pdf",
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
        mock=False,
    )
    stderr = repo / "logs" / "webui" / rec.run_id / "stderr.log"
    stderr.write_text("provider 500\n", encoding="utf-8")
    stopped = _stopped(mgr, rec)

    mgr.resume(stopped.run_id)

    assert stderr.read_text(encoding="utf-8") == "provider 500\n"


def test_resume_pins_the_model_the_run_itself_reported(repo, planned, monkeypatch):
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
        mock=False,
    )
    (repo / "logs" / "webui" / rec.run_id / "run_20260101_000000.jsonl").write_text(
        json.dumps({"type": "run_start", "agents": [{"name": "translator", "model": "gpt-4o"}]})
        + "\n",
        encoding="utf-8",
    )
    stopped = _stopped(mgr, rec)
    monkeypatch.setattr(
        RunManager,
        "_endpoint_check",
        lambda self, model: {
            "model": "gpt-4o",
            "host": "",
            "test": {"ok": True},
        },
    )

    mgr.resume(stopped.run_id)

    assert made[1][1]["env"]["TRANSLATOR_MODEL"] == "gpt-4o"

    # A fresh run never pins: it should track whatever config is in effect now.
    fresh = mgr.start(
        file="doc.pdf",
        mock=False,
    )
    assert fresh.translator_model == ""
    assert "TRANSLATOR_MODEL" not in made[2][1]["env"]


def test_a_resume_pins_the_endpoint_the_run_used(repo, planned, monkeypatch):
    """The model is only half the checkpoint identity; the URL is the other half.

    The settings dialog's endpoint reaches the child *before* the run's own
    values, so pinning the model alone kept the old name on a new server: the
    checkpoint key changed and every finished chunk had to be paid for again.
    """
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(file="doc.pdf", mock=False)
    (repo / "logs" / "webui" / rec.run_id / "run_20260101_000000.jsonl").write_text(
        json.dumps(
            {
                "type": "run_start",
                "agents": [
                    {
                        "name": "translator",
                        "model": "gpt-4o",
                        "endpoint": "http://old-host:4000/v1",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stopped = _stopped(mgr, rec)
    monkeypatch.setattr(
        RunManager,
        "_endpoint_check",
        lambda self, model: {"model": "gpt-4o", "test": {"ok": True}},
    )
    monkeypatch.setattr(
        RunManager,
        "_settings_env",
        lambda self: (
            {"TRANSLATOR_BASE_URL": "http://new-host:4000/v1", "TRANSLATOR_MODEL": "new"},
            [],
        ),
    )

    again = mgr.resume(stopped.run_id)

    env = made[1][1]["env"]
    assert env["TRANSLATOR_MODEL"] == "gpt-4o"
    assert env["TRANSLATOR_BASE_URL"] == "http://old-host:4000/v1"
    assert again.translator_base_url == "http://old-host:4000/v1"


def test_resume_drops_a_model_the_provider_no_longer_serves(repo, planned, monkeypatch):
    """One button is enough: a stale pin is checked and replaced by today's config."""
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(file="doc.pdf", mock=False)
    (repo / "logs" / "webui" / rec.run_id / "run_20260101_000000.jsonl").write_text(
        json.dumps(
            {
                "type": "run_start",
                "agents": [{"name": "translator", "model": "old-model", "host": "api.old"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stopped = _stopped(mgr, rec)
    monkeypatch.setattr(
        RunManager,
        "_endpoint_check",
        lambda self, model: {
            "model": "new-model",
            "host": "api.old",
            "test": {"ok": False, "error": "HTTP 404"},
        },
    )

    again = mgr.resume(stopped.run_id)

    assert "TRANSLATOR_MODEL" not in made[1][1]["env"]
    assert "cấu hình hiện tại" in again.note
    assert "--resume" in again.command  # same run, same log and output


def test_resume_archives_the_old_translation_before_switching_models(
    repo, planned, monkeypatch
):
    """A model fallback re-translates, but must not destroy the old deliverable."""
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(file="doc.pdf", mock=False)
    (repo / "logs" / "webui" / rec.run_id / "run_20260101_000000.jsonl").write_text(
        json.dumps(
            {
                "type": "run_start",
                "agents": [{"name": "translator", "model": "old-model", "host": "api.old"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stopped = _stopped(mgr, rec)
    markdown = Path(stopped.output_path)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("old md", encoding="utf-8")
    pdf = markdown.with_suffix(".pdf")
    pdf.write_bytes(b"%PDF-old")
    monkeypatch.setattr(
        RunManager,
        "_endpoint_check",
        lambda self, model: {
            "model": "new-model",
            "host": "api.old",
            "test": {"ok": False, "error": "HTTP 404"},
        },
    )

    again = mgr.resume(stopped.run_id)

    assert not markdown.exists() and not pdf.exists()
    archives = list((markdown.parent / "archive").iterdir())
    assert len(archives) == 1
    assert (archives[0] / markdown.name).read_text(encoding="utf-8") == "old md"
    assert (archives[0] / pdf.name).read_bytes() == b"%PDF-old"
    assert "bản cũ đã lưu" in again.note


def test_resume_keeps_the_pin_when_only_the_endpoint_host_changed(
    repo, planned, monkeypatch
):
    """A moved endpoint alone is not proof that the old model is gone."""
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(file="doc.pdf", mock=False)
    (repo / "logs" / "webui" / rec.run_id / "run_20260101_000000.jsonl").write_text(
        json.dumps(
            {
                "type": "run_start",
                "agents": [{"name": "translator", "model": "old-model", "host": "api.old"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stopped = _stopped(mgr, rec)
    monkeypatch.setattr(
        RunManager,
        "_endpoint_check",
        lambda self, model: {
            "model": "new-model",
            "host": "api.new",
            "test": {"ok": True},
        },
    )

    mgr.resume(stopped.run_id)

    assert made[1][1]["env"]["TRANSLATOR_MODEL"] == "old-model"


def test_only_404_or_model_not_found_marks_the_model_missing(repo, monkeypatch):
    """Ambiguous failures must never trigger a paid retranslation."""
    mgr = _manager(repo)
    cases = [
        ({"ok": False, "status": 404, "error": ""}, True),
        ({"ok": False, "status": 400, "error": "model_not_found"}, True),
        ({"ok": False, "status": 500, "error": "HTTP 500: upstream"}, False),
        ({"ok": False, "status": 0, "error": "connection reset"}, False),
        ({"ok": False, "status": 429, "error": "rate limited"}, False),
    ]
    for test, expected in cases:
        monkeypatch.setattr(
            RunManager,
            "_endpoint_check",
            lambda self, model, test=test: {"model": "new-model", "test": test},
        )
        assert mgr._model_missing("old-model") is expected, test


def test_resume_keeps_the_pin_on_a_timeout(repo, planned, monkeypatch):
    """A timeout says nothing about the model; do not pay to retranslate."""
    made: list = []
    mgr = _manager(repo, made)
    rec = mgr.start(file="doc.pdf", mock=False)
    (repo / "logs" / "webui" / rec.run_id / "run_20260101_000000.jsonl").write_text(
        json.dumps(
            {
                "type": "run_start",
                "agents": [{"name": "translator", "model": "old-model", "host": "api.old"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stopped = _stopped(mgr, rec)
    monkeypatch.setattr(
        RunManager,
        "_endpoint_check",
        lambda self, model: {
            "model": "new-model",
            "test": {"ok": False, "status": 0, "error": "timed out"},
        },
    )

    again = mgr.resume(stopped.run_id)

    assert made[1][1]["env"]["TRANSLATOR_MODEL"] == "old-model"
    assert "cấu hình hiện tại" not in again.note


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
        mock=False,
    )
    b = mgr.start(
        file="doc.pdf",
        mock=False,
    )

    assert a.run_id != b.run_id
    assert mgr.logs_dir(a.run_id) != mgr.logs_dir(b.run_id)
    assert mgr.env_for(a)["LOGS_DIR"] == str(mgr.logs_dir(a.run_id))
    assert "LLM_MOCK" not in mgr.env_for(a)  # không bật mock thì không đặt biến


def test_probe_failure_raises_and_starts_nothing(repo, monkeypatch):
    def boom(self, file_path):
        raise RunPlanError("probe failed: ValueError: không đọc được file")

    monkeypatch.setattr(RunManager, "plan", boom)
    made: list = []
    mgr = _manager(repo, made)

    with pytest.raises(RunPlanError):
        mgr.start(
            file="doc.pdf",
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
            mock=False,
        )


def test_missing_input_file_is_a_plan_error(repo, planned):
    mgr = _manager(repo)
    with pytest.raises(RunPlanError):
        mgr.start(
            file="nope.pdf",
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
        mgr.plan(repo / "input" / "doc.pdf")

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
    assert _manager(repo).plan(repo / "input" / "doc.pdf")["chunks"] == 7


# --- Task 6: status, stop, events -----------------------------------------


def test_status_reports_no_events_when_log_is_absent(repo, planned):
    """Review Focus 1: LOG_JSONL=0 ⇒ không có file log; phải nói ra, không đứng im."""
    mgr = _manager(repo)
    rec = mgr.start(
        file="doc.pdf",
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
        mock=False,
    )
    mgr.procs.clear()  # như thể server vừa khởi động lại
    monkeypatch.setattr("webui.runs.pid_alive", lambda pid: False)

    st = mgr.status(rec.run_id)
    assert st.status == "failed"
    assert "khởi động lại" in st.note.lower()


def test_status_uses_run_end_when_the_process_handle_was_already_reaped(
    repo, planned, monkeypatch
):
    """A finished rebuild must not be called failed in the sweep/poll race."""
    mgr = _manager(repo)
    rec = mgr.start(
        file="doc.pdf",
        mock=False,
    )
    (mgr.logs_dir(rec.run_id) / "run_20260101_000000.jsonl").write_text(
        json.dumps({"type": "run_end", "status": "done", "exit_code": 0}) + "\n",
        encoding="utf-8",
    )
    mgr.procs.clear()  # another thread reaped the child just before this poll
    monkeypatch.setattr("webui.runs.pid_alive", lambda pid: False)

    st = mgr.status(rec.run_id)
    assert st.status == "done"
    assert st.exit_code == 0
    assert "khởi động lại" not in st.note.lower()


def test_status_folds_in_the_exit_code(repo, planned):
    mgr = _manager(repo)
    ok = mgr.start(
        file="doc.pdf",
        mock=False,
    )
    bad = mgr.start(
        file="doc.pdf",
        mock=False,
    )
    partial = mgr.start(
        file="doc.pdf",
        mock=False,
    )
    mgr.procs[ok.run_id].returncode = 0
    mgr.procs[bad.run_id].returncode = 1
    mgr.procs[partial.run_id].returncode = 2

    assert mgr.status(ok.run_id).status == "done"
    assert mgr.status(bad.run_id).status == "failed"
    assert mgr.status(bad.run_id).exit_code == 1
    # A written file with missing checkpoints is its own state, not a crash.
    assert mgr.status(partial.run_id).status == "incomplete"
    assert mgr.status(partial.run_id).exit_code == 2
    assert ok.run_id not in mgr.procs  # đã chốt thì không giữ handle nữa


def test_status_of_an_unknown_run_raises(repo, planned):
    with pytest.raises(KeyError):
        _manager(repo).status("không-tồn-tại")


def test_events_are_read_through_the_store(repo, planned):
    mgr = _manager(repo)
    rec = mgr.start(
        file="doc.pdf",
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
        mock=False,
    )
    second = mgr.start(
        file="doc.pdf",
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
    RunManager(repo, python=sys.executable).plan(repo / "input" / "doc.pdf")
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


@pytest.mark.parametrize(
    ("chunks_rejected", "fatal_error", "retryable"),
    [
        pytest.param(1, "", False, id="quality-only"),
        pytest.param(1, "TransientError: service unavailable", True, id="fatal-transient"),
        pytest.param(0, "ZeroDivisionError: division by zero", False, id="fatal-bug"),
        pytest.param(1, "ZeroDivisionError: division by zero", False, id="fatal-bug-and-quality"),
        pytest.param(0, "PermanentError: timeout on invalid request", False, id="fatal-permanent"),
        pytest.param(0, "", True, id="unclassified-fallback"),
    ],
)
def test_sweep_uses_current_run_end_failure(repo, planned, chunks_rejected, fatal_error, retryable):
    """Recovered Planner timeouts must not override the terminal failure."""
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    log = mgr.logs_dir(rec.run_id) / "run_20260101_000000.jsonl"
    log.write_text(
        json.dumps({"step": "plan", "status": "retry", "error": "Planner timeout"})
        + "\n"
        + json.dumps(
            {
                "type": "run_end",
                "status": "failed",
                "chunks_failed": 1,
                "chunks_rejected": chunks_rejected,
                "fatal_error": fatal_error,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _died(mgr, rec, "Planner timeout (recovered)\nRun failed: document incomplete\n")

    scheduled = mgr.sweep(now=1000.0)

    stored = mgr.store.read_record(rec.run_id)
    assert stored.status == "failed" and stored.exit_code == 1
    assert mgr._missing_chunks(stored) == 1
    if retryable:
        assert [r.run_id for r in scheduled] == [rec.run_id]
        assert stored.auto_resumes == 1
        assert stored.retry_at == 1000.0 + AUTO_RESUME_DELAYS[0]
        assert len(made) == 1
        continued = mgr.sweep(now=stored.retry_at)
        assert [r.run_id for r in continued] == [rec.run_id]
        assert continued[0].status == "running"
        assert made[1][0] == rec.command + ["--resume"]
    else:
        assert scheduled == []
        assert stored.auto_resumes == 0 and stored.retry_at == 0.0
        assert mgr.sweep(now=1_000_000.0) == []
        assert len(made) == 1


def test_manual_resume_of_quality_rejected_run_keeps_identity_and_logs(repo, planned):
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    log = mgr.logs_dir(rec.run_id) / "run_20260101_000000.jsonl"
    events = (
        json.dumps(
            {
                "type": "run_end",
                "status": "failed",
                "chunks_failed": 1,
                "chunks_rejected": 1,
                "fatal_error": "",
            }
        )
        + "\n"
    )
    log.write_text(events, encoding="utf-8")
    _died(mgr, rec, "Run failed: chunk quality rejected\n")

    again = mgr.resume(rec.run_id)

    assert again.run_id == rec.run_id
    assert again.started == rec.started and again.output_path == rec.output_path
    assert again.status == "running" and again.exit_code is None
    assert again.auto_resumes == 0 and again.retry_at == 0.0
    assert made[1][0] == rec.command + ["--resume"]
    assert made[1][1]["env"]["LOGS_DIR"] == str(mgr.logs_dir(rec.run_id))
    assert log.read_text(encoding="utf-8") == events


@pytest.mark.parametrize(
    ("current_end", "exit_code", "retryable"),
    [
        pytest.param(
            {"status": "done", "chunks_failed": 0, "chunks_rejected": 0, "fatal_error": ""},
            0,
            False,
            id="clean-attempt",
        ),
        pytest.param(
            {
                "status": "failed",
                "chunks_failed": 1,
                "chunks_rejected": 0,
                "fatal_error": "TransientError: connection reset",
            },
            1,
            True,
            id="current-fatal-attempt",
        ),
        pytest.param({"status": "failed", "chunks_failed": 1}, 1, True, id="legacy-attempt"),
        pytest.param(None, 1, True, id="current-attempt-without-run-end"),
        pytest.param(None, None, False, id="current-attempt-still-running"),
    ],
)
def test_sweep_ignores_quality_counts_from_previous_attempts(
    repo, planned, current_end, exit_code, retryable
):
    made: list = []
    mgr = _manager(repo, made)
    rec = _run(mgr)
    logs = mgr.logs_dir(rec.run_id)
    (logs / "run_20260101_000000.jsonl").write_text(
        json.dumps(
            {
                "type": "run_end",
                "status": "failed",
                "chunks_failed": 1,
                "chunks_rejected": 1,
                "fatal_error": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events = [{"type": "run_start"}]
    if current_end is not None:
        events.append({"type": "run_end", **current_end})
    (logs / "run_20260101_000001.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (logs / "stderr.log").write_text("TransientError: connection reset\n", encoding="utf-8")
    mgr.procs[rec.run_id].returncode = exit_code

    scheduled = mgr.sweep(now=1000.0)

    assert [r.run_id for r in scheduled] == ([rec.run_id] if retryable else [])
    stored = mgr.store.read_record(rec.run_id)
    assert stored.auto_resumes == int(retryable)
    assert stored.status == (
        "running" if exit_code is None else "done" if exit_code == 0 else "failed"
    )
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


def test_stop_answers_without_waiting_for_the_child_to_exit(repo, planned, monkeypatch):
    """"Dừng" must not leave the page blocked while a killed child finishes dying."""
    import threading
    import time

    monkeypatch.setattr("webui.runs.kill_tree", lambda pid: None)
    mgr = _manager(repo)
    rec = mgr.start(file="doc.pdf", mock=False)

    released = threading.Event()

    class SlowExit(FakeProc):
        def wait(self, timeout=None):
            released.wait(timeout)
            self.returncode = -15
            return self.returncode

    mgr.procs[rec.run_id] = SlowExit(rec.command)

    started = time.perf_counter()
    stopped = mgr.stop(rec.run_id)
    elapsed = time.perf_counter() - started
    released.set()

    assert stopped.status == "stopped"
    assert mgr.store.read_record(rec.run_id).status == "stopped"
    assert elapsed < 1.0, f"stop waited {elapsed:.1f}s for the child to exit"
