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
    (tmp_path / "input" / "doc.md").write_text("# T\n\nxin chào\n", encoding="utf-8")
    monkeypatch.setattr("webui.runs.REPO_ROOT", tmp_path)
    return tmp_path


@pytest.fixture()
def planned(monkeypatch):
    """Replace the probe call so start-up logic is tested without a subprocess."""
    monkeypatch.setattr(
        RunManager,
        "plan",
        lambda self, file_path, max_tokens: {
            "chunks": 3, "output_path": "/out/x.en.md", "title": "doc.md",
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

    rec = mgr.start(file="doc.md", src="vi", tgt="en", domain="tech",
                    glossary="g.csv", max_tokens=555, mock=True)

    argv, kw = made[0]
    assert argv[:3] == ["PY", "-m", "polyglot_lc.cli"]
    assert argv[3] == "translate"
    for flag, value in (("--src", "vi"), ("--tgt", "en"), ("--domain", "tech"),
                        ("--glossary", "g.csv"), ("--max-tokens", "555")):
        assert argv[argv.index(flag) + 1] == value, flag
    assert kw["cwd"] == repo
    assert kw["env"]["LOGS_DIR"] == str(repo / "logs" / "webui" / rec.run_id)
    assert kw["env"]["LOG_JSONL"] == "1"
    assert kw["env"]["LLM_MOCK"] == "1"
    assert Path(rec.output_path).parent == repo / "output" / "webui" / rec.run_id
    assert (repo / "logs" / "webui" / rec.run_id / "run.json").is_file()
    assert rec.status == "running" and rec.chunks == 3 and rec.pid == 4242


def test_two_runs_keep_separate_log_dirs(repo, planned):
    mgr = _manager(repo)
    a = mgr.start(file="doc.md", src="vi", tgt="en", domain="tech",
                  glossary=None, max_tokens=None, mock=False)
    b = mgr.start(file="doc.md", src="vi", tgt="en", domain="tech",
                  glossary=None, max_tokens=None, mock=False)

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
        mgr.start(file="doc.md", src="vi", tgt="en", domain="tech",
                  glossary=None, max_tokens=None, mock=False)

    assert made == []
    # Chưa tạo gì cả: probe là bước đầu tiên, và nó đã thất bại.
    assert not (repo / "logs" / "webui").exists()
    assert not (repo / "output" / "webui").exists()


def test_input_file_outside_input_dir_is_rejected(repo, planned):
    mgr = _manager(repo)
    with pytest.raises(ValueError):
        mgr.start(file="../.env", src="vi", tgt="en", domain="tech",
                  glossary=None, max_tokens=None, mock=False)


def test_missing_input_file_is_a_plan_error(repo, planned):
    mgr = _manager(repo)
    with pytest.raises(RunPlanError):
        mgr.start(file="nope.md", src="vi", tgt="en", domain="tech",
                  glossary=None, max_tokens=None, mock=False)


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
        mgr.plan(repo / "input" / "doc.md", None)

    argv = calls[0][0]
    assert argv[0] == "PY" and argv[1].endswith("probe.py") and argv[2] == "--plan"
    assert "boom: bad file" in str(err.value)


def test_plan_parses_probe_json(repo, monkeypatch):
    import json
    import subprocess

    payload = {"chunks": 7, "output_path": "x.en.md", "title": "x.md", "model": "m"}
    monkeypatch.setattr(
        "webui.runs.subprocess.run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload),
                                                      stderr=""),
    )
    assert _manager(repo).plan(repo / "input" / "doc.md", 100)["chunks"] == 7


def test_probe_command_uses_the_configured_interpreter(repo, monkeypatch):
    import subprocess
    import sys

    seen: list = []
    monkeypatch.setattr(
        "webui.runs.subprocess.run",
        lambda argv, **kw: (seen.append(argv), subprocess.CompletedProcess(
            argv, 0, stdout='{"chunks": 1}', stderr=""))[1],
    )
    RunManager(repo, python=sys.executable).plan(repo / "input" / "doc.md", None)
    assert seen[0][0] == sys.executable
