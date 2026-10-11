"""Run supervision: one child `polyglot_lc.cli` process per document.

The web UI never imports the pipeline: it launches the CLI the same way a user
would, so a run cannot wedge the server and the two front ends cannot drift.
Plan-then-start: :meth:`RunManager.plan` probes the document first, and only
when it succeeds does :meth:`RunManager.start` create a run directory, spawn the
child and persist the record.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from webui.paths import REPO_ROOT, safe_child
from webui.store import RunRecord, RunsStore

#: Where uploaded/translator inputs live, relative to the repository root.
INPUT_DIR = "input"

#: Per-run directories for artefacts and for metadata + logs, under the repo.
OUTPUT_DIR = Path("output") / "webui"
LOGS_DIR = Path("logs") / "webui"#: The probe script, resolved from this file so a caller-provided root cannot
#: redirect it: the script ships with the package, the data does not.
_PROBE = Path(__file__).with_name("probe.py")

#: Exit code that means the run finished its work.
EXIT_OK = 0

#: How long a stopped run may take to die before we escalate once.
STOP_TIMEOUT = 5.0

#: Explained in the record when there is no event log to read.
NO_LOG_NOTE = (
    "lượt chạy không có file sự kiện (LOG_JSONL=0?): không có tiến độ để hiển thị"
)


class RunPlanError(RuntimeError):
    """The document could not be planned, so no run was started."""


def _note(record: RunRecord, text: str) -> None:
    """Append an explanation, without repeating one a previous poll wrote."""
    if text and text not in record.note:
        record.note = f"{record.note}; {text}" if record.note else text


def pid_alive(pid: int) -> bool:
    """Whether a process with this pid currently exists (best effort).

    Used to tell a run that is genuinely still working from one whose process
    died while the server was down. On Windows ``os.kill`` would *terminate*
    the target rather than probe it, so this asks the kernel directly.
    """
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - platform branch
        import ctypes

        still_active = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - another user's process
        return True
    return True


def kill_tree(pid: int) -> None:
    """Signal a run and everything it spawned.

    Runs are started in their own process group (or a new process group on
    Windows) and spawn worker processes of their own, so killing the single
    child pid would leave orphans holding provider connections open.
    """
    if pid is None or pid <= 0:
        return
    if os.name == "nt":  # pragma: no cover - platform branch
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            check=False, capture_output=True,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def _reap(proc: Any | None) -> None:
    """Collect a killed child so it does not linger, escalating once.

    A stopped run must not keep the request hanging: after the graceful
    timeout the child is killed outright, and if even that does not land the
    record still says ``stopped`` — stopping is best effort by nature.
    """
    wait = getattr(proc, "wait", None)
    if wait is None:
        return
    try:
        wait(timeout=STOP_TIMEOUT)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name != "nt":  # Windows already used `taskkill /F`
        proc.kill()
        try:
            wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass


class RunManager:
    """Starts, tracks and stops CLI runs, one directory per run."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        python: str | None = None,
        popen: Callable[..., Any] | None = None,
    ) -> None:
        # `root` is read at call time via REPO_ROOT so tests can redirect the
        # whole data layout by patching one module attribute.
        self.root = Path(root) if root is not None else REPO_ROOT
        self.python = python or os.environ.get("PYTHON") or _default_python()
        self._popen = popen or subprocess.Popen
        self.store = RunsStore(self.root / LOGS_DIR)
        self.procs: dict[str, Any] = {}

    # Layout ------------------------------------------------------------
    def logs_dir(self, run_id: str) -> Path:
        """The run's metadata/log directory; ``ValueError`` if the id escapes."""
        return safe_child(self.root / LOGS_DIR, run_id)

    def output_dir(self, run_id: str) -> Path:
        """The run's artefact directory; ``ValueError`` if the id escapes."""
        return safe_child(self.root / OUTPUT_DIR, run_id)

    def env_for(self, record: RunRecord) -> dict[str, str]:
        """Environment overrides that keep a run's logs inside its own dir.

        ``PYTHONIOENCODING`` is not cosmetic: the child's stdout/stderr are
        files we open as UTF-8, but Python would otherwise encode them with
        the console locale (cp1252 on Windows) and die the first time a log
        line contains Vietnamese text.
        """
        env = {
            "LOGS_DIR": str(self.logs_dir(record.run_id)),
            "LOG_JSONL": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        if record.mock:
            env["LLM_MOCK"] = "1"
        return env

    # Planning ----------------------------------------------------------
    def plan(self, file: Path, max_tokens: int | None = None) -> dict[str, Any]:
        """Ask ``probe.py`` how the document would be split.

        ``probe.py`` reports the default output path for its own default target
        language; the caller writes to ``--out`` explicitly, so only ``chunks``
        and ``title`` are consumed here.
        """
        argv = [self.python, str(_PROBE), "--plan", str(file)]
        if max_tokens:
            argv += ["--max-tokens", str(max_tokens)]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, cwd=self.root, check=False
            )
        except OSError as exc:  # interpreter missing, permissions, ...
            raise RunPlanError(f"cannot run the probe: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RunPlanError(f"probe failed: {detail or 'no diagnostics'}")
        try:
            data = json.loads(proc.stdout)
        except ValueError as exc:
            raise RunPlanError(f"probe returned no usable plan: {exc}") from exc
        if not isinstance(data, dict):
            raise RunPlanError("probe returned no usable plan: not an object")
        return data

    # Launch ------------------------------------------------------------
    def start(
        self,
        *,
        file: str,
        src: str,
        tgt: str,
        domain: str,
        glossary: str | None,
        max_tokens: int | None,
        mock: bool,
    ) -> RunRecord:
        """Plan, then launch one CLI run and return its record.

        Raises ``ValueError`` for a path outside ``input/`` and
        :class:`RunPlanError` when the document cannot be read or planned — in
        both cases nothing is created and no process is started.
        """
        source = safe_child(self.root / INPUT_DIR, file)
        if not source.is_file():
            raise RunPlanError(f"input file not found: {file}")

        # Before any side effect: a bad document must not leave an empty run dir.
        plan = self.plan(source, max_tokens)

        run_id = _new_run_id()
        logs_dir, out_dir = self.logs_dir(run_id), self.output_dir(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{source.stem}.{tgt}.md"

        argv = self._argv(source, src, tgt, domain, glossary, max_tokens, output_path)
        record = RunRecord(
            run_id=run_id,
            started=time.time(),
            status="running",
            file=Path(file).name,
            src=src,
            tgt=tgt,
            domain=domain,
            chunks=int(plan.get("chunks") or 0),
            exit_code=None,
            output_path=str(output_path),
            mock=mock,
            pid=None,
            command=argv,
        )
        env = {**os.environ, **self.env_for(record)}
        with (logs_dir / "stdout.log").open("w", encoding="utf-8") as out, (
            logs_dir / "stderr.log"
        ).open("w", encoding="utf-8") as err:
            proc = self._popen(
                argv, cwd=self.root, env=env, stdout=out, stderr=err,
                **_detach_kwargs(),
            )
        record.pid = proc.pid
        self.procs[run_id] = proc
        self.store.write_record(record)
        return record

    def _argv(
        self,
        source: Path,
        src: str,
        tgt: str,
        domain: str,
        glossary: str | None,
        max_tokens: int | None,
        output_path: Path,
    ) -> list[str]:
        """The exact CLI command line, also stored in the record for auditing."""
        argv = [
            self.python, "-m", "polyglot_lc.cli", "translate",
            "--file", str(source),
            "--src", src,
            "--tgt", tgt,
            "--domain", domain,
            "--out", str(output_path),
            "--quiet",
        ]
        if glossary:
            argv += ["--glossary", str(glossary)]
        if max_tokens:
            argv += ["--max-tokens", str(max_tokens)]
        return argv

    # Observation and control -------------------------------------------
    def has_events(self, run_id: str) -> bool:
        """Whether the run wrote an event log at all (``LOG_JSONL=1``)."""
        return self.store.log_path(run_id) is not None

    def events(self, run_id: str, after: int) -> tuple[list[dict[str, Any]], int, int]:
        """Events from index *after* on, plus ``(next_index, malformed)``."""
        return self.store.read_events(run_id, after)

    def status(self, run_id: str) -> RunRecord:
        """The run's state, reconciled with the child process and the disk.

        Called on every poll: without this a run whose process died while the
        server was down would stay "running" forever, which is exactly the
        state the UI can never resolve on its own.
        """
        record = self.store.read_record(run_id)
        if record is None:
            raise KeyError(run_id)
        before = (record.status, record.exit_code, record.note)
        if record.status == "running":
            proc = self.procs.get(run_id)
            if proc is not None:
                code = proc.poll()
                if code is not None:
                    record.exit_code = int(code)
                    record.status = "done" if record.exit_code == EXIT_OK else "failed"
                    self.procs.pop(run_id, None)
            elif record.pid is not None and not pid_alive(record.pid):
                record.status = "failed"
                _note(record, "server đã khởi động lại khi lượt chạy đang chạy")
        if not self.has_events(run_id):
            _note(record, NO_LOG_NOTE)
        if (record.status, record.exit_code, record.note) != before:
            self.store.write_record(record)
        return record

    def stop(self, run_id: str) -> RunRecord:
        """Kill a running run and its children, then mark it stopped.

        ``KeyError`` for an unknown run, ``RuntimeError`` when it already
        finished: a pid that has been recycled must never be signalled.
        """
        record = self.status(run_id)  # hoà giải trước, để không kill pid đã tái sử dụng
        if record.status != "running":
            raise RuntimeError("not running")
        proc = self.procs.pop(run_id, None)
        pid = record.pid if record.pid is not None else getattr(proc, "pid", None)
        if pid:
            kill_tree(pid)
        _reap(proc)
        record.status = "stopped"
        _note(record, f"đã dừng từ web UI (pid {pid})" if pid else "đã dừng từ web UI")
        self.store.write_record(record)
        return record


def _new_run_id() -> str:
    """A sortable, collision-proof id: timestamp plus a short random suffix."""
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def _default_python() -> str:
    """Prefer the running interpreter: it is the one with the package installed."""
    import sys

    return sys.executable or "python"


def _detach_kwargs() -> dict[str, Any]:
    """Platform flags that let us signal the whole child group later."""
    if os.name == "nt":  # pragma: no cover - platform branch
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}
