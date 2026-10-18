"""Run supervision: one child `polyglot_lc.cli` process per document.

The web UI never imports the pipeline: it launches the CLI the same way a user
would, so a run cannot wedge the server and the two front ends cannot drift.
Plan-then-start: :meth:`RunManager.plan` probes the document first, and only
when it succeeds does :meth:`RunManager.start` create a run directory, spawn the
child and persist the record.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from webui.paths import REPO_ROOT, is_within, safe_child
from webui.store import RunRecord, RunsStore

logger = logging.getLogger("webui.runs")

#: Where uploaded/translator inputs live, relative to the repository root.
INPUT_DIR = "input"

#: Per-run directories for artefacts and for metadata + logs, under the repo.
OUTPUT_DIR = Path("output") / "webui"
LOGS_DIR = Path("logs") / "webui"

#: The probe script, resolved from this file so a caller-provided root cannot
#: redirect it: the script ships with the package, the data does not.
_PROBE = Path(__file__).with_name("probe.py")

#: Exit code that means the run finished its work.
EXIT_OK = 0

#: How long a stopped run may take to die before we escalate once.
STOP_TIMEOUT = 5.0

#: Explained in the record when there is no event log to read. Written in the
#: present tense on purpose: the log appears a moment after the child starts, so
#: "chưa có" (not yet) is true while "không có" would be a claim we cannot make.
#: :meth:`RunManager.status` takes the note back once a log does appear.
NO_LOG_NOTE = "chưa thấy file sự kiện của lượt chạy (LOG_JSONL=0?): chưa có tiến độ để hiển thị"

#: Every wording this note has had. A record written by an older server keeps its
#: sentence in `run.json` forever, and once a log exists that sentence is false on
#: screen — the history would go on claiming a finished run reports no progress
#: while its own step rail is full of ticks. Matching the old text too is what lets
#: an upgrade clean up the records it inherited.
NO_LOG_NOTES = (
    NO_LOG_NOTE,
    "lượt chạy không có file sự kiện (LOG_JSONL=0?): không có tiến độ để hiển thị",
)

#: Event statuses that mean something went wrong: the pipeline's own word for a
#: step that failed, and for a call it had to retry.
PROBLEM_STATUSES = frozenset({"fail", "retry"})

#: How long the supervisor waits before each automatic continue, in seconds.
#: Three attempts over about ten minutes: long enough for a rate limit or a
#: provider wobble to clear, short enough that a run which cannot recover reaches
#: its final state while the reader still remembers starting it.
AUTO_RESUME_DELAYS = (30.0, 120.0, 480.0)

#: How often the supervisor looks for a run that is due.
SWEEP_INTERVAL = 5.0

#: Words that mean trying the failed work again can work: the provider was busy,
#: slow, or briefly unreachable. Matched case-insensitively against the run's own
#: error records and against whatever the child printed while dying, because both
#: carry our normalised class name in front of the provider's message
#: (``RateLimitError:``, ``TransientError:``).
RETRYABLE_HINTS = (
    "ratelimiterror",
    "transienterror",
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "overloaded",
    "try again in",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
)

#: Words that mean it cannot: the request itself was refused, so a second
#: identical attempt would only spend the budget and bury the real reason under
#: three failures with the same cause. Checked *before* the hints above, so one
#: refused chunk is enough to leave the whole run alone.
PERMANENT_HINTS = (
    "permanenterror",
    "configerror",
    "configuration error",
    "invalid api key",
    "incorrect api key",
    "unauthorized",
    "authenticationerror",
    "invalid_request_error",
)


class RunPlanError(RuntimeError):
    """The document could not be planned, so no run was started."""


def _chunk_order(chunk_id: str) -> tuple[int, str]:
    """Sort key for a chunk id: the number in it first, then the id itself.

    Ids are ``chunk-000`` and sort as text perfectly well right up to the point
    where a run has a thousand chunks and ``chunk-1000`` sorts before
    ``chunk-999``. A thousand-chunk book is exactly the size where a reader
    watches the translation fill in, so the document's own order is taken from
    the number rather than from the string. Ties — two ids carrying the same
    number — keep the ids' own order instead of a set's.
    """
    digits = "".join(character for character in chunk_id if character.isdigit())
    return (int(digits) if digits else 0, chunk_id)


def _note(record: RunRecord, text: str) -> None:
    """Append an explanation, without repeating one a previous poll wrote."""
    if text and text not in record.note:
        record.note = f"{record.note}; {text}" if record.note else text


def _un_note(record: RunRecord, text: str) -> None:
    """Take an explanation back once it has stopped being true.

    A note is written to disk on the first poll that cannot find the run's log,
    and the log then appears a moment later. Leaving the sentence behind would
    keep a healthy run looking broken for the rest of its life.
    """
    if not text or text not in record.note:
        return
    parts = [
        part.strip() for part in record.note.split(";") if part.strip() and part.strip() != text
    ]
    record.note = "; ".join(parts)


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
            check=False,
            capture_output=True,
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


def _remove_tree(path: Path) -> None:
    """Remove a run's directory, retrying over a transient lock.

    On Windows a *brief* handle to the directory (an antivirus scan, a lingering
    child) can make ``shutil.rmtree`` fail on the final ``rmdir``; ignoring that
    failure would leave an empty run directory behind and the deleted run would
    keep a ghost entry on disk. Retrying a few times clears such a lock.
    """
    if not path.is_dir():
        return
    # A held handle can keep the final rmdir failing for a second or two, so
    # retry on a 2s window; the common case (no lock) returns on the first try.
    for _ in range(20):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            time.sleep(0.1)


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

    def input_path(self, name: str) -> Path:
        """The input file *name* refers to; ``ValueError`` if it escapes ``input/``."""
        return safe_child(self.root / INPUT_DIR, name)

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
            # The id the run's own trace records carry, so ``run.json`` here and
            # the trace on disk name the same run instead of two parallel ids.
            "RUN_ID": record.run_id,
        }
        if record.mock:
            env["LLM_MOCK"] = "1"
        # A resumed run reuses the stopped run's checkpoints, and those are
        # keyed by the translator model (and the tokenizer it picks). Pinning
        # the model the stopped run reported keeps the resumed run on the very
        # same checkpoint identity instead of recomputing a different one.
        # A variable in the child's env wins over `.env`, so this overrides
        # whatever the server is currently configured with. Fresh runs leave
        # the field empty, so they track the config that is in effect now.
        if record.translator_model:
            env["TRANSLATOR_MODEL"] = record.translator_model
        return env

    # Probing -----------------------------------------------------------
    def probe(self, args: list[str]) -> dict[str, Any]:
        """Run ``probe.py`` with *args* and return the JSON object it prints.

        The probe is the only thing that may observe the pipeline's real
        configuration, so both planning and the settings screen go through
        here; a failure becomes :class:`RunPlanError` carrying the reason the
        probe printed, never a traceback.
        """
        argv = [self.python, str(_PROBE), *args]
        # The probe prints non-ASCII (filenames, model ids) with
        # `ensure_ascii=False`, so both ends are pinned to UTF-8 rather than
        # left to whatever code page this console happens to use.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                encoding="utf-8",
                cwd=self.root,
                env=env,
                check=False,
            )
        except OSError as exc:  # interpreter missing, permissions, ...
            raise RunPlanError(f"cannot run the probe: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RunPlanError(f"probe failed: {detail or 'no diagnostics'}")
        try:
            data = json.loads(proc.stdout)
        except ValueError as exc:
            raise RunPlanError(f"probe returned no usable result: {exc}") from exc
        if not isinstance(data, dict):
            raise RunPlanError("probe returned no usable result: not an object")
        return data

    # Planning ----------------------------------------------------------
    def plan(self, file: Path, max_tokens: int | None = None) -> dict[str, Any]:
        """Ask ``probe.py`` how the document would be split.

        ``probe.py`` reports the default output path for its own default target
        language; the caller writes to ``--out`` explicitly, so only ``chunks``
        is consumed here.
        """
        args = ["--plan", str(file)]
        if max_tokens:
            args += ["--max-tokens", str(max_tokens)]
        return self.probe(args)

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
        output_format: str = "md",
    ) -> RunRecord:
        """Plan, then launch one **new** CLI run and return its record.

        A new run gets its own id, its own log directory and its own output
        file, and the pipeline translates the document from its first chunk.
        Continuing an interrupted run is :meth:`resume`: same run, same files,
        only the chunks that never finished are paid for again.
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
        # The Markdown path is always the base name: a run asked for the PDF
        # alone writes `<stem>.<tgt>.pdf` right next to it (see
        # `polyglot_lc.pipeline.assembler.output_paths`).
        output_path = self.output_dir(run_id) / f"{source.stem}.{tgt}.md"

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
            command=self._argv(
                source, src, tgt, domain, glossary, max_tokens, output_path, output_format
            ),
        )
        return self._launch(record)

    def resume(self, run_id: str) -> RunRecord:
        """Continue an interrupted run **as that same run**.

        "Tiếp tục" is not a new translation. The chunks the stopped attempt
        already finished are the work left to continue, and the checkpoint
        store keys them by the command that produced them — file, languages,
        domain, glossary, token budget — plus the Translator model. So the
        command the run actually ran is relaunched verbatim with ``--resume``
        (rebuilding flags from today's settings would re-chunk the document
        under another token budget and miss every checkpoint), the model the
        run itself reported is pinned into the child's env, and the run keeps
        its id, log directory, output path and start time. The page goes on
        watching one run instead of gaining a second one for the same document.

        ``KeyError`` for an unknown run; ``RuntimeError`` when it is still
        running (a second child would write the same log and artefacts) or when
        its record carries no command to continue.
        """
        record = self.store.read_record(run_id)
        if record is None:
            raise KeyError(run_id)
        if self._alive(record):
            raise RuntimeError(f"run {run_id} is still running")
        if not record.command:
            raise RuntimeError(f"run {run_id} has no stored command to continue")
        argv = list(record.command)
        if "--resume" not in argv:
            argv.append("--resume")
        record.command = argv
        # Read from the run's own `run_start` record when the field is not
        # filled in yet: checkpoints are keyed by the model (and the tokenizer
        # it picks), so a resumed attempt that fell back to the server's
        # current `.env` would derive a different key and re-translate every
        # chunk it was meant to reuse.
        record.translator_model = record.translator_model or self.translator_model(run_id)
        record.status = "running"
        record.exit_code = None
        record.note = ""
        # An attempt is starting now, whoever asked for it: a wait the supervisor
        # had scheduled no longer means anything. The attempt count is kept — it
        # is the run's, not the reader's, and handing it back would let a manual
        # click buy a fresh budget of automatic attempts.
        record.retry_at = 0.0
        return self._launch(record, keep_logs=True)

    def _alive(self, record: RunRecord) -> bool:
        """Whether this run still has a child process running.

        A pid alone is not enough: a process that died while the server was
        down leaves the record saying ``running`` over a pid nobody owns.
        """
        proc = self.procs.get(record.run_id)
        if proc is not None:
            return proc.poll() is None
        return record.status == "running" and record.pid is not None and pid_alive(record.pid)

    def _launch(self, record: RunRecord, *, keep_logs: bool = False) -> RunRecord:
        """Spawn the child for *record* and persist it.

        The record must already carry its run id, its command and its output
        path: a fresh run has just built them, a resumed one reuses the ones
        the interrupted attempt ran under.
        """
        logs_dir, out_dir = self.logs_dir(record.run_id), self.output_dir(record.run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **self.env_for(record)}
        # Append rather than truncate when continuing a run: the previous
        # attempt's console output is still the best answer to "why did it
        # stop?", and the event log beside it is read the same way.
        mode = "a" if keep_logs else "w"
        with (
            (logs_dir / "stdout.log").open(mode, encoding="utf-8") as out,
            (logs_dir / "stderr.log").open(mode, encoding="utf-8") as err,
        ):
            proc = self._popen(
                record.command,
                cwd=self.root,
                env=env,
                stdout=out,
                stderr=err,
                **_detach_kwargs(),
            )
        record.pid = proc.pid
        self.procs[record.run_id] = proc
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
        output_format: str = "md",
    ) -> list[str]:
        """The exact CLI command line, also stored in the record for auditing."""
        argv = [
            self.python,
            "-m",
            "polyglot_lc.cli",
            "translate",
            "--file",
            str(source),
            "--src",
            src,
            "--tgt",
            tgt,
            "--domain",
            domain,
            "--out",
            str(output_path),
            "--format",
            output_format,
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

    def partial(self, run_id: str) -> dict[str, Any]:
        """The chunks this run has translated so far, in document order.

        A run writes each finished chunk's translation to its own checkpoint
        file as it completes, so the text of a run that is still going is
        already on disk — the page can show it instead of an empty column that
        promises something later. The order is the document's, taken from the
        chunk ids themselves: a checkpoint directory is written in whatever
        order the chunks happen to finish, and chunks run concurrently here.

        Deliberately not cached: the page asks only when the number of finished
        chunks has moved, and re-reading a few dozen small files then costs
        less than keeping a second copy of the text in step with the disk.
        ponytail: one directory scan per ask; add a cache if a run ever carries
        thousands of chunks.
        """
        directory = self._checkpoint_dir(run_id)
        if directory is None:
            return {"pieces": 0, "text": ""}
        written: list[tuple[tuple[int, str], str]] = []
        for path in directory.glob("*.json"):
            if path.name == "meta.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue  # a half-written file is a chunk that has not landed yet
            if not isinstance(data, dict):
                continue
            text = data.get("translated")
            if not isinstance(text, str) or not text.strip():
                continue  # an empty checkpoint is a chunk still to be translated
            chunk_id = str(data.get("chunk_id") or path.stem)
            written.append((_chunk_order(chunk_id), text))
        written.sort(key=lambda piece: piece[0])
        return {
            "pieces": len(written),
            "text": "\n\n".join(text for _, text in written),
        }

    def _checkpoint_dir(self, run_id: str) -> Path | None:
        """The directory this run's own log says its checkpoints live in.

        The directory is named by the run rather than derived here, because it
        is the pipeline's configuration — and it is refused unless it lands
        inside the repository: a path recorded in a log is a pointer, never a
        grant to read something outside the project.
        """
        events, _next, _malformed = self.events(run_id, 0)
        for event in reversed(events):
            if str(event.get("type") or "") != "checkpoint":
                continue
            candidate = Path(str(event.get("path") or ""))
            # The child ran with the repository root as its working directory,
            # so a relative directory means "under the repository", whatever
            # directory this server happens to have been started from.
            if not candidate.is_absolute():
                candidate = self.root / candidate
            if is_within(self.root, candidate) and candidate.is_dir():
                return candidate
            return None
        return None

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
        if self.has_events(run_id):
            # The log exists after all: a note written in the first second, when
            # the child had not opened it yet, is no longer true — and neither is
            # the same sentence written by an earlier version of this server.
            for wording in NO_LOG_NOTES:
                _un_note(record, wording)
        else:
            _note(record, NO_LOG_NOTE)
        if (record.status, record.exit_code, record.note) != before:
            self.store.write_record(record)
        return record

    def translator_model(self, run_id: str) -> str:
        """The translator model the run actually ran under, or ``""``.

        Read from the child's own ``run_start`` event: the record only gains
        the field lazily, so the model is known from the first poll onward and
        survives a server restart. A run that died before its first event has
        no checkpoint to reuse, so an empty answer is the right one for it.
        """
        record = self.store.read_record(run_id)
        if record is not None and record.translator_model:
            return record.translator_model
        model = ""
        events, _next, _malformed = self.events(run_id, 0)
        for event in events:
            if str(event.get("type") or "") != "run_start":
                continue
            agents = event.get("agents") or []
            for agent in agents:
                if isinstance(agent, dict) and str(agent.get("name")) == "translator":
                    model = str(agent.get("model") or "")
                    break
            if model:
                break
        if record is None:
            return model
        if model and model != record.translator_model:
            record.translator_model = model
            self.store.write_record(record)
        return model

    # Automatic continue ------------------------------------------------
    def sweep(self, now: float | None = None) -> list[RunRecord]:
        """Continue, by itself, the runs a temporary provider failure interrupted.

        Called on a timer while the server runs (see :func:`start_sweeper`). A
        run is only ever continued through :meth:`resume`, so it keeps its id,
        its log directory, its checkpoints and its output path, and the page goes
        on watching one run — the reader's "Tiếp tục" is simply pressed for them.

        Returns the records this pass scheduled or relaunched, which is also what
        lets a test drive the whole policy without a clock or a thread.
        """
        moment = time.time() if now is None else now
        touched: list[RunRecord] = []
        for record in self.store.list_records():
            try:
                outcome = self._auto_continue(record, moment)
            except (KeyError, RuntimeError, OSError, ValueError) as exc:
                # One run that cannot be continued right now must not stop the
                # pass: the next one will find it again.
                logger.warning("auto-continue skipped for %s: %s", record.run_id, exc)
                continue
            if outcome is not None:
                touched.append(outcome)
        return touched

    def _auto_continue(self, record: RunRecord, now: float) -> RunRecord | None:
        """Schedule or perform this run's automatic continue, if it is owed one."""
        if record.status == "running":
            # Reconcile first: a child that died while no page was open is the one
            # case where the record still says "running" over a dead pid, and
            # that is exactly the run this feature is for.
            record = self.status(record.run_id)
        interrupted = record.status == "failed" or (
            record.status == "done" and self._missing_chunks(record)
        )
        if not interrupted:
            # Running, stopped by the reader, or finished cleanly: not ours to
            # touch. "Dừng" is a decision, and it outranks the error.
            return None
        if record.auto_resumes >= len(AUTO_RESUME_DELAYS) or not record.command:
            return None
        if not self._retryable(record):
            return None
        if record.retry_at == 0:
            return self._schedule_continue(record, now)
        if now < record.retry_at:
            return None
        return self._continue(record)

    def _schedule_continue(self, record: RunRecord, now: float) -> RunRecord:
        """Spend one attempt and put the run to sleep until it is due.

        The wait is the point: a provider that just refused everything needs time
        to clear, and relaunching at once would burn all three attempts inside the
        same outage.
        """
        record.auto_resumes += 1
        wait = AUTO_RESUME_DELAYS[record.auto_resumes - 1]
        record.retry_at = now + wait
        record.note = (
            f"lỗi tạm thời từ nhà cung cấp — tự động tiếp tục lần "
            f"{record.auto_resumes}/{len(AUTO_RESUME_DELAYS)}, "
            f"thử lại sau {_human_wait(wait)}"
        )
        self.store.write_record(record)
        logger.info(
            "auto-continue scheduled for %s: attempt %d/%d in %.0fs",
            record.run_id,
            record.auto_resumes,
            len(AUTO_RESUME_DELAYS),
            wait,
        )
        return record

    def _continue(self, record: RunRecord) -> RunRecord:
        """Relaunch a due run through ``resume``, and say so on the record."""
        continued = self.resume(record.run_id)
        continued.note = (
            f"đang tự tiếp tục lần {continued.auto_resumes}/{len(AUTO_RESUME_DELAYS)} "
            "sau lỗi tạm thời từ nhà cung cấp"
        )
        self.store.write_record(continued)
        logger.info(
            "auto-continued run %s (attempt %d/%d)",
            continued.run_id,
            continued.auto_resumes,
            len(AUTO_RESUME_DELAYS),
        )
        return continued

    def _missing_chunks(self, record: RunRecord) -> int:
        """How many chunks the run's own last attempt says it never translated.

        A provider failure on individual chunks does not kill a run: it finishes
        and marks the gaps, so its exit code says nothing about the work that is
        missing. The count comes from the attempt's own ``run_end`` record.
        """
        return int(self.store.last_run_end(record.run_id).get("chunks_failed") or 0)

    def _retryable(self, record: RunRecord) -> bool:
        """Whether this failure reads like one that trying again can fix.

        The evidence is the run's own words — the error records it wrote and what
        the child printed while dying — and nothing here interprets the pipeline's
        internals: it matches text a reader could find in the run's log. A failure
        that reads like neither kind is left alone rather than retried on a guess,
        which is what keeps a bug from being looped over three times.
        """
        text = self._failure_text(record).lower()
        if not text:
            return False
        if any(hint in text for hint in PERMANENT_HINTS):
            return False
        return any(hint in text for hint in RETRYABLE_HINTS)

    def _failure_text(self, record: RunRecord) -> str:
        """Everything the run said about why it stopped, as one string."""
        try:
            events, _next, _malformed = self.events(record.run_id, 0)
        except Exception:  # noqa: BLE001 - a broken log must not hide the run
            events = []
        parts = [
            str(event.get("error") or event.get("message") or "")
            for event in events
            if str(event.get("status") or "") in PROBLEM_STATUSES
        ]
        if record.status == "failed":
            # A crash before the first error record is explained only here.
            parts.append(self.store.tail_output(record.run_id))
        return " ".join(parts)

    def stop(self, run_id: str) -> RunRecord:
        """Kill a running run and its children, or call off a waiting retry.

        ``KeyError`` for an unknown run, ``RuntimeError`` when there is nothing
        to stop: a pid that has been recycled must never be signalled.
        """
        record = self.status(run_id)  # hoà giải trước, để không kill pid đã tái sử dụng
        if record.status != "running":
            if not record.retry_at:
                raise RuntimeError("not running")
            # Between two attempts there is no child to signal, and "Dừng" here
            # means "đừng thử lại nữa". The budget is what the supervisor reads,
            # so spending the rest of it is the cancel; the run keeps the status
            # it earned rather than claiming it was stopped, and its note stops
            # promising an attempt that will not come.
            record.retry_at = 0.0
            record.auto_resumes = len(AUTO_RESUME_DELAYS)
            record.note = "đã huỷ tự động tiếp tục sau lỗi tạm thời từ nhà cung cấp"
            self.store.write_record(record)
            return record
        proc = self.procs.pop(run_id, None)
        pid = record.pid if record.pid is not None else getattr(proc, "pid", None)
        if pid:
            kill_tree(pid)
        _reap(proc)
        record.status = "stopped"
        _note(record, f"đã dừng từ web UI (pid {pid})" if pid else "đã dừng từ web UI")
        self.store.write_record(record)
        return record

    def delete(self, run_id: str) -> None:
        """Erase a run's history, its output, and the input PDF it came from.

        A running run is stopped first (best effort), then the run's log
        directory and artefact directory are removed, and finally the input
        PDF is deleted — but only when no other run still references the same
        file name: two runs may share one document, and removing it under the
        other would break its resume. ``KeyError`` for an unknown run.
        """
        record = self.store.read_record(run_id)
        if record is None:
            raise KeyError(run_id)
        # A live run cannot leave a child holding its output open behind the
        # rmtree; stop is best effort — a record already final just skips it.
        if record.status == "running" or run_id in self.procs:
            try:
                self.stop(run_id)
            except (KeyError, RuntimeError):
                pass
        for dirname in (self.logs_dir(run_id), self.output_dir(run_id)):
            _remove_tree(dirname)
        self.procs.pop(run_id, None)
        self._remove_input(record)

    def _remove_input(self, record: RunRecord) -> None:
        """Delete the run's input PDF when no other run still points at it.

        The input file is shared by name: several runs can translate the same
        document, and ``resume`` on any of them needs the file to still be
        there. So it is removed only when this run is the last one referencing
        that name. The path is validated the same way as on upload (a name that
        is really a path is refused), so a hand-edited record cannot point the
        delete at an arbitrary file.
        """
        try:
            path = self.input_path(record.file)
        except ValueError:
            return
        others = [
            other
            for other in self.store.list_records()
            if other.run_id != record.run_id and other.file == record.file
        ]
        if others:
            return
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def _human_wait(seconds: float) -> str:
    """A wait a reader can act on: ``30 giây``, ``2 phút``."""
    if seconds < 60:
        return f"{seconds:.0f} giây"
    return f"{seconds / 60:.0f} phút"


def start_sweeper(
    manager: RunManager,
    *,
    interval: float = SWEEP_INTERVAL,
    stop: threading.Event | None = None,
) -> threading.Thread:
    """Run :meth:`RunManager.sweep` on a timer until *stop* is set.

    A daemon thread beside the HTTP server, not a request handler: the automatic
    continue has to happen whether or not a page is open, and a request must stay
    a request — no ``GET`` should launch a process because of this. A pass that
    raises is logged and the loop goes on; the supervisor dying quietly would look
    exactly like a run that never comes back to life.
    """

    def loop() -> None:
        while stop is None or not stop.is_set():
            try:
                # Each outcome names itself in the log (scheduled, or
                # relaunched); repeating that here would call a wait a continue.
                manager.sweep()
            except Exception:
                logger.exception("auto-continue sweep failed")
            time.sleep(interval)

    thread = threading.Thread(target=loop, name="webui-auto-resume", daemon=True)
    thread.start()
    return thread


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
