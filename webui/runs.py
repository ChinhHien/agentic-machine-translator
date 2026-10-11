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
LOGS_DIR = Path("logs") / "webui"

#: The probe script, resolved from this file so a caller-provided root cannot
#: redirect it: the script ships with the package, the data does not.
_PROBE = Path(__file__).with_name("probe.py")

class RunPlanError(RuntimeError):
    """The document could not be planned, so no run was started."""


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
        """Environment overrides that keep a run's logs inside its own dir."""
        env = {"LOGS_DIR": str(self.logs_dir(record.run_id)), "LOG_JSONL": "1"}
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
            note=str(plan.get("title") or ""),
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
