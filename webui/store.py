"""Stores for the web UI: runs and glossaries.

Everything here is a plain file reader meant to be called from the server
process, so nothing in this module imports :mod:`polyglot_lc` (see the package
docstring in :mod:`webui`).

There is deliberately no configuration store: the web application has no view
of `.env` at all — not a value, not a key name, not whether a key is set. What
a run needs from the environment it inherits from the process that started the
server, and the terminal is where a human inspects or changes that.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from webui.paths import safe_child


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    """Parsed JSON at *path*, or ``None`` when it is absent or malformed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* through a temp file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Every object in a JSONL file, plus the count of unusable lines.

    A run's log is written while this is read, so the last line can be a partial
    write. Malformed lines are counted and skipped rather than aborting the
    read: the caller reports the count and the next poll picks the line up
    again once it is complete.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return [], 0
    items: list[dict[str, Any]] = []
    malformed = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(payload, dict):
            items.append(payload)
        else:
            malformed += 1
    return items, malformed


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
#: Name of the metadata file inside each run directory.
RECORD_NAME = "run.json"

#: A run's lifecycle: is the child alive, and how did it end?
RUN_STATUSES = ("running", "done", "failed", "stopped")


@dataclass
class RunRecord:
    """One run's launch parameters and observed outcome.

    Persisted as ``logs/webui/<run_id>/run.json`` so the history survives a
    server restart: in-memory state would not, and the plan's history view is
    the whole reason the metadata is written at all.
    """

    run_id: str
    started: float
    status: str
    file: str
    src: str
    tgt: str
    chunks: int
    exit_code: int | None
    output_path: str
    mock: bool
    pid: int | None
    command: list[str]
    note: str = ""
    #: The translator model this run actually ran under, read from the child's
    #: own `run_start` event once it appears. A resume reuses the stopped run's
    #: checkpoints, which are keyed by the model, so it must pin this value
    #: rather than re-read it from `.env`; empty until the first event arrives.
    translator_model: str = ""
    #: How many times the supervisor has continued this run by itself after a
    #: temporary provider failure, and when the next such attempt is due (epoch
    #: seconds; 0 means none is pending). Both live on the record rather than in
    #: the server's memory: a restart must neither forget the attempts already
    #: spent nor lose a scheduled one, and the budget has to survive a browser
    #: that is closed for the whole wait.
    auto_resumes: int = 0
    retry_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> RunRecord | None:
        """Rebuild a record, ignoring anything malformed."""
        if not isinstance(data, dict):
            return None
        run_id = str(data.get("run_id", ""))
        if not run_id:
            return None
        raw_exit = data.get("exit_code")
        raw_pid = data.get("pid")
        return cls(
            run_id=run_id,
            started=float(data.get("started") or 0.0),
            status=str(data.get("status") or "running"),
            file=str(data.get("file", "")),
            src=str(data.get("src", "")),
            tgt=str(data.get("tgt", "")),
            chunks=int(data.get("chunks") or 0),
            exit_code=None if raw_exit is None else int(raw_exit),
            output_path=str(data.get("output_path", "")),
            mock=bool(data.get("mock", False)),
            pid=None if raw_pid is None else int(raw_pid),
            command=[str(x) for x in (data.get("command") or [])],
            note=str(data.get("note") or ""),
            translator_model=str(data.get("translator_model") or ""),
            auto_resumes=int(data.get("auto_resumes") or 0),
            retry_at=float(data.get("retry_at") or 0.0),
        )


class RunsStore:
    """Per-run metadata and the run's own event log, both under one root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def dir_for(self, run_id: str) -> Path:
        """The run's directory; ``ValueError`` if the id could escape the root."""
        return safe_child(self.root, run_id)

    def write_record(self, record: RunRecord) -> None:
        _write_json_atomic(self.dir_for(record.run_id) / RECORD_NAME, record.to_dict())

    def read_record(self, run_id: str) -> RunRecord | None:
        try:
            path = self.dir_for(run_id) / RECORD_NAME
        except ValueError:
            return None
        return RunRecord.from_dict(_read_json(path))

    def list_records(self) -> list[RunRecord]:
        """Every run with metadata, newest first."""
        if not self.root.is_dir():
            return []
        records = [
            record
            for path in sorted(self.root.glob(f"*/{RECORD_NAME}"))
            if (record := self.read_record(path.parent.name)) is not None
        ]
        records.sort(key=lambda r: r.started, reverse=True)
        return records

    def log_files(self, run_id: str) -> list[Path]:
        """Every event log this run wrote, oldest first.

        A run writes one ``run_<stamp>.jsonl`` per attempt, because "Tiếp tục"
        continues the same run in the same directory and the child opens a new
        file each time. Names sort in the order the attempts ran, so the merged
        stream is chronological without asking any clock.
        """
        try:
            directory = self.dir_for(run_id)
        except ValueError:
            return []
        if not directory.is_dir():
            return []
        return sorted(directory.glob("run_*.jsonl"))

    def log_path(self, run_id: str) -> Path | None:
        """The run's newest event log, or ``None`` when it wrote none."""
        logs = self.log_files(run_id)
        return logs[-1] if logs else None

    def last_run_end(self, run_id: str) -> dict[str, Any]:
        """The newest attempt's own ``run_end`` record, or ``{}``.

        The supervisor asks this on a timer to tell a run that finished from one
        that finished with gaps, and it reads only the newest attempt's file:
        that attempt's verdict is the current one, and stopping there keeps the
        question cheap however many times the run has already been continued.
        """
        path = self.log_path(run_id)
        if path is None:
            return {}
        events, _malformed = _read_jsonl(path)
        for event in reversed(events):
            if str(event.get("type") or "") == "run_end":
                return event
        return {}

    def read_events(self, run_id: str, after: int) -> tuple[list[dict[str, Any]], int, int]:
        """Events from index *after* on, plus ``(next_index, malformed_count)``.

        Indexing is by position in the usable event list rather than by
        timestamp, so a poll does not depend on any clock and cannot lose an
        event to a timestamp tie.

        Every attempt of the run is read, oldest file first: continuing a run
        appends a file, and reading only the newest one would erase the run's
        earlier history from the page — its progress bar among other things.
        The earlier indices keep their meaning, so a page polling with an
        ``after=`` cursor keeps following the same run across a resume.
        """
        events: list[dict[str, Any]] = []
        malformed = 0
        for path in self.log_files(run_id):
            written, bad = _read_jsonl(path)
            events.extend(written)
            malformed += bad
        start = max(0, int(after))
        return events[start:], len(events), malformed

    def tail_output(self, run_id: str, limit: int = 40) -> str:
        """The last *limit* lines the run itself printed.

        A child can die before it writes any event (a configuration error at
        import time is the common case), and then the event log is empty while
        the reason is sitting in stderr. This is what lets the page answer
        "why did it stop?" with the run's own words instead of a guess.
        """
        try:
            directory = self.dir_for(run_id)
        except ValueError:
            return ""
        for name in ("stderr.log", "stdout.log"):
            try:
                text = (directory / name).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = [line.rstrip() for line in text.splitlines() if line.strip()]
            if lines:
                return "\n".join(lines[-limit:])
        return ""


# ---------------------------------------------------------------------------
# skills: no store — skills live as skill.md files under skills/<pair>/ and
# nothing in the server reads them.
# ---------------------------------------------------------------------------
# Glossaries
# ---------------------------------------------------------------------------
#: Glossaries are plain CSV: one ``src,tgt`` pair per row.
GLOSSARY_SUFFIX = ".csv"


class GlossaryStore:
    """The CSV glossaries the UI manages, under one directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, name: str) -> Path:
        """Resolve *name* to a CSV path, refusing anything that escapes *root*.

        Public because a caller must be able to tell "you may not name that"
        (``ValueError``) from "that glossary does not exist" (a missing file) —
        a 403 and a 404 are different messages to the user.
        """
        filename = name if name.lower().endswith(GLOSSARY_SUFFIX) else f"{name}{GLOSSARY_SUFFIX}"
        return safe_child(self.root, filename)

    def list_files(self) -> list[dict[str, Any]]:
        """Each glossary with its row count, ordered by name."""
        if not self.root.is_dir():
            return []
        return [
            {"name": path.name, "entries": len(self.read(path.name))}
            for path in sorted(self.root.glob(f"*{GLOSSARY_SUFFIX}"))
            if path.is_file()
        ]

    def read(self, name: str) -> list[list[str]]:
        """Rows of a glossary; empty when the file does not exist.

        Reads with ``utf-8-sig`` so a byte-order mark written by a spreadsheet
        does not become part of the first term.
        """
        path = self.path_for(name)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return [[cell.strip() for cell in row] for row in csv.reader(handle) if row]
        except OSError:
            return []

    def write(self, name: str, pairs: list[list[str]]) -> None:
        """Replace a glossary with *pairs*, written atomically."""
        path = self.path_for(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for row in pairs:
                writer.writerow([str(cell) for cell in row])
        os.replace(tmp, path)
