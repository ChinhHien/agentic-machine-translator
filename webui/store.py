"""Stores for the web UI: configuration now, runs/skills/glossaries later.

Everything here is a plain file reader meant to be called from the server
process, so nothing in this module imports :mod:`polyglot_lc` (see the package
docstring in :mod:`webui`).
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from webui.paths import safe_child

#: Settings the UI may change; values are written into `.env`.
EDITABLE_KEYS: tuple[str, ...] = (
    "CHUNK_MAX_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "MAX_CONCURRENCY",
    "CRITIC_THRESHOLD",
    "REFINE_ROUNDS",
    "REFINE_MIN_SCORE",
    "RB_TOP_K",
    "UNIFIED_PLANNER",
    "ADAPTIVE_QUALITY",
    "FAST_PATH_MAX_COMPLEXITY",
    "FAST_PATH_MIN_SCORE",
    "FAST_PATH_SKIP_LLM_VERIFY",
    "CONSISTENCY_CHECK",
    "CONSISTENCY_MAX_PATCHES",
    "PARTIAL_OUTPUT",
    "RESUME",
    "PDF_EXTRACTOR",
    "LLM_MOCK",
)

#: Settings the UI only displays. Repointing a path from the UI would send a
#: later run's output somewhere unexpected, for no benefit the UI can explain.
READONLY_KEYS: tuple[str, ...] = (
    "LOGS_DIR",
    "SKILLS_DIR",
    "REASONING_BANK_PATH",
    "CHECKPOINT_DIR",
)

#: Credentials the project documents in `.env.example`. The UI reports only
#: whether each one is set, so a user can see why a run would fail while no
#: value ever enters the server process. `tests/test_webui_store.py` fails when
#: `.env.example` grows a credential this list does not know about.
SECRET_KEYS: tuple[str, ...] = (
    "TRANSLATOR_API_KEY",
    "TRANSLATOR_BACKUP_API_KEY",
    "PLANNER_API_KEY",
    "VERIFIER_API_KEY",
    "CRITIC_API_KEY",
    "REFINER_API_KEY",
    "SOURCE_GROUP_API_KEY",
    "EMBED_API_KEY",
)

#: Name fragments marking a value the server must never read or return.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")

#: A `NAME = value` (or bare `NAME`) assignment line.
_KEY_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")


def is_secret(name: str) -> bool:
    """True when *name* looks like a credential the UI must not touch.

    Markers are matched as whole underscore-separated segments, or as a suffix
    of the collapsed name, rather than as plain substrings. Plain substring
    matching flags ``CHUNK_MAX_TOKENS`` (it contains ``TOKEN``) and so would
    declare a token *budget* a credential — contradicting the spec, which lists
    that setting as editable. The direction is still conservative: a name is
    secret when it has any doubt, so ``x_token`` and ``OPENAI_APIKEY`` are both
    caught.
    """
    upper = name.upper()
    if any(segment in _SECRET_MARKERS for segment in upper.split("_")):
        return True
    collapsed = upper.replace("_", "")
    return any(collapsed.endswith(marker) for marker in _SECRET_MARKERS)


class ConfigStore:
    """Reads and writes `.env` without ever touching a secret's value.

    ``env_keys`` returns the *names* present in the file (never values), and
    ``save`` rewrites exactly the lines for the keys it was asked to change,
    leaving comments, blank lines and every other setting byte-for-byte alone.
    Writes go through a temporary file in the same directory so a reader can
    never see a half-written `.env`.
    """

    def __init__(self, root: Path, env_path: Path) -> None:
        self.root = root
        self.env_path = env_path

    def _read_text(self) -> str:
        """The file's exact text, with its own line endings preserved.

        ``Path.read_text`` uses universal newlines, which silently rewrites
        CRLF to LF on the way in; the style is then lost and ``save`` would
        write the whole file with the wrong ending. Reading with ``newline=""``
        keeps the bytes as they are.
        """
        try:
            with self.env_path.open("r", encoding="utf-8", newline="") as handle:
                return handle.read()
        except FileNotFoundError:
            return ""
        except OSError:  # pragma: no cover - unreadable or exotic file
            return ""

    def env_keys(self) -> set[str]:
        """Names of the settings assigned in `.env` — never their values."""
        return {
            match.group(1)
            for line in self._read_text().splitlines()
            if (match := _KEY_LINE_RE.match(line))
        }

    def secret_status(self) -> dict[str, str]:
        """``"set"``/``"unset"`` per credential name — never a value.

        Names come from :data:`SECRET_KEYS` plus anything secret-shaped already
        in `.env`, so a credential the user added by hand also shows up. The
        process environment is consulted **by name only**, which is why a key
        exported in the shell reads "set" even when `.env` never mentions it —
        exactly the priority order the UI has to explain to the user.
        """
        in_dotenv = self.env_keys()
        in_env = set(os.environ)
        names = set(SECRET_KEYS) | {name for name in in_dotenv if is_secret(name)}
        return {
            name: ("set" if name in in_dotenv or name in in_env else "unset")
            for name in sorted(names)
        }

    @staticmethod
    def _render(value: Any) -> str:
        """Render a value the way `python-dotenv` reads it back."""
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    def save(self, values: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
        """Apply an editable whitelist of *values*; return (saved, rejected)."""
        saved: list[str] = []
        rejected: list[dict[str, str]] = []
        text = self._read_text()
        newline = "\r\n" if "\r\n" in text else "\n"
        lines = text.splitlines()
        for key, value in values.items():
            # Secrets are refused before the whitelist check, so a credential
            # can never be written even if a whitelist entry were added wrongly.
            if is_secret(key):
                rejected.append({"key": key, "reason": "secret"})
                continue
            if key not in EDITABLE_KEYS:
                rejected.append({"key": key, "reason": "not editable"})
                continue
            rendered = f"{key}={self._render(value)}"
            for index, line in enumerate(lines):
                match = _KEY_LINE_RE.match(line)
                if match is not None and match.group(1) == key:
                    lines[index] = rendered
                    break
            else:
                lines.append(rendered)
            saved.append(key)
        if lines:
            self._write_atomic(newline.join(lines) + newline)
        return saved, rejected

    def _write_atomic(self, body: str) -> None:
        """Write *body* through a temp file so readers never see it half-written."""
        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.env_path.with_name(self.env_path.name + ".tmp")
        tmp.write_text(body, encoding="utf-8", newline="")
        os.replace(tmp, self.env_path)


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
    domain: str
    chunks: int
    exit_code: int | None
    output_path: str
    mock: bool
    pid: int | None
    command: list[str]
    note: str = ""

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
            domain=str(data.get("domain", "")),
            chunks=int(data.get("chunks") or 0),
            exit_code=None if raw_exit is None else int(raw_exit),
            output_path=str(data.get("output_path", "")),
            mock=bool(data.get("mock", False)),
            pid=None if raw_pid is None else int(raw_pid),
            command=[str(x) for x in (data.get("command") or [])],
            note=str(data.get("note") or ""),
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

    def log_path(self, run_id: str) -> Path | None:
        """The run's event log, or ``None`` when the run wrote none."""
        try:
            directory = self.dir_for(run_id)
        except ValueError:
            return None
        if not directory.is_dir():
            return None
        logs = sorted(directory.glob("run_*.jsonl"))
        return logs[-1] if logs else None

    def read_events(self, run_id: str, after: int) -> tuple[list[dict[str, Any]], int, int]:
        """Events from index *after* on, plus ``(next_index, malformed_count)``.

        Indexing is by position in the usable event list rather than by
        timestamp, so a poll does not depend on any clock and cannot lose an
        event to a timestamp tie.
        """
        path = self.log_path(run_id)
        if path is None:
            return [], 0, 0
        events, malformed = _read_jsonl(path)
        start = max(0, int(after))
        return events[start:], len(events), malformed


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------
#: Suffix of the per-pair score history the optimizer appends to.
HISTORY_SUFFIX = ".history.jsonl"


class SkillsStore:
    """The promoted skills under ``skills/<src>-<tgt>/<domain>.json``.

    The directory also holds files this view has no business reading (a
    ``session_*.json`` scratch file, a half-written entry), so anything that is
    not a readable skill object is skipped rather than reported as a failure.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def _pair_dir(self, pair: str) -> Path:
        return safe_child(self.root, pair)

    def _skill_path(self, pair: str, domain: str) -> Path:
        return safe_child(self._pair_dir(pair), f"{domain or 'general'}.json")

    def _history_path(self, pair: str, domain: str) -> Path:
        return safe_child(self._pair_dir(pair), f"{domain or 'general'}{HISTORY_SUFFIX}")

    def list_skills(self) -> list[dict[str, Any]]:
        """One summary per readable skill, ordered by pair then domain."""
        if not self.root.is_dir():
            return []
        found: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*/*.json")):
            if path.name.endswith(HISTORY_SUFFIX) or path.name.startswith("session_"):
                continue
            if not path.is_file():
                continue
            data = _read_json(path)
            if not isinstance(data, dict):
                continue
            found.append({
                "pair": path.parent.name,
                "domain": path.stem,
                "mean_score": data.get("mean_score"),
                "promoted_at": data.get("promoted_at"),
                "updated_at": data.get("updated_at"),
                "instruction": str(data.get("instruction", "")),
                "src_lang": str(data.get("src_lang", "")),
                "tgt_lang": str(data.get("tgt_lang", "")),
            })
        found.sort(key=lambda skill: (skill["pair"], skill["domain"]))
        return found

    def get_skill(self, pair: str, domain: str) -> dict[str, Any] | None:
        """The full skill object, or ``None`` when absent or unreadable."""
        try:
            path = self._skill_path(pair, domain)
        except ValueError:
            return None
        data = _read_json(path)
        return dict(data) if isinstance(data, dict) else None

    def history(self, pair: str, domain: str) -> list[dict[str, Any]]:
        """The pair/domain score history, oldest first."""
        try:
            path = self._history_path(pair, domain)
        except ValueError:
            return []
        entries, _ = _read_jsonl(path)
        return entries


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
                return [
                    [cell.strip() for cell in row]
                    for row in csv.reader(handle)
                    if row
                ]
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
