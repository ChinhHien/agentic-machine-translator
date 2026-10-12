"""Structured trace: the record vocabulary and the reader behind ``polyglot trace``.

One translation run appends *records* to a single JSONL file inside its
``LOGS_DIR``. Steps were already written there as progress events; this module
owns everything the newer, richer records need and nothing else:

* :data:`CAUSES` — the **closed** set of error causes, and :func:`cause_for`,
  which maps an exception the resilience layer already classified onto one of
  them. The taxonomy is not re-derived from the message text: a wrong cause is
  worse than an honest ``unknown``.
* :func:`plan_hash` — the chunk plan's hash, which is what tells two runs of the
  same document with a *different* split apart (``sha256`` cannot).
* :func:`mint_run_id` — the run id shape the web UI mints, so both sides agree.
* The reader: :func:`read_trace`, :func:`resolve_run` and :func:`load_trace`,
  which turn the file back into settings, a per-chunk timeline, the problems
  with their causes, and the totals.

The records are **metadata only**: no document text and no translation is ever
written, by this module or by any caller. Reason strings are built from numbers.

Per-step token counts are a *delta against a process-wide counter*, so they are
approximate when ``MAX_CONCURRENCY > 1``; exact attribution is sub-project E.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polyglot_lc.config import ConfigError
from polyglot_lc.llm.resilience import (
    PermanentError,
    RateLimitError,
    TransientError,
    is_concurrency_limit,
)

__all__ = [
    "CAUSES",
    "ChunkTimeline",
    "StepRecord",
    "Trace",
    "cause_for",
    "load_trace",
    "mint_run_id",
    "plan_hash",
    "read_trace",
    "resolve_run",
]

#: The closed set of error causes (spec section 6). Every ``cause`` written to a
#: trace, and every ``cause`` a reader trusts, comes from here.
CAUSES: frozenset[str] = frozenset(
    {
        "rate_limit",
        "concurrency",
        "timeout",
        "auth",
        "provider",
        "parse",
        "verify",
        "placeholder",
        "config",
        "input",
        "unknown",
    }
)

#: HTTP statuses that mean "your credentials are the problem", not the request.
_AUTH_STATUS = frozenset({401, 403})

#: Record type for a line that carries none. Older logs wrote bare step events,
#: so a missing type has to read as a step rather than as unreadable.
DEFAULT_TYPE = "step"

#: Statuses that mean something went wrong.
FAILURE_STATUSES = frozenset({"fail", "retry"})


def cause_for(exc: BaseException, *, step: str = "") -> str:
    """The closed-set cause for *exc*, from the taxonomy the caller already has.

    *step* disambiguates the one case the exception type cannot: an unclassified
    failure while reading the input file is an ``input`` problem, and the same
    exception during translation is ``unknown``.
    """
    if isinstance(exc, (RateLimitError, TransientError)) and is_concurrency_limit(exc):
        # Also a 429, but a different problem: a retry collides with the request
        # still in flight rather than fixing anything.
        return "concurrency"
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, (TransientError, TimeoutError)):
        return "timeout"
    if isinstance(exc, PermanentError):
        return "auth" if exc.status in _AUTH_STATUS else "provider"
    if isinstance(exc, ConfigError):
        return "config"
    if isinstance(exc, json.JSONDecodeError):
        return "parse"
    if step == "ingest":
        return "input"
    return "unknown"


def plan_hash(chunks: Sequence[Any]) -> str:
    """Hash of the chunk plan — same document, same split, same hash.

    Uses the ``Chunk.hash()`` each chunk already carries, so a chunk whose text
    changed changes the hash even when its id did not. An empty plan hashes to
    ``""`` rather than to the hash of nothing.
    """
    if not chunks:
        return ""
    joined = "\n".join(f"{chunk.id}:{chunk.hash()}" for chunk in chunks)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def mint_run_id() -> str:
    """A new run id in the shape the web UI mints: ``YYYYMMDD_HHMMSS_<6 hex>``."""
    return time.strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(3).hex()


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepRecord:
    """One step of one chunk, as recorded."""

    name: str
    status: str = "ok"
    duration_ms: int | None = None
    tokens: dict[str, int] = field(default_factory=dict)
    error: str = ""
    cause: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "tokens": dict(self.tokens),
            "error": self.error,
            "cause": self.cause,
        }

    def as_text(self) -> str:
        timing = f"{self.duration_ms} ms" if self.duration_ms is not None else "?"
        tok = self.tokens or {}
        spent = ""
        if tok:
            spent = (
                f"  tokens {tok.get('sent', 0)}/{tok.get('received', 0)}"
                f" (+{tok.get('cached', 0)} cached)"
            )
        line = f"      {self.name}: {self.status} ({timing}){spent}"
        if self.error:
            line += f"\n        {self.cause or 'unknown'}: {self.error}"
        return line


@dataclass
class ChunkTimeline:
    """Everything recorded about one chunk: its steps, its decisions, its problems."""

    chunk_id: str
    steps: list[StepRecord] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    problems: list[dict[str, Any]] = field(default_factory=list)

    @property
    def route(self) -> str:
        """``fast`` | ``deep`` | ``standard``, derived from the decisions.

        The graph records *why* it branched, not a final route label, so the
        route is reconstructed the same way the graph decides it: an accepted
        fast-path gate means fast, any accepted refine means deep, everything
        else is the standard path.
        """
        for decision in self.decisions:
            if decision.get("kind") == "fast_path":
                if decision.get("chosen"):
                    return "fast"
                break
        if any(d.get("kind") == "refine_gate" and d.get("chosen") for d in self.decisions):
            return "deep"
        # Anything that recorded a decision or a step went through the graph;
        # a chunk with neither was never reached.
        return "standard" if (self.decisions or self.steps) else ""

    @property
    def reasons(self) -> list[str]:
        """The routing reasons, in the order they were recorded."""
        out: list[str] = []
        for decision in self.decisions:
            if decision.get("kind") in {"fast_path", "refine_gate", "retranslate", "verify_mode"}:
                out.extend(str(r) for r in decision.get("reasons") or [])
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "route": self.route,
            "reasons": self.reasons,
            "steps": [s.as_dict() for s in self.steps],
            "decisions": self.decisions,
            "problems": self.problems,
        }

    def as_text(self) -> str:
        reasons = self.reasons
        label = f"  {self.chunk_id or '(document)'}"
        if self.route:
            label += f"  route={self.route}"
        lines = [label]
        if reasons:
            lines.append("      why: " + " | ".join(reasons))
        lines.extend(step.as_text() for step in self.steps if step.status != "start")
        for problem in self.problems:
            lines.append(
                f"      problem [{problem.get('cause') or 'unknown'}]"
                f" {problem.get('step', '')}: {problem.get('message', '')}"
            )
        return "\n".join(lines)


@dataclass
class Trace:
    """One run, read back: what it was asked to do, and what it did."""

    path: Path | None = None
    run_id: str = ""
    started: dict[str, Any] = field(default_factory=dict)
    ended: dict[str, Any] = field(default_factory=dict)
    timeline: list[ChunkTimeline] = field(default_factory=list)
    problems: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    malformed: int = 0

    @property
    def totals(self) -> dict[str, Any]:
        """The run's totals, taken from ``run_end`` minus its envelope keys."""
        return {k: v for k, v in self.ended.items() if k not in {"type", "ts", "run_id"}}

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": str(self.path) if self.path else "",
            "started": self.started,
            "ended": self.ended,
            "timeline": [chunk.as_dict() for chunk in self.timeline],
            "problems": self.problems,
            "notes": self.notes,
            "totals": self.totals,
            "malformed": self.malformed,
        }

    def as_text(self) -> str:
        if not self.started and not self.timeline and not self.problems:
            return (
                f"No records in {self.path}: this run left no trace. It may have died "
                "before its first record, or logging was off for it."
            )
        lines: list[str] = [f"Run {self.run_id or '(unknown id)'}  —  {self.path}", ""]
        started = self.started
        if not started:
            lines.append("No run_start record: the settings this run used were never written.")
        else:
            lines.append("Settings")
            lines.append(
                f"  file: {started.get('file', '?')}  ({started.get('bytes', 0):,} bytes, "
                f"{started.get('pages') if started.get('pages') is not None else 'no'} pages)"
            )
            lines.append(
                f"  sha256: {started.get('sha256', '?')}   plan: {started.get('plan_hash') or '?'}"
                f"   chunks: {started.get('chunks', '?')}"
            )
            lines.append(
                f"  {started.get('src', '?')} -> {started.get('tgt', '?')}"
                f"  domain: {started.get('domain', '?')}"
                f"  max tokens: {started.get('max_tokens', '?')}"
            )
            for agent in started.get("agents") or []:
                lines.append(
                    f"  {agent.get('name', '?')}: {agent.get('model', '?')} @ "
                    f"{agent.get('host') or '?'} (temperature {agent.get('temperature', '?')})"
                )
        if self.timeline:
            lines.append("")
            lines.append("Chunks")
            lines.extend(chunk.as_text() for chunk in self.timeline)
        if self.problems:
            lines.append("")
            lines.append(f"Problems ({len(self.problems)})")
            for problem in self.problems:
                where = problem.get("chunk_id") or problem.get("step") or "run"
                lines.append(
                    f"  [{problem.get('cause') or 'unknown'}] {where}: {problem.get('message', '')}"
                )
        if self.notes:
            lines.append("")
            lines.append("Notes")
            lines.extend(f"  {note}" for note in self.notes)
        if self.ended:
            totals = self.totals
            tokens = totals.get("tokens") or {}
            lines.append("")
            lines.append("Totals")
            lines.append(
                f"  status: {totals.get('status', '?')}  duration: "
                f"{(totals.get('duration_ms') or 0) / 1000:.1f}s"
                f"  chunks: {totals.get('chunks_done', '?')} done / "
                f"{totals.get('chunks_failed', 0)} failed"
            )
            lines.append(
                f"  tokens: {tokens.get('sent', 0):,} sent / {tokens.get('received', 0):,} received"
                f" ({tokens.get('cached', 0):,} cached)"
                f"  requests: {totals.get('calls', '?')}"
                f"  cache hits: {totals.get('cache_hits', 0)}"
                f"  retries: {totals.get('retries', 0)}"
                f"  fallbacks: {totals.get('fallbacks', 0)}"
            )
            for artifact in totals.get("artifacts") or []:
                lines.append(
                    f"  {artifact.get('kind', '?')}: {artifact.get('path', '?')}"
                    f" ({artifact.get('bytes', 0):,} bytes)"
                )
            rebuild = totals.get("rebuild") or {}
            if rebuild:
                lines.append(
                    f"  pdf rebuild: {rebuild.get('placed', 0)} placed, "
                    f"{rebuild.get('kept', 0)} kept, {len(rebuild.get('failures') or [])} failure(s)"
                )
        else:
            lines.append("")
            lines.append("No run_end record: the run did not finish.")
        if self.malformed:
            lines.append("")
            lines.append(f"({self.malformed} unreadable line(s) skipped)")
        return "\n".join(lines)


def read_trace(path: str | Path) -> Trace:
    """Read a run log into a :class:`Trace`.

    A line that cannot be parsed — a partial write from a chunk still running —
    is counted in ``Trace.malformed`` and skipped; every record before it is
    kept. A missing file is not an error either: the reader returns an empty
    trace and :meth:`Trace.as_text` says so in words.
    """
    p = Path(path)
    trace = Trace(path=p)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        # No file at all (never created, or removed): an empty trace whose id
        # still comes from the name the caller asked about.
        trace.run_id = _run_id_from_name(p)
        return trace

    by_chunk: dict[str, ChunkTimeline] = {}

    def timeline_for(chunk_id: str) -> ChunkTimeline:
        return by_chunk.setdefault(chunk_id, ChunkTimeline(chunk_id=chunk_id))

    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            trace.malformed += 1
            continue
        if not isinstance(record, dict):
            trace.malformed += 1
            continue
        trace.run_id = trace.run_id or str(record.get("run_id") or "")
        kind = str(record.get("type") or DEFAULT_TYPE)
        if kind == "run_start":
            trace.started = record
        elif kind == "run_end":
            trace.ended = record
        elif kind == "run_note":
            trace.notes.append(str(record.get("text") or ""))
        elif kind == "error":
            entry = {
                "step": str(record.get("step") or ""),
                "chunk_id": str(record.get("chunk_id") or ""),
                "cause": str(record.get("cause") or ""),
                "message": str(record.get("message") or record.get("error") or ""),
            }
            trace.problems.append(entry)
            if entry["chunk_id"]:
                timeline_for(entry["chunk_id"]).problems.append(entry)
        elif kind == "decision":
            timeline_for(str(record.get("chunk_id") or "")).decisions.append(record)
        else:
            # step_start / step_end, and the bare step events of older logs.
            timeline_for(str(record.get("chunk_id") or "")).steps.append(
                StepRecord(
                    name=str(record.get("step") or ""),
                    status=str(record.get("status") or ""),
                    duration_ms=record.get("duration_ms"),
                    tokens=dict(record.get("tokens") or {}),
                    error=str(record.get("error") or ""),
                    cause=str(record.get("cause") or ""),
                )
            )
            if str(record.get("status") or "") in FAILURE_STATUSES:
                problem = {
                    "step": str(record.get("step") or ""),
                    "chunk_id": str(record.get("chunk_id") or ""),
                    "cause": str(record.get("cause") or ""),
                    "message": str(record.get("error") or ""),
                }
                trace.problems.append(problem)
                if problem["chunk_id"]:
                    timeline_for(problem["chunk_id"]).problems.append(problem)
    trace.timeline = list(by_chunk.values())
    if not trace.run_id:
        trace.run_id = _run_id_from_name(p)
    return trace


def _run_id_from_name(path: Path) -> str:
    """The run id baked into a log's name or its directory, when no record had one."""
    stem = path.stem
    stem = stem.removeprefix("run_")
    return stem or path.parent.name


def _is_within(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _newness(path: Path) -> tuple[float, str]:
    """Sort key for "the newest log": modification time, then the file name.

    Two runs started in the same clock tick are a real possibility (and Windows
    timestamps are coarse), and the name carries the same timestamp, so the tie
    breaks in the same direction the times would have.
    """
    return (path.stat().st_mtime, path.name)


def resolve_run(target: str | None, root: str | Path) -> Path:
    """The log file *target* names, or the newest log under *root*.

    With no target, the whole tree below *root* is searched for ``run_*.jsonl``
    and the newest wins — the web UI puts each run one level deeper, in its own
    directory, so a non-recursive search would see none of its runs. A target
    may be a path (which must stay inside *root*), a run directory name, or a
    file name.
    """
    root_path = Path(root)
    if target:
        candidate = Path(target)
        if candidate.is_file():
            resolved = candidate.resolve()
            if not _is_within(root_path.resolve(), resolved):
                raise FileNotFoundError(f"{target} is outside the log root {root_path}")
            return resolved
        wanted = Path(target).name
        matches = [
            p
            for p in root_path.rglob("run_*.jsonl")
            if p.parent.name == wanted or p.stem == f"run_{wanted}" or wanted in p.name
        ]
        if matches:
            return max(matches, key=_newness)
        raise FileNotFoundError(f"no run {target!r} under {root_path}")
    candidates = list(root_path.rglob("run_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no run_*.jsonl under {root_path}")
    return max(candidates, key=_newness)


def default_root() -> Path:
    """Where to look for runs: the parent of ``LOGS_DIR``, i.e. ``logs/``.

    The pipeline writes into ``LOGS_DIR`` and the web UI gives each run its own
    directory *inside* it, so the parent is the one root that reaches both.
    """
    logs_dir = os.environ.get("LOGS_DIR", "").strip()
    if logs_dir:
        return Path(logs_dir).parent
    return Path("logs")


def load_trace(target: str | None = None, root: str | Path | None = None) -> Trace:
    """Resolve *target* under *root* (or the default root) and read it."""
    base = Path(root) if root is not None else default_root()
    return read_trace(resolve_run(target, base))
