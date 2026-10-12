"""Event sink: JSONL run log plus a bounded in-memory event tail.

Every step emits an event ``{ts, step, chunk_id, status, latency, error}``
that is appended to ``logs/run_<ts>.jsonl`` and kept in a ring buffer for
inspection. Records that are not steps — ``run_start``, ``decision``,
``error``, ``run_end``, ``run_note`` — go to the same file through
:meth:`EventSink.record`; the reader for both lives in
:mod:`polyglot_lc.ui.trace`.

Every record carries a ``run_id``, so two runs sharing one ``LOGS_DIR`` still
separate on read (their file names only resolve to the second).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from polyglot_lc.ui.trace import mint_run_id


class Event(BaseModel):
    """One pipeline event (log line + UI row)."""

    ts: float = Field(default_factory=time.time)
    step: str
    chunk_id: str = ""
    status: str = "ok"  # start | ok | fail | retry | info
    latency: float = 0.0  # ms
    error: str = ""
    # --- structured trace (all optional: older logs have none of them) ---
    run_id: str = ""
    type: str = "step"
    duration_ms: int | None = None
    #: Token delta for this step, ``{sent, received, cached}``. Approximate when
    #: chunks run concurrently: ``TokenUsage`` is a process-wide counter.
    tokens: dict[str, int] | None = None
    cause: str = ""

    def to_json(self) -> str:
        return self.model_dump_json()


class EventSink:
    """Writes JSONL and keeps a bounded in-memory tail of recent events."""

    def __init__(
        self, logs_dir: str | Path, enabled: bool = True, run_id: str | None = None
    ) -> None:
        self.enabled = enabled
        # An explicit id wins; then the one the parent process exported (the web
        # UI does); then one minted here, in the same shape the web UI mints.
        self.run_id = run_id or os.environ.get("RUN_ID", "").strip() or mint_run_id()
        #: How many problems this run recorded, so the final record can report a
        #: count that reconciles with the problems a reader finds in the file.
        self.problem_count = 0
        self.path: Path | None = None
        self._tail: deque[Event] = deque(maxlen=400)
        # Concurrent chunks emit events from multiple threads.
        self._lock = threading.Lock()
        if enabled:
            logs = Path(logs_dir)
            logs.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.path = logs / f"run_{ts}.jsonl"

    def record(self, **fields: Any) -> None:
        """Append one typed record: ``{"ts", "run_id", **fields}``.

        Every record carries the keys a step event has always carried
        (``step``, ``chunk_id``, ``status``, ``latency``, ``error``), so a reader
        written before the richer records existed still sees the shape it knows
        and never has to guard a ``KeyError``.

        A silent no-op when logging is off, so a caller never has to check:
        ``LOG_JSONL=0`` means "this run leaves no trace", not "this run
        crashes the first time it wants to say something".
        """
        if not self.enabled or self.path is None:
            return
        payload: dict[str, Any] = {
            "ts": time.time(),
            "run_id": self.run_id,
            "step": "",
            "chunk_id": "",
            "status": "ok",
            "latency": 0.0,
            "error": "",
            **fields,
        }
        if payload.get("type") == "error":
            self.problem_count += 1
            # An error record *is* a failure: say so in the old vocabulary too,
            # so the web UI's existing ``fail``/``retry`` filter finds it.
            if payload["status"] == "ok":
                payload["status"] = "fail"
            if not payload["error"]:
                payload["error"] = str(payload.get("message") or "")
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Convenience -------------------------------------------------------
    def emit(self, event: Event) -> None:
        if not event.run_id:
            # Stamp the run id here rather than at every construction site.
            event.run_id = self.run_id
        if event.status in {"fail", "retry"}:
            self.problem_count += 1
        with self._lock:
            self._tail.append(event)
            if self.enabled and self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")

    # Convenience -------------------------------------------------------
    def info(self, step: str, chunk_id: str = "", message: str = "") -> None:
        self.emit(Event(step=step, chunk_id=chunk_id, status="info", error=message))

    def fail(self, step: str, chunk_id: str = "", error: str = "") -> None:
        self.emit(Event(step=step, chunk_id=chunk_id, status="fail", error=error))

    def tail(self, n: int = 12) -> Iterator[Event]:
        """The last *n* events (inspection helper)."""
        return list(self._tail)[-n:].__iter__()

    def counts(self) -> dict[str, int]:
        """Aggregate counts per status (inspection helper)."""
        out = {"ok": 0, "fail": 0, "retry": 0}
        for e in self._tail:
            if e.status in out:
                out[e.status] += 1
        return out
