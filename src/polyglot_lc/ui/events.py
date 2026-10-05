"""Event sink: parallel JSONL log + in-memory stream for the Rich UI.

Every step emits an event ``{ts, step, chunk_id, status, latency, error}``
that is appended to ``logs/run_<ts>.jsonl`` and fanned out to subscribers
(the Live panel tail).
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Iterator

from pydantic import BaseModel, Field


class Event(BaseModel):
    """One pipeline event (log line + UI row)."""

    ts: float = Field(default_factory=time.time)
    step: str
    chunk_id: str = ""
    status: str = "ok"  # start | ok | fail | retry | info
    latency: float = 0.0  # ms
    error: str = ""

    def to_json(self) -> str:
        return self.model_dump_json()


class EventSink:
    """Writes JSONL and keeps a bounded in-memory tail for the UI."""

    def __init__(self, logs_dir: str | Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path: Path | None = None
        self._tail: Deque[Event] = deque(maxlen=400)
        self._subs: list[Callable[[Event], None]] = []
        # Concurrent chunks emit events from multiple threads.
        self._lock = threading.Lock()
        if enabled:
            logs = Path(logs_dir)
            logs.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.path = logs / f"run_{ts}.jsonl"

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._subs.append(fn)

    def emit(self, event: Event) -> None:
        with self._lock:
            self._tail.append(event)
            subs = list(self._subs)
            if self.enabled and self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")
        for fn in subs:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - UI must never break the pipeline
                pass

    # Convenience -------------------------------------------------------
    def info(self, step: str, chunk_id: str = "", message: str = "") -> None:
        self.emit(Event(step=step, chunk_id=chunk_id, status="info", error=message))

    def fail(self, step: str, chunk_id: str = "", error: str = "") -> None:
        self.emit(Event(step=step, chunk_id=chunk_id, status="fail", error=error))

    def tail(self, n: int = 12) -> Iterator[Event]:
        """The last *n* events (for the streaming log panel)."""
        return list(self._tail)[-n:].__iter__()

    def counts(self) -> dict[str, int]:
        """Aggregate counts per status (for the metrics panel)."""
        out = {"ok": 0, "fail": 0, "retry": 0}
        for e in self._tail:
            if e.status in out:
                out[e.status] += 1
        return out
