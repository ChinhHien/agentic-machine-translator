"""Standalone ``step`` context manager (works with or without a UI).

Provides ``with step(name, chunk_id):`` semantics; when a UI (SimpleUI or any
object exposing ``.step``) is given it delegates to it; otherwise it falls
back to JSONL logging only.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from polyglot_lc.ui.events import Event, EventSink


@contextmanager
def step(
    name: str,
    chunk_id: str = "",
    ui: Any = None,
    sink: EventSink | None = None,
) -> Iterator[Any]:
    """Log start/ok/fail around a block; UI step when a ``ui`` is given."""
    if ui is not None:
        with ui.step(name, chunk_id) as handle:
            yield handle
        return
    sink = sink or EventSink(logs_dir="logs", enabled=False)
    import time

    sink.emit(Event(step=name, chunk_id=chunk_id, status="start"))
    t0 = time.monotonic()
    try:
        yield None
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        sink.emit(
            Event(step=name, chunk_id=chunk_id, status="fail",
                  latency=round(latency, 1), error=f"{type(exc).__name__}: {exc}")
        )
        raise
    else:
        latency = (time.monotonic() - t0) * 1000
        sink.emit(
            Event(step=name, chunk_id=chunk_id, status="ok",
                  latency=round(latency, 1))
        )


__all__ = ["step"]
