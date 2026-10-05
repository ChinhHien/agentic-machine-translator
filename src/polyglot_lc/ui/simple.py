"""Simple, non-dynamic pipeline UI.

Replaces the previous Rich ``Live`` dashboard (spinner, progress bar, redraw
panels). It only prints the step that is currently running and keeps counters
for the final report — no animations, no live redraw. Full event logging still
goes to ``logs/run_<ts>.jsonl`` via :class:`EventSink` for debugging.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from rich.console import Console

from polyglot_lc.ui.events import Event, EventSink


class SimpleStep:
    """Context manager that prints a single ``Running ...`` line per step."""

    def __init__(self, ui: "SimpleUI", name: str, chunk_id: str, index: int) -> None:
        self.ui = ui
        self.name = name
        self.chunk_id = chunk_id
        self.index = index
        self._t0 = 0.0

    # kept for API compatibility with the old LiveUI step handle
    def retry(self, error: str) -> None:
        self.ui.retries += 1
        self.ui.add_error(f"retry {self.name} {self.chunk_id}: {error}".strip())
        self.ui.sink.emit(
            Event(step=self.name, chunk_id=self.chunk_id, status="retry", error=error)
        )

    def __enter__(self) -> "SimpleStep":
        self._t0 = time.monotonic()
        self.ui._print_step(self.index, self.name, self.chunk_id)
        self.ui.sink.emit(Event(step=self.name, chunk_id=self.chunk_id, status="start"))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        latency = (time.monotonic() - self._t0) * 1000
        if exc is None:
            self.ui.sink.emit(
                Event(step=self.name, chunk_id=self.chunk_id, status="ok",
                      latency=round(latency, 1))
            )
            return False
        short = f"{self.name} {self.chunk_id}: {exc_type.__name__}: {exc}".strip()
        self.ui.add_error(short)
        self.ui.sink.emit(
            Event(step=self.name, chunk_id=self.chunk_id, status="fail",
                  latency=round(latency, 1), error=short)
        )
        return False  # propagate


class SimpleUI:
    """Plain-text step reporter + counters.

    Usage::

        with SimpleUI(sink) as ui:
            with ui.step("translate", "chunk-000"):
                ...
    """

    def __init__(
        self,
        sink: EventSink,
        console: Console | None = None,
        quiet: bool = False,
        verbose: bool = True,
    ) -> None:
        self.sink = sink
        self.console = console or Console(highlight=False)
        self.quiet = quiet
        self.verbose = verbose and not quiet
        self.errors: list[str] = []
        self.retries = 0
        self.stats: dict[str, Any] = {}
        self._index = 0
        self._total_chunks = 0
        self._done_chunks = 0
        self._t0 = time.monotonic()
        # Chunks run concurrently; keep step numbering and output lines atomic.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _print_step(self, index: int, name: str, chunk_id: str) -> None:
        if not self.verbose:
            return
        label = f"[{index}] Running {name}"
        if chunk_id:
            label += f" ({chunk_id})"
        with self._lock:
            self.console.print(f"  {label} ...", highlight=False, markup=False)

    def add_error(self, message: str) -> None:
        if message:
            self.errors.append(message)

    def set_total_chunks(self, n: int) -> None:
        self._total_chunks = n

    def advance(self, n: int = 1) -> None:
        self._done_chunks += n

    def set_stats(self, **kv: Any) -> None:
        self.stats.update(kv)

    def step(self, name: str, chunk_id: str = "") -> SimpleStep:
        with self._lock:
            self._index += 1
            index = self._index
        return SimpleStep(self, name, chunk_id, index)

    # ------------------------------------------------------------------
    def __enter__(self) -> "SimpleUI":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None
