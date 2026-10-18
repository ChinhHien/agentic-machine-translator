"""Simple, non-dynamic pipeline UI.

Replaces the previous Rich ``Live`` dashboard (spinner, progress bar, redraw
panels). It only prints the step that is currently running and keeps counters
for the final report — no animations, no live redraw. Full event logging still
goes to ``logs/run_<ts>.jsonl`` via :class:`EventSink` for debugging.

Each step writes a ``step_start`` and a ``step_end`` record carrying its
``duration_ms``, its token delta and — when it failed — a cause from
:mod:`polyglot_lc.ui.trace`'s closed set.

**The token delta is approximate when ``MAX_CONCURRENCY > 1``**: it is measured
against :class:`~polyglot_lc.llm.client.TokenUsage`, a process-wide counter, so
two chunks in flight at once can charge each other's tokens. Exact per-step
attribution needs thread-local accounting (sub-project E); nothing downstream
should treat these numbers as audited.
"""

from __future__ import annotations

import threading
import time
from types import TracebackType
from typing import Any, Self

from rich.console import Console

from polyglot_lc.llm.client import TokenUsage, active_api_calls
from polyglot_lc.ui.events import Event, EventSink
from polyglot_lc.ui.trace import cause_for


def _token_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """What one step spent, floored at zero (see the module docstring)."""
    return {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        for key in ("sent", "received", "cached")
    }


class SimpleStep:
    """Context manager that prints a single ``Running ...`` line per step."""

    def __init__(self, ui: SimpleUI, name: str, chunk_id: str, index: int) -> None:
        self.ui = ui
        self.name = name
        self.chunk_id = chunk_id
        self.index = index
        self._t0 = 0.0
        self._tokens0: dict[str, int] = {}

    def __enter__(self) -> Self:
        self._t0 = time.monotonic()
        self._tokens0 = TokenUsage.snapshot()
        self.ui._print_step(self.index, self.name, self.chunk_id)
        calls = active_api_calls()
        self.ui.sink.emit(
            Event(
                step=self.name,
                chunk_id=self.chunk_id,
                status="start",
                type="step_start",
                active_api_calls=calls.get("active"),
                max_active_api_calls=calls.get("peak"),
            )
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        latency = (time.monotonic() - self._t0) * 1000
        tokens = _token_delta(self._tokens0, TokenUsage.snapshot())
        if exc is None:
            self.ui.sink.emit(
                Event(
                    step=self.name,
                    chunk_id=self.chunk_id,
                    status="ok",
                    latency=round(latency, 1),
                    type="step_end",
                    duration_ms=round(latency),
                    tokens=tokens,
                )
            )
            return False
        short = f"{self.name} {self.chunk_id}: {exc_type.__name__}: {exc}".strip()
        self.ui.add_error(short)
        self.ui.sink.emit(
            Event(
                step=self.name,
                chunk_id=self.chunk_id,
                status="fail",
                latency=round(latency, 1),
                error=short,
                type="step_end",
                duration_ms=round(latency),
                tokens=tokens,
                cause=cause_for(exc, step=self.name),
            )
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

    def retry(self, step: str, chunk_id: str, message: str) -> None:
        """Say that a chunk is being waited for and will run again.

        A wait measured in minutes must not look like a hang, so it is printed
        and written as a ``retry`` event — the status the web UI reads to keep
        the chunk out of its "failed" state and to caption the progress bar.
        """
        where = f"{step} ({chunk_id})".strip()
        if self.verbose:
            with self._lock:
                self.console.print(f"  Waiting: {where} - {message}", highlight=False, markup=False)
        calls = active_api_calls()
        self.sink.emit(
            Event(
                step=step,
                chunk_id=chunk_id,
                status="retry",
                error=message,
                type="retry_wait",
                active_api_calls=calls.get("active"),
                max_active_api_calls=calls.get("peak"),
            )
        )

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
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None
