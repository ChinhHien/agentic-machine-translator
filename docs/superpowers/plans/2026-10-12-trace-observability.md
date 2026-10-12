# Trace Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one translation run explain itself after the fact — what it was asked to do, which route each chunk took and why, what each step cost, and what failed with which cause — by writing a richer, still-metadata-only record to the JSONL file the run already produces.

**Architecture:** `EventSink` (which already owns the file handle and its lock) grows `record()` for the record types that are not steps, and the `SimpleStep` context manager upgrades the two records it already writes. A new module `ui/trace.py` owns three things the rest of the pipeline must not duplicate: the record vocabulary's closed cause set, the record builders, and the reader behind the new `polyglot trace` command. The web UI keeps its own reader, because it may not import the package.

**Tech Stack:** Python 3.11+, pydantic v2 (existing `Event`), typer + rich CLI, LangGraph (existing chunk graph), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-12-trace-observability-design.md`

## Global Constraints

- **Metadata only.** No record may contain document text or a translation, in any field, including reason strings and error messages copied from providers. Lengths, hashes, counts, ids, durations only.
- **No new dependencies.** Standard library plus what `pyproject.toml` already pins.
- **The existing keys survive.** Every step record keeps `{ts, step, chunk_id, status, latency, error}`. New keys are optional with defaults, so an older log on disk stays readable and today's tests keep passing unchanged.
- **The file name stays `run_<ts>.jsonl`** and stays inside the `LOGS_DIR` the pipeline was configured with. The web UI discovers logs by globbing that name.
- **`webui/*.py` must not import `polyglot_lc`** (its `load_dotenv` would poison the child process environment). The web UI reads the same lines with its own parser.
- **`cause` is a closed set** of exactly eleven values (§6 of the spec): `rate_limit`, `concurrency`, `timeout`, `auth`, `provider`, `parse`, `verify`, `placeholder`, `config`, `input`, `unknown`.
- **Gates:** `python -m pytest`, `python -m ruff check .`, `python -m ruff format --check .` all green. Tests run in mock mode (`LLM_MOCK=1`) with no API key.

## Review Focus

Inputs and failure modes the spec does not name. Each gets a test in the task that owns the code:

1. **A partial last line while a chunk is still writing.** The reader polls a file being appended to concurrently; the truncated line must be counted and skipped, never raise, and never lose the records before it.
2. **`LOG_JSONL=0`.** There is no file at all. `record()` must be a silent no-op and `trace` must say so in words, not traceback.
3. **A log written by the previous version** (no `type`, no `run_id`). It must still read as a timeline of steps rather than being discarded as unreadable.
4. **A `trace` target that is not inside the log root**, or a run id that matches nothing. Must exit with a clear message and never read a path outside the root.
5. **A secret-shaped value in the environment plus a document sentence that looks like metadata.** Neither may appear in the file — the canary and no-document-text tests.

---

### Task 1: The cause vocabulary and the two pure helpers

**Files:**
- Create: `src/polyglot_lc/ui/trace.py`
- Test: `tests/test_trace.py`

**Interfaces:**
- Consumes: `polyglot_lc.llm.resilience.{RateLimitError, TransientError, PermanentError, is_concurrency_limit}`; `polyglot_lc.config.ConfigError`; `Chunk.hash()` (`src/polyglot_lc/segment/chunker.py:37`).
- Produces: `CAUSES: frozenset[str]`; `cause_for(exc: BaseException, *, step: str = "") -> str`; `plan_hash(chunks: Sequence[Any]) -> str`; `mint_run_id() -> str`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_trace.py
from polyglot_lc.config import ConfigError
from polyglot_lc.llm.resilience import PermanentError, RateLimitError, TransientError
from polyglot_lc.segment.chunker import Chunk
from polyglot_lc.ui.trace import CAUSES, cause_for, mint_run_id, plan_hash


def test_cause_for_maps_the_taxonomy_the_resilience_layer_already_computes():
    assert cause_for(RateLimitError("429 rate limit exceeded")) == "rate_limit"
    assert cause_for(RateLimitError("only one concurrent request allowed")) == "concurrency"
    assert cause_for(TransientError("read timeout")) == "timeout"
    assert cause_for(PermanentError("401 unauthorized", status=401)) == "auth"
    assert cause_for(PermanentError("422 unprocessable", status=422)) == "provider"


def test_cause_for_falls_back_to_unknown_rather_than_inventing_a_cause():
    assert cause_for(ValueError("something nobody classified")) == "unknown"


def test_every_cause_comes_from_the_closed_set():
    for exc in (
        RateLimitError("x"),
        TransientError("y"),
        PermanentError("z"),
        ConfigError("c"),
        ValueError("v"),
    ):
        assert cause_for(exc) in CAUSES
    assert len(CAUSES) == 11


def test_ingest_step_unclassified_failure_is_an_input_cause():
    # The step name is the only evidence that the file, not the provider, failed.
    assert cause_for(ValueError("cannot open document"), step="ingest") == "input"
    assert cause_for(ValueError("cannot open document"), step="translate") == "unknown"


def test_plan_hash_tracks_the_split_not_the_document():
    a = [Chunk(id="chunk-000", text="alpha", token_count=1, index=0)]
    same = [Chunk(id="chunk-000", text="alpha", token_count=1, index=0)]
    resplit = [
        Chunk(id="chunk-000", text="al", token_count=1, index=0),
        Chunk(id="chunk-001", text="pha", token_count=1, index=1),
    ]
    assert plan_hash(a) == plan_hash(same)
    assert plan_hash(a) != plan_hash(resplit)
    assert len(plan_hash(a)) == 12


def test_mint_run_id_matches_the_shape_the_web_ui_mints():
    import re

    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", mint_run_id())
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_trace.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot_lc.ui.trace'`

- [x] **Step 3: Implement the helpers in `src/polyglot_lc/ui/trace.py`**

`CAUSES` is the eleven-value frozenset from the spec §6 verbatim. `cause_for` checks, in this order, `RateLimitError`/`TransientError` with `is_concurrency_limit` → `concurrency`; then `RateLimitError` → `rate_limit`; `TransientError` (or `TimeoutError`) → `timeout`; `PermanentError` with `status in {401, 403}` → `auth`, other `PermanentError` → `provider`; `ConfigError` → `config`; JSON-decode failures (`json.JSONDecodeError`, or an exception whose message mentions parsing a JSON reply) → `parse`; else `input` when `step == "ingest"`, else `unknown`. Import the resilience classes at module scope; import `ConfigError` from `polyglot_lc.config`.

`plan_hash(chunks)` returns `hashlib.sha256("\n".join(f"{c.id}:{c.hash()}" for c in chunks).encode()).hexdigest()[:12]`, and returns `""` for an empty sequence rather than the hash of nothing.

`mint_run_id()` returns `time.strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(3).hex()`.

- [x] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_trace.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_trace.py src/polyglot_lc/ui/trace.py
git commit -m "feat(trace): add the closed error-cause vocabulary and plan hash"
```

---

### Task 2: Step records carry duration, tokens and cause; `EventSink` learns `record()`

**Files:**
- Modify: `src/polyglot_lc/ui/events.py`
- Modify: `src/polyglot_lc/ui/simple.py`
- Test: `tests/test_ui_events.py`

**Interfaces:**
- Consumes: `cause_for` from Task 1; `TokenUsage.snapshot()` (`src/polyglot_lc/llm/client.py:149`).
- Produces: `Event(run_id="", type="step", duration_ms=None, tokens=None, cause="")`; `EventSink(logs_dir, enabled=True, run_id=None)` with attributes `.run_id` and `.path`; `EventSink.record(**fields) -> None` writing one line of `{"ts", "run_id", **fields}`; `EventSink.resume_run_id -> str`.

- [x] **Step 1: Write the failing tests** (append to `tests/test_ui_events.py`)

```python
def test_step_end_carries_duration_and_the_token_delta_for_that_step(tmp_path):
    """Tokens per step are a delta against the global counter, not a total."""
    TokenUsage.reset()
    sink = EventSink(tmp_path, enabled=True)
    ui = SimpleUI(sink, quiet=True)
    with ui.step("translate", "chunk-000"):
        TokenUsage.add(120, 40, agent="TRANSLATOR", model="m", provider="http://x/v1")
    line = json.loads(sink.path.read_text(encoding="utf-8").splitlines()[-1])
    assert line["status"] == "ok"
    assert line["type"] == "step_end"
    assert line["duration_ms"] >= 0
    assert line["tokens"] == {"sent": 120, "received": 40, "cached": 0}


def test_a_failing_step_records_a_cause_from_the_closed_set(tmp_path):
    from polyglot_lc.llm.resilience import RateLimitError
    from polyglot_lc.ui.trace import CAUSES

    sink = EventSink(tmp_path, enabled=True)
    ui = SimpleUI(sink, quiet=True)
    with pytest.raises(RateLimitError), ui.step("translate", "chunk-000"):
        raise RateLimitError("429 rate limit exceeded")
    line = json.loads(sink.path.read_text(encoding="utf-8").splitlines()[-1])
    assert line["status"] == "fail" and line["cause"] == "rate_limit" and line["cause"] in CAUSES


def test_every_record_carries_the_run_id_and_the_existing_keys_survive(tmp_path):
    sink = EventSink(tmp_path, enabled=True, run_id="20261012_015104_dad6b2")
    sink.record(type="run_note", text="reasoning bank unavailable")
    line = json.loads(sink.path.read_text(encoding="utf-8").splitlines()[-1])
    assert line["run_id"] == "20261012_015104_dad6b2"
    assert line["type"] == "run_note" and line["ts"] > 0
    assert "step" not in line  # a record that is not a step claims no step


def test_record_is_a_silent_no_op_when_logging_is_off(tmp_path):
    sink = EventSink(tmp_path, enabled=False)
    sink.record(type="run_start", file="x.pdf")  # must not raise, must not write
    assert sink.path is None
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ui_events.py -v`
Expected: FAIL — `TypeError: EventSink.__init__() got an unexpected keyword argument 'run_id'`

- [x] **Step 3: Implement**

In `events.py`: add the four optional fields to `Event` (`run_id: str = ""`, `type: str = "step"`, `duration_ms: int | None = None`, `tokens: dict[str, int] | None = None`, `cause: str = ""`). `EventSink.__init__` takes `run_id: str | None = None` and resolves `run_id or os.environ.get("RUN_ID") or mint_run_id()` (import `mint_run_id` from `polyglot_lc.ui.trace`). Add:

```python
def record(self, **fields: Any) -> None:
    """Append one typed record: {"ts", "run_id", **fields}."""
```

which builds the dict, appends `{"ts": time.time(), "run_id": self.run_id, **fields}`, writes it as one JSON line through the same lock, and returns immediately when `enabled` is false or `path` is None. Build the line with `json.dumps(..., ensure_ascii=False)` so Vietnamese text in a message does not become escapes.

In `simple.py`: `SimpleStep.__enter__` records the token baseline (`TokenUsage.snapshot()`) and emits the existing `status="start"` event with `type="step_start"`. `__exit__` emits `type="step_end"`, `duration_ms=round(latency)`, `tokens=` the three-key delta (`sent`, `received`, `cached`, floored at 0), and on the failing path `cause=cause_for(exc, step=self.name)`. The existing `latency` field and the `error` string are unchanged.

Document in the `SimpleStep` docstring that **the per-step token delta is approximate when `MAX_CONCURRENCY > 1`**, because `TokenUsage` is a process-wide counter, and that exact attribution is sub-project E's job.

- [x] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ui_events.py -v`
Expected: PASS — the four new tests plus all pre-existing ones

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/ui/events.py src/polyglot_lc/ui/simple.py tests/test_ui_events.py
git commit -m "feat(trace): record step duration, token delta and error cause"
```

---

### Task 3: The reader and the `Trace` view

**Files:**
- Modify: `src/polyglot_lc/ui/trace.py`
- Test: `tests/test_trace.py`

**Interfaces:**
- Consumes: `CAUSES`, `cause_for` from Task 1; the record shapes from Task 2.
- Produces: `resolve_run(target: str | None, root: Path) -> Path`; `read_trace(path: Path) -> Trace`; `load_trace(target: str | None, root: Path | None = None) -> Trace`; the dataclasses `Trace`, `ChunkTimeline`, `StepRecord`, each with `as_dict()`; `Trace.as_text() -> str`.

- [x] **Step 1: Write the failing tests**

```python
def _write(tmp_path, lines):
    p = tmp_path / "run_20261012_015104.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return p


def test_reader_groups_records_into_settings_a_timeline_and_totals(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "ts": 1.0,
                "run_id": "r1",
                "type": "run_start",
                "file": "d.pdf",
                "sha256": "ab",
                "chunks": 2,
            },
            {
                "ts": 2.0,
                "run_id": "r1",
                "type": "step_end",
                "step": "translate",
                "chunk_id": "chunk-000",
                "status": "ok",
                "latency": 10.0,
                "duration_ms": 10,
                "tokens": {"sent": 5, "received": 2, "cached": 0},
            },
            {
                "ts": 3.0,
                "run_id": "r1",
                "type": "decision",
                "kind": "fast_path",
                "chunk_id": "chunk-000",
                "chosen": True,
                "reasons": ["complexity 0.10 <= 0.35"],
                "numbers": {"complexity": 0.1},
            },
            {
                "ts": 4.0,
                "run_id": "r1",
                "type": "error",
                "step": "translate",
                "chunk_id": "chunk-001",
                "cause": "rate_limit",
                "message": "429",
            },
            {
                "ts": 5.0,
                "run_id": "r1",
                "type": "run_end",
                "status": "done",
                "duration_ms": 4000,
                "calls": 3,
            },
        ],
    )
    trace = read_trace(path)
    assert trace.run_id == "r1"
    assert trace.started["file"] == "d.pdf"
    assert trace.ended["status"] == "done"
    assert [t.chunk_id for t in trace.timeline] == ["chunk-000", "chunk-001"]
    assert trace.timeline[0].steps[0].name == "translate"
    assert trace.timeline[0].decisions[0]["kind"] == "fast_path"
    assert [p["cause"] for p in trace.problems] == ["rate_limit"]
    assert trace.malformed == 0


def test_a_truncated_final_line_is_counted_and_never_hides_earlier_records(tmp_path):
    path = tmp_path / "run_x.jsonl"
    path.write_text(
        json.dumps({"ts": 1.0, "type": "run_start", "file": "d.pdf"})
        + "\n"
        + '{"ts": 2.0, "type": "step_',
        encoding="utf-8",
    )
    trace = read_trace(path)
    assert trace.malformed == 1
    assert trace.started["file"] == "d.pdf"


def test_a_log_from_the_previous_version_reads_as_steps(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "ts": 1.0,
                "step": "translate",
                "chunk_id": "chunk-000",
                "status": "ok",
                "latency": 4.0,
                "error": "",
            },
        ],
    )
    trace = read_trace(path)
    assert trace.started == {} and trace.malformed == 0
    assert trace.timeline[0].chunk_id == "chunk-000"


def test_a_run_that_died_before_its_first_record_says_so_in_words():
    assert "no records" in trace.as_text().lower() or "no run_start" in trace.as_text().lower()


def test_resolve_run_takes_an_explicit_path_or_finds_the_newest_under_the_root_recursively(
    tmp_path,
):
    nested = tmp_path / "webui" / "20261012_015104_dad6b2"
    nested.mkdir(parents=True)
    older = tmp_path / "run_20261012_010000.jsonl"
    newer = nested / "run_20261012_015104.jsonl"
    older.write_text("", encoding="utf-8")
    newer.write_text("", encoding="utf-8")
    assert resolve_run(str(newer), tmp_path) == newer
    assert resolve_run("20261012_015104_dad6b2", tmp_path) == newer
    assert resolve_run(None, tmp_path) == newer


def test_resolve_run_refuses_a_path_outside_the_log_root(tmp_path):
    outside = tmp_path.parent / "elsewhere.jsonl"
    outside.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        resolve_run(str(outside), tmp_path / "logs")
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_trace.py -v`
Expected: FAIL — `ImportError: cannot import name 'read_trace'`

- [x] **Step 3: Implement the reader**

`StepRecord` is a frozen dataclass `(name, status, duration_ms, tokens, error, cause)`. `ChunkTimeline` is `(chunk_id, steps: list[StepRecord], decisions: list[dict], problems: list[dict])` with a `route` property returning `"fast" | "standard" | "deep" | ""` derived from the decisions, and a `why()` returning the joined reason strings of the routing decision. `Trace` is `(run_id, path, started, timeline, problems, notes, ended, malformed)` with `as_dict()` and `as_text()`.

`read_trace` reads the file line by line, skipping whitespace, counting unparseable or non-object lines into `malformed`, and dispatching on `type` with `"step"` as the default for any line without one (that is how an older log reads). A record with a `chunk_id` and a status of `start`/`ok`/`fail` is a step; a `decision` attaches to its chunk; a top-level `error` goes to `problems`; `run_note` goes to `notes`; `run_start`/`run_end` fill `started`/`ended`. `trace.run_id` is the first non-empty `run_id` seen, else the file stem after `run_`.

`resolve_run` resolves an explicit path (raising `FileNotFoundError` when it is absent or resolves outside `root`), then a run id against a directory name, then a file name, then — with no target — the newest `run_*.jsonl` found by `root.rglob`. `load_trace(target, root=None)` defaults `root` to `get_app_config().logs_dir.parent` and calls `read_trace(resolve_run(...))`.

`as_text()` renders the settings, a per-chunk timeline with route and why, the problems with their causes, the notes, and the totals; with no records at all it states that the run left no records (the file may be empty or not exist) rather than printing empty tables.

- [x] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_trace.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/ui/trace.py tests/test_trace.py
git commit -m "feat(trace): read a run log back into settings, timeline, problems and totals"
```

---

### Task 4: `run_start` and `run_end`, and a run id the pipeline knows

**Files:**
- Modify: `src/polyglot_lc/pipeline/coordinator.py:137-197` (`__init__`, `run`, `_run_inner` ingest/chunking region)
- Modify: `src/polyglot_lc/pipeline/pdf_output.py` (only if `RebuildStats.failures` needs no change — it does not)
- Test: `tests/test_pipeline.py` (extend)

**Interfaces:**
- Consumes: `EventSink(..., run_id=...)` and `record()` from Task 2; `plan_hash` from Task 1.
- Produces: `TranslationPipeline(quiet=False, resume=None, run_id=None)` with attribute `self.run_id: str`; `PipelineResult.log_path` unchanged.

- [x] **Step 1: Write the failing test** (append to `tests/test_pipeline.py`, following that file's existing fixture style for a mock-mode run)

```python
def test_a_run_records_its_settings_at_the_start_and_its_totals_at_the_end(tmp_path, mock_pipeline_run):
    records = mock_pipeline_run.records()  # helper: parsed JSONL of the run's log
    start, end = records[0], records[-1]
    assert start["type"] == "run_start" and start["chunks"] == mock_pipeline_run.chunks
    assert start["sha256"] == mock_pipeline_run.sha256_of_input
    assert start["plan_hash"] and start["agents"][0]["name"] == "translator"
    assert "api_key" not in json.dumps(start).lower()
    assert end["type"] == "run_end" and end["status"] == "done" and end["chunks_done"] > 0
    assert end["tokens"]["sent"] >= end["tokens"]["received"] >= 0
    assert end["rebuild"]["placed"] >= 0


def test_the_run_end_record_survives_a_run_that_raised(...):
    # a pipeline configured to fail mid-run still writes run_end with status "failed"
```

Build the two tests against the fixture the file already uses; if that fixture returns a `PipelineResult` without a log path, pass `LOGS_DIR` through the same monkeypatch style the file already uses for `tmp_path` and read `result.log_path`.

- [x] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -k run_start -v`
Expected: FAIL — `IndexError`/`KeyError: 'type'`: the file only holds step events

- [x] **Step 3: Implement**

`TranslationPipeline.__init__` passes `run_id=run_id` to `EventSink` and keeps `self.run_id = self.sink.run_id`. In `_run_inner`, immediately after the `segment` step (where `doc` and `chunks` are both known), call `self.sink.record(type="run_start", ...)` with exactly the spec §5 fields: `file` (the file's name, not its path), `sha256` (of the file's bytes, computed here), `bytes`, `pages` (`doc.metadata.get("pages")` or `None`), `chunks`, `plan_hash(chunks)`, `src`, `tgt`, `domain`, `agents` (one `{name, model, host, temperature}` per configured agent, `host` from `urlparse(base_url).hostname`), `max_tokens`, `flags` (`resume`, `format`, `partial_output`), `version` (from `polyglot_lc.__version__`).

In the `finally` block of `run()`, after the retry stats are read, call `self.sink.record(type="run_end", ...)` with `status` (`"done"` when `result.errors` is empty else `"failed"`), `exit_code` (0 or 1), `duration_ms`, the `tokens`/`calls`/`cache_hits`/`cache_misses` totals from `TokenUsage.snapshot()`, `retries`, `fallbacks`, `chunks_done`, `chunks_failed`, `artifacts`, `rebuild`, `problems`. Do this **after** assigning `result.log_path` so the record is written even when the run raised; the rebuild stats must be carried out of `_run_inner` for this (add fields to `PipelineResult`: `pdf_kept`, `pdf_failures`).

- [x] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/pipeline/coordinator.py tests/test_pipeline.py
git commit -m "feat(trace): record what a run was asked to do and what it cost"
```

---

### Task 5: The routing decisions, with the numbers that made them

**Files:**
- Modify: `src/polyglot_lc/orchestrator/graph.py:167-181` (`should_take_fast_path`), `:204-215` (translator node), `:270-300` (verifier node), `:355-380` (`make_chunk_graph` edges)
- Test: `tests/test_adaptive_routing.py` (extend)

**Interfaces:**
- Consumes: `EventSink.record()`; `ctx.sink`; `after_critic`/`after_verifier` from `orchestrator/router.py`.
- Produces: `fast_path_verdict(state: ChunkState, app_cfg: Any) -> tuple[bool, list[str]]`, with `should_take_fast_path` delegating to it and returning only the bool.

- [x] **Step 1: Write the failing tests** (extend `tests/test_adaptive_routing.py`)

```python
def test_fast_path_verdict_explains_itself_with_the_numbers_it_used():
    state = {"complexity": 0.62, "draft_score": 0.41, "draft_issues": [], "draft_signals": None}
    cfg = _app_config(fast_path_max_complexity=0.35, fast_path_min_score=0.75)
    taken, reasons = fast_path_verdict(state, cfg)
    assert taken is False
    assert any("0.62" in r and "0.35" in r for r in reasons)


def test_a_taken_fast_path_says_what_it_passed():
    taken, reasons = fast_path_verdict({"complexity": 0.1, "draft_score": 0.9, "draft_issues": []}, _app_config())
    assert taken is True and reasons and any("<= 0.35" in r or "clean" in r for r in reasons)


def test_should_take_fast_path_still_returns_only_a_bool(...):
    # the existing behaviour is unchanged for every existing case in this file
```

- [x] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_adaptive_routing.py -k verdict -v`
Expected: FAIL — `ImportError: cannot import name 'fast_path_verdict'`

- [x] **Step 3: Implement**

Add `fast_path_verdict` next to `should_take_fast_path` and reduce the latter to `return fast_path_verdict(state, app_cfg)[0]`. Reasons name the comparison that decided it: `f"complexity {c:.2f} > {limit:.2f}"`, `"complexity unknown for this chunk"`, `"ADAPTIVE_QUALITY off"`, `f"draft issues: {', '.join(issues)}"`, `f"draft score {score:.2f} < {min_score:.2f}"`, and for a taken fast path `f"complexity {c:.2f} <= {limit:.2f}"` plus `"draft clean"`.

Record from the translator node, right after `fast` is computed:

```python
ctx.record_decision(
    chunk_id=chunk.id,
    kind="fast_path",
    chosen=fast,
    reasons=reasons,
    numbers={"complexity": ..., "max_complexity": ..., "draft_score": ..., "min_score": ...},
)
```

where `GraphContext.record_decision(**fields)` is a thin guard — `if self.sink is not None: self.sink.record(type="decision", **fields)` — added to `GraphContext` in the same file. Then wrap the two conditional edges in `make_chunk_graph` in small closures that compute `after_critic(...)`/`after_verifier(...)` once, record a `refine_gate`/`retranslate` decision with the numbers (`critic_score`, `threshold`, `refine_min_score`, `refine_round`; `ok`, `retranslate_count`), and return the same route the pure router returned. The verifier node records a `verify_mode` decision when it takes the fast verify path (`skip_llm_verify`) or the LLM path, and an `error` record with `cause="verify"` plus the verifier's own reasons when `ok` is false and the retranslate budget is gone. The placeholder repair in `collect_node` and `verifier_node` records `cause="placeholder"` when `repaired` is non-empty.

- [x] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_adaptive_routing.py tests/test_agents.py -v`
Expected: PASS — the existing routing behaviour is untouched

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/orchestrator/graph.py tests/test_adaptive_routing.py
git commit -m "feat(trace): record why each chunk took the route it took"
```

---

### Task 6: `polyglot trace` and `--run-id`

**Files:**
- Modify: `src/polyglot_lc/cli.py` (new command; `--run-id` on `translate`)
- Test: `tests/test_trace_cli.py`

**Interfaces:**
- Consumes: `load_trace`, `Trace` from Task 3; `TranslationPipeline(run_id=...)` from Task 4.
- Produces: the `trace` typer command; `translate` gains `--run-id`.

- [x] **Step 1: Write the failing tests**

```python
from typer.testing import CliRunner
from polyglot_lc.cli import app

runner = CliRunner()


def test_trace_prints_settings_reasons_problems_and_totals(tmp_path, a_finished_run_log):
    result = runner.invoke(app, ["trace", str(a_finished_run_log)])
    assert result.exit_code == 0
    out = result.stdout
    assert "d.pdf" in out and "chunk-000" in out
    assert "rate_limit" in out  # the problem's cause, not just its message


def test_trace_explains_a_run_that_never_started(tmp_path):
    empty = tmp_path / "run_empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["trace", str(empty)])
    assert result.exit_code == 0  # reading a broken run is not a command failure
    assert "no records" in result.stdout.lower() or "no run_start" in result.stdout.lower()


def test_trace_json_is_parseable_and_stable(tmp_path, a_finished_run_log):
    first = runner.invoke(app, ["trace", str(a_finished_run_log), "--json"]).stdout
    second = runner.invoke(app, ["trace", str(a_finished_run_log), "--json"]).stdout
    assert first == second
    assert json.loads(first)["run_id"]


def test_trace_without_a_target_explains_when_the_log_directory_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGS_DIR", str(tmp_path))
    result = runner.invoke(app, ["trace"])
    assert result.exit_code != 0
    assert "no run" in result.stdout.lower()
```

- [x] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_trace_cli.py -v`
Expected: FAIL — `Error: No such command 'trace'`

- [x] **Step 3: Implement**

Add `trace(target: str | None = typer.Argument(None), json_out: bool = typer.Option(False, "--json"))`. It calls `load_trace(target)`, prints `as_text()` or `json.dumps(trace.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)`, and exits 0 even for a broken run — except when no run can be resolved at all, which prints one line naming the directory searched and exits 1. Add `--run-id` to `translate` (default `None`), pass it to `TranslationPipeline(quiet=quiet, run_id=run_id)`, and extend the end-of-run report with the reason line: how many chunks the fast-path gate rejected and the most common reason among them, read from the run's own decisions (`result.fast_path_rejections: int`, `result.fast_path_top_reason: str`, both filled in Task 4's report block from the decision records the coordinator can count).

- [x] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_trace_cli.py tests/test_ui_events.py tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/polyglot_lc/cli.py src/polyglot_lc/pipeline/coordinator.py tests/test_trace_cli.py
git commit -m "feat(cli): add polyglot trace and --run-id"
```

---

### Task 7: The web UI shows the cause and the route mix

**Files:**
- Modify: `webui/runs.py:202` (`env_for`)
- Modify: `webui/server.py:452-481` (`_problems`), `_run_detail` (route mix)
- Modify: `webui/static/app.js` (cause badge, route line)
- Test: `tests/test_webui_server.py` (extend)

**Interfaces:**
- Consumes: the `cause` and `type` keys on the records from Tasks 2 and 5; the child's `RUN_ID` env from `env_for`.
- Produces: a `problems` entry that carries `cause`; a `routes` object on the run detail payload.

- [x] **Step 1: Write the failing tests** (extend `tests/test_webui_server.py`, following its existing request helper)

```python
def test_a_problem_entry_carries_the_cause_from_the_log(...):
    payload = client.get(f"/api/runs/{run_id}").json()
    assert payload["problems"][0]["cause"] == "rate_limit"


def test_the_run_detail_reports_the_route_mix_from_the_decisions(...):
    payload = client.get(f"/api/runs/{run_id}").json()
    assert payload["routes"] == {"fast": 1, "standard": 1, "deep": 0}


def test_the_child_is_told_which_run_it_is(...):
    assert env["RUN_ID"] == record.run_id
```

- [x] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_webui_server.py -k "cause or route_mix" -v`
Expected: FAIL — `KeyError: 'cause'`

- [x] **Step 3: Implement**

`env_for` adds `"RUN_ID": record.run_id` beside `LOGS_DIR`. `_problems` adds `"cause": str(event.get("cause") or "")` to each entry (no import of the package — the value is copied, not validated; the pipeline is what guarantees it is in the closed set). `_run_detail` counts the `decision` records whose `kind == "fast_path"` by their chunk's resolved route, using the same guarded read `_problems` already uses, and returns `routes`. In `app.js`, render `cause` as a short badge before the message and add one `route:` line to the run's progress card when `routes` is present; `app.css` gets the badge style beside the existing problem styles.

- [x] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_webui_server.py tests/test_webui_runs.py tests/test_webui_store.py tests/test_webui_probe.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webui/runs.py webui/server.py webui/static/app.js webui/static/app.css tests/test_webui_server.py
git commit -m "feat(webui): show the error cause and the route mix of a run"
```

---

### Task 8: The whole-suite gate and the documented command

**Files:**
- Modify: `README.md` (the observability section: what `polyglot trace` prints, and that the log is metadata-only)

- [x] **Step 1: Add the canary and no-document-text tests** to `tests/test_trace.py` if Task 4 did not already cover them for a real mock run: set an env value shaped like a credential, run the mock pipeline, and assert the value appears nowhere in the file; run a document containing a unique marker sentence and assert that sentence is absent.

- [x] **Step 2: Run the full gate**

Run: `python -m pytest` then `python -m ruff check .` then `python -m ruff format --check .`
Expected: all green

- [x] **Step 3: Document and commit**

```bash
git add README.md tests/test_trace.py
git commit -m "docs: describe the structured trace and its metadata-only guarantee"
```
