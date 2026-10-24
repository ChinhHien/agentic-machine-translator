"""End-to-end coordinator: ingest -> analyse -> plan -> translate -> assemble.

Owns subsystem construction (agents, reasoning bank, skillopt, UI) and the final
report. The heavy lifting per chunk happens inside the LangGraph, and the
per-chunk results are checkpointed as they finish so a run can be resumed.

Workflow (adaptive, P0+P1 architecture)
---------------------------------------

1. **Ingest + chunk** — local, no model calls.
2. **Analyse** — heuristic terminology candidates plus a per-chunk complexity
   score. Local, no model calls.
3. **Plan** — ONE Planner request produces the document summary, strategy,
   glossary (with target forms), complexity and per-section notes.
4. **Translate** — bounded-parallel chunks, each routed through the fast or the
   full quality path (see :mod:`polyglot_lc.orchestrator.graph`), each
   checkpointed as it completes.
5. **Consistency** — the planned glossary is enforced across chunk boundaries,
   patching only the chunks that drifted.
6. **Assemble + report** — code blocks restored, output written once, SkillOpt
   updated best-effort.

One notable design decision: the per-chunk "self score" is computed
deterministically (no extra prompt, no extra tokens) instead of asking the
Translator to grade its own output in the reply. The README carries the full
stage-by-stage contract and the request/latency measurements behind it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from polyglot_lc import __version__
from polyglot_lc.agents import (
    ConsistencyAgent,
    CriticAgent,
    PlannerAgent,
    RefinerAgent,
    TranslatorAgent,
    VerifierAgent,
)
from polyglot_lc.analysis import DocumentPlan, score_chunk
from polyglot_lc.config import (
    ConfigError,
    RetryConfig,
    get_agent_config,
    get_app_config,
    get_embed_config,
)
from polyglot_lc.ingest import load_document
from polyglot_lc.llm.client import TokenUsage, active_api_calls
from polyglot_lc.llm.resilience import (
    LLMError,
    NoBackupError,
    PermanentError,
    RetryStats,
    jittered_delay,
)
from polyglot_lc.orchestrator.graph import (
    GraphContext,
    make_chunk_graph,
    make_document_plan,
)
from polyglot_lc.pipeline.assembler import (
    CHUNK_SEPARATOR,
    assemble,
    assemble_partial,
    block_translations,
    coverage_error,
    output_paths,
    write_atomic,
)
from polyglot_lc.pipeline.checkpoint import CheckpointStore, ChunkCheckpoint
from polyglot_lc.pipeline.consistency import ConsistencyChecker
from polyglot_lc.pipeline.pdf_output import rebuild_pdf
from polyglot_lc.reasoning_bank.embedder import Embedder
from polyglot_lc.reasoning_bank.store import ReasoningBank
from polyglot_lc.segment.chunker import Chunk, chunk_document
from polyglot_lc.segment.glossary import (
    MAX_USER_GLOSSARY_TERMS,
    default_glossary,
    extract_glossary,
    glossary_budget,
    load_glossary_csv,
)
from polyglot_lc.skillopt.optimizer import SkillOptimizer
from polyglot_lc.skillopt.skill_store import SkillStore
from polyglot_lc.ui.events import EventSink
from polyglot_lc.ui.simple import SimpleUI
from polyglot_lc.ui.trace import plan_hash

#: How often a chunk that is waiting to be run again checks whether the run was
#: stopped, so an unbounded wait is still interruptible.
_RETRY_TICK = 0.5

#: Jitter on the chunk-level wait. Chunks that failed together during an outage
#: must not come back in the same instant and collide again — the same reason the
#: per-request backoff has jitter (see :mod:`polyglot_lc.llm.resilience`).
_CHUNK_RETRY_JITTER = 0.2

#: How many extra times a chunk is asked when its reply comes back cut short
#: (see :func:`~polyglot_lc.pipeline.assembler.coverage_error`). Two is enough
#: for a stochastic early stop to change its mind, and few enough that a
#: deterministic one cannot spin the run; the reply that is still partial
#: afterwards is kept in the output but not checkpointed, so a later
#: ``--resume`` asks the chunk again.
_MAX_COVERAGE_RETRIES = 2

#: How many times a chunk whose reply came back *empty* is asked again before the
#: halves are tried. A cut reply is a stochastic early stop that a fresh sample
#: can fix; an empty one is a successful call that carried nothing — a refusal, a
#: content filter, a model that answered with whitespace — and asking the same
#: question forever never changes that answer. It used to: the wait for an empty
#: reply had no ceiling, so one such chunk held the whole run at its progress bar
#: until the reader stopped it.
_MAX_EMPTY_RETRIES = 2

#: How deep a chunk that fails the coverage gate is split and re-asked (see
#: ``salvage_by_split``): the chunk, then each half once. Two levels take a
#: reply down to ~a quarter of the chunk's length — short enough that the
#: early stop never reaches it — without turning one cut reply into a
#: pyramid of ever-smaller asks.
_MAX_SPLIT_DEPTH = 2


def _coverage_enabled(translator_cfg: Any) -> bool:
    """Whether replies are measured against the chunk they claim to translate.

    Only a real provider is asked to translate a whole chunk: the mock
    harness answers with a canned pseudo-reply by design, and measuring that
    would flag every test run (see
    :func:`~polyglot_lc.pipeline.assembler.coverage_error`).
    """
    return not translator_cfg.is_mock


@dataclass
class PipelineResult:
    """Final outcome of one translation run."""

    output_path: str = ""
    #: The rebuilt, layout-preserving PDF, when OUTPUT_FORMAT asked for one.
    pdf_path: str = ""
    pdf_blocks: int = 0
    chunks: int = 0
    avg_critic: float = 0.0
    #: Every retry this run paid for: request-level (rate limit, transient) and
    #: chunk-level (a chunk run again for an empty or cut-short reply).
    #: ``chunk_retries`` is the chunk-level part of this total, not extra to it.
    retries: int = 0
    chunk_retries: int = 0
    traces_stored: int = 0
    log_path: str = ""
    errors: list[str] = field(default_factory=list)
    skill_promoted: bool = False
    # --- adaptive / planning / durability metrics ---
    planner_calls: int = 0
    document_complexity: float = 0.0
    glossary_terms: int = 0
    fast_path_chunks: int = 0
    standard_path_chunks: int = 0
    deep_path_chunks: int = 0
    resumed_chunks: int = 0
    #: How many inconsistencies the pass found across chunks — the count is of
    #: *drifts*, not of terms the pass inspected (that is ``checked_terms``).
    consistency_inconsistencies: int = 0
    consistency_patches: int = 0
    #: Chunks left drifting — the patch budget ran out — and patches the pass
    #: refused because the rewrite lost the translation or did not carry the
    #: expected term. Both are reported: a run whose drift stayed in the file
    #: must not read like a clean one.
    consistency_skipped: int = 0
    consistency_rejected: int = 0
    #: Labels and repeated paragraph openings compared, and how many of them the
    #: chunks rendered differently — the part of the pass that needs no glossary.
    consistency_labels: int = 0
    consistency_label_drift: int = 0
    #: Terms the pass actually compared, and whether it had *no* term to compare
    #: at all (a fallback plan, no CSV): "0 inconsistencies" over zero terms and
    #: "0 inconsistencies" over 175 terms are not the same result.
    consistency_checked_terms: int = 0
    consistency_no_pairs: bool = False
    #: Requests the pass was allowed to spend on patches (see
    #: :class:`~polyglot_lc.pipeline.consistency.ConsistencyReport`), so the
    #: cost of the pass is visible before/with the report rather than inferred.
    consistency_patch_budget: int = 0
    #: True when the pass stopped after two patches in a row were refused — the
    #: rest of its budget was not spent, and the chunks it left are counted as
    #: skipped rather than silently looking consistent.
    consistency_stopped_early: bool = False
    #: Chunks that shipped without a verdict because the Verifier could not run.
    verify_skipped: int = 0
    partial_output: bool = False
    elapsed_s: float = 0.0
    chunks_failed: int = 0
    #: The error that ended the run, when it ended without finishing. Empty for
    #: a run that finished, however many provider failures it had to wait out.
    fatal_error: str = ""
    #: PDF rebuild detail, carried out of the run so the final record can report it.
    pdf_kept: int = 0
    pdf_failures: list[str] = field(default_factory=list)
    #: Why the fast path was refused, from the decisions the graph recorded.
    fast_path_rejections: int = 0
    fast_path_top_reason: str = ""

    @property
    def avg_critic_estimate(self) -> bool:
        """True when no Critic scored anything (all chunks took the fast path)."""
        return self.standard_path_chunks == 0 and self.deep_path_chunks == 0

    def consistency_fields(self) -> dict[str, Any]:
        """The consistency counters under their ``run_end`` names.

        Read off the dataclass rather than listed again: the fields' names are
        the record's keys, so a counter added here cannot silently fail to
        reach the run record (or be mapped with another field's value).
        """
        fields = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name.startswith("consistency_")
        }
        # The run record has named this one ``consistency_patched`` since it was
        # added, and the trace view and web UI read that key. The attribute keeps
        # its plural because it is a count of chunks, not a state.
        fields["consistency_patched"] = fields.pop("consistency_patches")
        return fields


def _file_sha256(path: Path) -> str:
    """SHA-256 of the input file, read once; empty when the file is unreadable.

    Only the digest is ever recorded: two runs of the same document can then be
    compared without the trace holding any of the document.
    """
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return ""


def _detect_lang(text: str) -> str:
    """Detect source language with langdetect (falls back to 'en')."""
    try:
        from langdetect import detect

        return detect(text[:2000]) or "en"
    except Exception:  # noqa: BLE001 - detection is best-effort
        return "en"


class TranslationPipeline:
    """Orchestrates one full document translation."""

    def __init__(
        self, quiet: bool = False, resume: bool | None = None, run_id: str | None = None
    ) -> None:
        self.app = get_app_config()
        self.quiet = quiet
        # None -> follow RESUME from the environment; explicit value wins.
        self.resume = self.app.resume if resume is None else bool(resume)
        # Validate agent configs up-front so missing keys hit the Errors panel.
        self.agent_configs = {
            name: get_agent_config(name)
            for name in ("PLANNER", "TRANSLATOR", "CRITIC", "REFINER", "VERIFIER")
        }
        self.embed_config = get_embed_config()
        # An explicit id wins, then RUN_ID (what the web UI exports), then one is
        # minted. Whichever it is, the web UI's run.json and every trace record
        # end up naming the same run.
        self.sink = EventSink(self.app.logs_dir, enabled=self.app.log_jsonl, run_id=run_id)
        self.run_id = self.sink.run_id
        #: Set when the run is interrupted; a chunk that is waiting for its next
        #: attempt checks it and gives up the wait instead of holding the run.
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    def run(
        self,
        file_path: str | Path,
        src: str = "auto",
        tgt: str = "vi",
        glossary: str | None = None,
        out: str | None = None,
        max_tokens: int | None = None,
        resume: bool | None = None,
        checkpoint_dir: str | Path | None = None,
        output_format: str | None = None,
    ) -> PipelineResult:
        """Translate *file_path* from *src* to *tgt*; returns a report."""
        result = PipelineResult()
        started = time.monotonic()

        with SimpleUI(self.sink, quiet=self.quiet) as ui:
            try:
                return self._run_inner(
                    file_path,
                    src,
                    tgt,
                    glossary,
                    out,
                    max_tokens,
                    ui,
                    result,
                    resume=self.resume if resume is None else bool(resume),
                    checkpoint_dir=checkpoint_dir or self.app.checkpoint_dir,
                    output_format=output_format or self.app.output_format,
                )
            except ConfigError as exc:
                ui.add_error(str(exc))
                result.errors = list(ui.errors)
                result.fatal_error = str(exc)
                logger.error("Config error: {}", exc)
                raise
            except Exception as exc:
                ui.add_error(f"{type(exc).__name__}: {exc}")
                result.errors = list(ui.errors)
                result.fatal_error = f"{type(exc).__name__}: {exc}"
                raise
            except BaseException as exc:
                # Ctrl-C, or a run cut short by its launcher: no deliverable was
                # finished, and the record has to say so. It is the one path that
                # reaches the final record without a chunk having failed.
                result.errors = result.errors or list(ui.errors)
                result.fatal_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                # Retries come from the resilience layer's own counters, which
                # are the only place a retry actually happens. A chunk run
                # again for an empty or cut-short reply is a retry too: counting
                # only the request-level ones understated the real cost, the
                # more so on a document where replies keep coming back short.
                stats = RetryStats.snapshot()
                result.chunk_retries = stats["chunk_retries"]
                result.retries = (
                    stats["rate_limit_retries"]
                    + stats["transient_retries"]
                    + stats["chunk_retries"]
                )
                result.log_path = str(self.sink.path or "")
                # Written from the `finally` so a run that raised still leaves a
                # record of how it ended — that is the run most worth reading.
                self._record_run_end(result, stats, round((time.monotonic() - started) * 1000))
                # Every provider failure this run had to wait out is in the
                # report, including the ones a later attempt resolved: the file
                # is finished, and what it cost to finish it is the reader's to
                # see.
                result.errors = result.errors or list(ui.errors)

    def _record_run_end(
        self, result: PipelineResult, stats: dict[str, int], duration_ms: int
    ) -> None:
        """Write the run's final record: totals, artefacts, rebuild stats, problems.

        Reads its numbers from wherever they actually live — the resilience
        layer for retries, ``TokenUsage`` for tokens — rather than from anything
        this class has been accumulating alongside them.
        """
        usage = TokenUsage.snapshot()
        # "Failed" is a verdict on the deliverable, not on the road to it: a run
        # that waited out a provider outage and translated every chunk is done.
        # It reads as failed only when a chunk has no translation or the run
        # stopped on an error. Its problems are still counted below.
        failed = bool(result.chunks_failed or result.fatal_error)
        artifacts: list[dict[str, Any]] = []
        for kind, path in (("md", result.output_path), ("pdf", result.pdf_path)):
            if not path:
                continue
            target = Path(path)
            artifacts.append(
                {
                    "kind": kind,
                    "path": str(target),
                    "bytes": target.stat().st_size if target.is_file() else 0,
                }
            )
        calls = active_api_calls()
        self.sink.record(
            type="run_end",
            status="failed" if failed else "done",
            active_api_calls=calls.get("active"),
            max_active_api_calls=calls.get("peak"),
            exit_code=1 if failed else 0,
            duration_ms=duration_ms,
            tokens={
                "sent": usage["sent"],
                "received": usage["received"],
                "cached": usage.get("cached", 0),
            },
            calls=usage["calls"],
            cache_hits=usage.get("cache_hits", 0),
            cache_misses=usage.get("cache_misses", 0),
            # ``retries`` is the total; ``chunk_retries`` is its chunk-level
            # share, reported so a reader can see what kind of retry it was
            # without adding the two together.
            retries=result.retries,
            chunk_retries=result.chunk_retries,
            fallbacks=stats.get("fallbacks", 0),
            chunks_done=max(0, result.chunks - result.chunks_failed),
            chunks_failed=result.chunks_failed,
            resumed_chunks=result.resumed_chunks,
            # The consistency pass's own tally, on the record a reader opens
            # first: chunk left drifting and patch refused were only ever
            # visible as lines buried in the log.
            **result.consistency_fields(),
            verify_skipped=result.verify_skipped,
            artifacts=artifacts,
            rebuild={
                "placed": result.pdf_blocks,
                "kept": result.pdf_kept,
                "failures": result.pdf_failures[:5],
            },
            # The problems actually recorded in this file, not only the errors
            # that reached the report: a reader can then reconcile the count
            # with the `error` records it finds.
            problems=self.sink.problem_count,
        )

    def _agent_summaries(self) -> list[dict[str, Any]]:
        """Who ran this run: model, **host**, sampling — never a URL or a key.

        The host rather than the configured ``base_url`` keeps any credential
        and any path in it out of a file that gets copied, shared and read.
        """
        return [
            {
                "name": name.lower(),
                "model": cfg.model,
                "host": urlparse(cfg.base_url).hostname or "",
                "temperature": cfg.temperature,
            }
            for name, cfg in self.agent_configs.items()
        ]

    # ------------------------------------------------------------------
    def _build_subsystems(self, ui: SimpleUI) -> dict[str, Any]:
        """Construct the agents plus the best-effort Reasoning Bank."""
        cfg = self.agent_configs
        reasoning_bank: ReasoningBank | None = None
        try:
            embedder = Embedder(self.embed_config)
            reasoning_bank = ReasoningBank(self.app.reasoning_bank_path, embedder)
        except Exception as exc:  # noqa: BLE001 - RB is best-effort
            ui.add_error(f"reasoning bank unavailable: {exc}")
            logger.warning("Reasoning bank unavailable: {}", exc)
        return {
            "planner": PlannerAgent(cfg["PLANNER"]),
            "translator": TranslatorAgent(cfg["TRANSLATOR"]),
            "critic": CriticAgent(cfg["CRITIC"]),
            "refiner": RefinerAgent(cfg["REFINER"]),
            "verifier": VerifierAgent(cfg["VERIFIER"]),
            "reasoning_bank": reasoning_bank,
        }

    # ------------------------------------------------------------------
    def _plan(
        self,
        *,
        ctx: GraphContext,
        chunks: list[Any],
        heuristic_terms: list[Any],
        user_glossary: dict[str, str],
        src: str,
        tgt: str,
        doc_text: str,
        doc_title: str,
    ) -> DocumentPlan:
        """Produce the document plan — one Planner request for the document.

        There used to be a second path (``UNIFIED_PLANNER=0``): a summary tree,
        one Planner request per section plus one for the document, then a
        separate plan request on top of it. It cost a request per section,
        returned no glossary pairs, and shared nothing with this one but the
        output shape, so it was removed rather than kept in step twice.
        """
        return make_document_plan(
            ctx,
            chunks=chunks,
            src_lang=src,
            tgt_lang=tgt,
            doc_text=doc_text,
            doc_title=doc_title,
            heuristic_terms=heuristic_terms,
            user_glossary=user_glossary,
        )

    # ------------------------------------------------------------------
    def _run_chunks(
        self,
        chunk_graph: Any,
        chunks: list[Any],
        base_state: dict[str, Any],
        ui: SimpleUI,
        *,
        ctx: GraphContext | None = None,
        per_chunk_state: dict[str, dict[str, Any]] | None = None,
        store: CheckpointStore | None = None,
        doc_key: str = "",
        on_chunk_done: Callable[[Any, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Run each chunk's graph, bounded-parallel across independent chunks.

        Each chunk keeps its internal Translate->[Critic->Refiner]->Verify order;
        only *different* chunks run concurrently (``MAX_CONCURRENCY``). Results
        are returned in the original chunk order regardless of completion order.

        **A chunk is never skipped.** Provider failures are waited out and the
        chunk is run again until it has a translation, so the file a finished
        run writes holds no placeholder for it. A chunk whose draft survived the
        failure resumes from that draft — the translator is not asked for the
        same words twice (:meth:`GraphContext.remember_draft`). The wait is
        unbounded by design for a *failure* — an outage ends by itself — while a
        reply that arrives empty is re-asked only ``_MAX_EMPTY_RETRIES`` times
        before the chunk is given up on, because no amount of waiting turns
        "nothing" into a translation.

        **Unless there is nothing left to wait for.** A failure that no retry
        can fix — a spent attempt with no backup endpoint behind it
        (:class:`NoBackupError`), or one the provider itself calls permanent
        (:class:`PermanentError`) — ends the run instead of the wait: the
        chunk is reported as failed, the other chunks stop, and the run's
        record says why. Waiting on an endpoint that has already said no is
        how a run used to sit at "0/38" for an hour.
        """
        per_chunk_state = per_chunk_state or {}
        self._stop.clear()
        # By the time a failure reaches this loop, the resilience layer has
        # already spent its own retries on the request, so the wait here starts
        # longer and is capped higher than the per-request one.
        retry_cfg = RetryConfig(
            base_delay=self.app.chunk_retry_base_delay,
            max_delay=self.app.chunk_retry_max_delay,
            jitter=_CHUNK_RETRY_JITTER,
        )

        # A reply can only be measured against the whole chunk when a real
        # provider was asked for one: the mock harness answers with a canned
        # pseudo-reply by design (see :func:`coverage_error`).
        measure_coverage = _coverage_enabled(self.agent_configs["TRANSLATOR"])

        def translate_chunk(chunk: Any) -> dict[str, Any]:
            state = {**base_state, **per_chunk_state.get(chunk.id, {}), "chunk": chunk}
            kept = ctx.drafts.get(chunk.id, "") if ctx is not None else ""
            if kept.strip():
                state["kept_draft"] = kept
            return chunk_graph.invoke(state, config={"recursion_limit": 100})

        def wait_and_retry(chunk: Any, attempt: int, reason: str) -> None:
            """Report the wait, then wait in slices so a stop can cut it short."""
            delay = jittered_delay(retry_cfg, attempt - 1)
            where = ctx.last_step(chunk.id) if ctx is not None else ""
            logger.warning(
                "Chunk {} attempt {} failed ({}); waiting {:.0f}s and running it again",
                chunk.id,
                attempt,
                reason,
                delay,
            )
            ui.retry(
                where,
                chunk.id,
                f"{reason} - waiting {delay:.0f}s, then attempt {attempt + 1}",
            )
            RetryStats.add_chunk_retry()
            waited = 0.0
            while waited < delay and not self._stop.is_set():
                slice_s = min(_RETRY_TICK, delay - waited)
                time.sleep(slice_s)
                waited += slice_s

        def settle(chunk: Any, depth: int) -> tuple[dict[str, Any], bool, bool]:
            """Translate *chunk* until its reply covers it.

            Returns ``(state, kept_partial, aborted)``; *aborted* means the
            whole run was stopped and the state carries the reason rather
            than a translation.
            """
            attempt = 0
            coverage_retries = 0
            empty_replies = 0
            while True:
                if self._stop.is_set():
                    # Another chunk hit something no retry can fix, so the run
                    # is over: this chunk is not asked again.
                    return {"translated": "", "chunk": chunk, "error": "run stopped"}, False, True
                attempt += 1
                try:
                    out = translate_chunk(chunk)
                except LLMError as exc:
                    ui.add_error(f"{chunk.id}: {type(exc).__name__}: {exc}")
                    if isinstance(exc, (NoBackupError, PermanentError)):
                        # Nothing left to try: the attempts are spent and there
                        # is no backup endpoint to switch to, or the failure is
                        # one no retry can fix. Asking again would be the same
                        # question with the same answer, so the run stops here
                        # and says why — instead of spinning on a dead provider
                        # for as long as the reader keeps the page open.
                        self._stop.set()
                        return {"translated": "", "chunk": chunk, "error": str(exc)}, False, True
                    wait_and_retry(chunk, attempt, f"{type(exc).__name__}: {exc}")
                    if self._stop.is_set():
                        return {"translated": "", "chunk": chunk, "error": str(exc)}, False, True
                    continue
                translated = out.get("translated", "") or ""
                if translated.strip():
                    reason = coverage_error(chunk, translated) if measure_coverage else None
                    if reason is None:
                        return out, False, False
                    if coverage_retries >= _MAX_COVERAGE_RETRIES:
                        salvaged = salvage_by_split(chunk, depth)
                        if salvaged is not None:
                            return {**out, "translated": salvaged}, False, False
                        # Every way of asking — whole, then in halves — came
                        # back cut short, and the words that DID arrive read
                        # better in the output than none at all. Keep them for
                        # this run's output, leave them out of the checkpoint
                        # so the next --resume asks the chunk again, and say so
                        # where a reader can see it.
                        logger.warning(
                            "chunk={} {}; keeping the partial reply for this "
                            "run's output but not checkpointing it",
                            chunk.id,
                            reason,
                        )
                        ui.add_error(f"{chunk.id}: {reason} - kept partial, not checkpointed")
                        return out, True, False
                    coverage_retries += 1
                    # Re-ask immediately: unlike an outage, a cut reply needs
                    # no cool-down — a fresh sample IS the retry.
                    RetryStats.add_chunk_retry()
                    logger.warning(
                        "chunk={} {}; asking again ({}/{})",
                        chunk.id,
                        reason,
                        coverage_retries,
                        _MAX_COVERAGE_RETRIES,
                    )
                    ui.retry(
                        ctx.last_step(chunk.id) if ctx is not None else "",
                        chunk.id,
                        f"{reason} - asking again",
                    )
                    if self._stop.is_set():
                        return {"translated": "", "chunk": chunk, "error": reason}, False, True
                    continue
                # An empty translation is not a translation: the chunk is still
                # owed its words. But a provider that answers with nothing does
                # so deterministically, so the asks are bounded: a couple of
                # re-asks, then the halves (a shorter question is a different
                # question), then this chunk is left with a visible marker in
                # the output instead of holding the whole run.
                if empty_replies >= _MAX_EMPTY_RETRIES:
                    salvaged = salvage_by_split(chunk, depth)
                    if salvaged is not None:
                        return {"translated": salvaged, "chunk": chunk}, False, False
                    logger.warning(
                        "chunk={} kept coming back empty; leaving it untranslated for "
                        "this run and out of the checkpoint",
                        chunk.id,
                    )
                    ui.add_error(f"{chunk.id}: empty translation - left untranslated")
                    return (
                        {
                            "translated": "",
                            "chunk": chunk,
                            "error": "empty translation",
                        },
                        True,
                        False,
                    )
                empty_replies += 1
                wait_and_retry(chunk, attempt, "empty translation")
                if self._stop.is_set():
                    return (
                        {"translated": "", "chunk": chunk, "error": "empty translation"},
                        False,
                        True,
                    )

        def salvage_by_split(chunk: Any, depth: int) -> str | None:
            """Ask for the halves of a chunk that keeps coming back cut short.

            A deterministic early stop does not change its mind when the same
            question is asked again — but a shorter question is a different
            question. Splitting the chunk at a paragraph boundary and
            translating each half as its own request halves what a reply must
            cover, and short requests near-never get cut. The halves' replies
            are merged and measured against the whole chunk: a merge that
            still fails the gate is no better than the partial reply, so the
            caller falls back to it.
            """
            if depth >= _MAX_SPLIT_DEPTH:
                return None
            paragraphs = chunk.text.split(CHUNK_SEPARATOR)
            if len(paragraphs) < 2:
                return None  # one paragraph cannot be asked in two halves
            # Cut nearest the middle by characters, so neither half is the
            # long one that reaches the early stop.
            mid = len(chunk.text) / 2
            acc = 0.0
            cut = 1
            for i, paragraph in enumerate(paragraphs[:-1]):
                acc += len(paragraph) + len(CHUNK_SEPARATOR)
                if acc >= mid:
                    cut = i + 1
                    break
            halves: list[str] = []
            for k, part in enumerate((paragraphs[:cut], paragraphs[cut:])):
                text = CHUNK_SEPARATOR.join(part)
                sub = Chunk(
                    id=f"{chunk.id}.d{depth}.{k}",
                    text=text,
                    token_count=max(
                        1, int(chunk.token_count * len(text) / max(1, len(chunk.text)))
                    ),
                    section_path=chunk.section_path,
                    index=chunk.index,
                )
                sub_out, _, aborted = settle(sub, depth + 1)
                sub_text = (sub_out.get("translated", "") or "").strip()
                if aborted or not sub_text:
                    logger.warning(
                        "chunk={} salvage half {} got no translation; "
                        "keeping the partial reply for the whole chunk",
                        chunk.id,
                        k,
                    )
                    return None
                halves.append(sub_text)
            joined = CHUNK_SEPARATOR.join(halves)
            if coverage_error(chunk, joined) is not None:
                logger.warning(
                    "chunk={} salvage merge of {} halves is still cut short; "
                    "keeping the partial reply for the whole chunk",
                    chunk.id,
                    len(halves),
                )
                return None
            logger.info(
                "chunk={} salvaged by splitting into {} halves ({} + {} chars)",
                chunk.id,
                len(halves),
                len(halves[0]),
                len(halves[1]) if len(halves) > 1 else 0,
            )
            return joined

        def run_one(index: int, chunk: Any) -> dict[str, Any]:
            out, kept_partial, aborted = settle(chunk, 0)
            if aborted:
                return out
            # A translation — whole or salvaged — is worth keeping: a resume
            # never has to translate the chunk again. A partial one that
            # failed the coverage check is kept out of the checkpoint so a
            # later --resume can still recover the words it never got.
            if store is not None and store.usable and not kept_partial:
                store.put(doc_key, self._to_checkpoint(chunk, out))
            if on_chunk_done is not None:
                on_chunk_done(chunk, out)
            return out

        results: list[Any] = [None] * len(chunks)
        workers = max(1, self.app.max_concurrency)
        if workers == 1 or len(chunks) <= 1:
            for i, chunk in enumerate(chunks):
                results[i] = run_one(i, chunk)
                ui.advance()
            return results

        with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
            futures = {pool.submit(run_one, i, chunk): i for i, chunk in enumerate(chunks)}
            try:
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()  # raise the first failure
                    ui.advance()
            except BaseException:
                # A chunk is retried without a ceiling, so an interrupted run
                # has to say so: the workers check this while they wait, and this
                # block joins them on the way out.
                self._stop.set()
                raise
        return results

    @staticmethod
    def _to_checkpoint(chunk: Any, state: dict[str, Any]) -> ChunkCheckpoint:
        critique = state.get("critique")
        if critique is None:
            score = float(state.get("draft_score", 0.0) or 0.0)
        else:
            score = float(getattr(critique, "score", 0.0) or 0.0)
        translated = state.get("translated", "")
        return ChunkCheckpoint(
            chunk_id=chunk.id,
            chunk_hash=chunk.hash(),
            translated=translated,
            score=round(score, 2),
            path=str(state.get("path", "standard")),
            stored=bool(state.get("stored", False)),
        )

    @staticmethod
    def _from_checkpoint(chunk: Any, checkpoint: ChunkCheckpoint) -> dict[str, Any]:
        """Rebuild a chunk state from a checkpoint so the run can skip it."""
        return {
            "chunk": chunk,
            "translated": checkpoint.translated,
            "trace": {"score": round(checkpoint.score / 10.0, 3), "resumed": True},
            "stored": checkpoint.stored,
            "path": checkpoint.path,
            "draft_score": checkpoint.score,
            "critique": None,
            "resumed": True,
        }

    # ------------------------------------------------------------------
    def _run_inner(
        self,
        file_path: str | Path,
        src: str,
        tgt: str,
        glossary: str | None,
        out: str | None,
        max_tokens: int | None,
        ui: SimpleUI,
        result: PipelineResult,
        *,
        resume: bool = False,
        checkpoint_dir: str | Path | None = None,
        output_format: str = "md",
    ) -> PipelineResult:
        app = self.app
        fmt = (output_format or "md").strip().lower()
        want_md, want_pdf = fmt in {"md", "both"}, fmt in {"pdf", "both"}

        # ---- ingest ----
        with ui.step("ingest") as _:
            doc = load_document(file_path)
            ui.set_stats(file=doc.metadata.get("filename", ""), blocks=len(doc.blocks))

        # ---- language detection ----
        if src == "auto":
            with ui.step("detect_lang"):
                src = _detect_lang(doc.plain_text())
        ui.set_stats(src=src, tgt=tgt)

        # ---- chunking (token budget from TRANSLATOR_MODEL) ----
        translator_cfg = self.agent_configs["TRANSLATOR"]
        with ui.step("segment"):
            chunks = chunk_document(doc, max_tokens=max_tokens, model=translator_cfg.model)
            ui.set_stats(chunks=len(chunks))

        # ---- what this run was asked to do, while it is still true ----
        source_path = Path(file_path)
        self.sink.record(
            type="run_start",
            file=source_path.name,
            sha256=_file_sha256(source_path),
            bytes=source_path.stat().st_size if source_path.is_file() else 0,
            pages=doc.metadata.get("pages"),
            chunks=len(chunks),
            plan_hash=plan_hash(chunks),
            src=src,
            tgt=tgt,
            agents=self._agent_summaries(),
            max_tokens=max_tokens if max_tokens is not None else app.chunk_max_tokens,
            flags={
                "resume": bool(resume),
                "format": fmt,
                "partial_output": bool(app.partial_output),
            },
            version=__version__,
        )

        # ---- terminology candidates (heuristic, local) + user glossary ----
        # Read once: the same text feeds the extractor, the budget and the plan.
        doc_text = doc.plain_text()
        with ui.step("glossary"):
            # A 1000-page book deserves more than a memo's 40 terms; the budget
            # grows with the document (glossary_budget).
            heuristic_terms = extract_glossary(doc_text, max_terms=glossary_budget(len(doc_text)))
            # The pair's own core terms first: every chunk prompt then carries
            # the same target form for them. The user's CSV is applied over it,
            # so an explicit choice always wins.
            user_glossary: dict[str, str] = default_glossary(src, tgt)
            if glossary:
                user_glossary.update(load_glossary_csv(glossary))
                if len(user_glossary) > MAX_USER_GLOSSARY_TERMS:
                    logger.warning(
                        "Glossary has {} entries; keeping the first {} to bound context",
                        len(user_glossary),
                        MAX_USER_GLOSSARY_TERMS,
                    )

        # ---- subsystems ----
        subsystems = self._build_subsystems(ui)
        planner: PlannerAgent = subsystems["planner"]
        skill_opt = SkillOptimizer(SkillStore(app.skills_dir), planner)
        ctx = GraphContext(
            planner=planner,
            translator=subsystems["translator"],
            critic=subsystems["critic"],
            refiner=subsystems["refiner"],
            verifier=subsystems["verifier"],
            ui=ui,
            sink=self.sink,
            reasoning_bank=subsystems["reasoning_bank"],
        )

        chunk_graph = make_chunk_graph(ctx)
        ui.set_total_chunks(len(chunks))
        code_map = doc.metadata.get("code_map", {})
        out_path, pdf_path = output_paths(file_path, tgt, out)

        # ---- checkpoints / resume (before planning: a fully resumed run must
        # ---- not pay for a fresh document plan or a new consistency pass) ----
        store = CheckpointStore(checkpoint_dir or "", enabled=checkpoint_dir is not None)
        # The user's glossary is part of the job, not a detail of the run: a
        # chunk translated without a term the reader has since added is not the
        # same deliverable. Hashing it here makes `--resume` re-ask those chunks
        # instead of replaying them and leaving the new term out.
        glossary_fingerprint = "\n".join(
            f"{term}\t{target}" for term, target in sorted(user_glossary.items())
        )
        doc_key = CheckpointStore.doc_key(
            file_path,
            src=src,
            tgt=tgt,
            model=translator_cfg.model,
            chunk_hashes=[c.hash() for c in chunks],
            glossary=glossary_fingerprint,
        )
        # The page shows the translation as it arrives, chunk by chunk, out of
        # the chunks' own files on disk. It cannot work out where those live:
        # the directory is the pipeline's configuration, which the web UI is
        # not allowed to read. So the run names its own checkpoint directory,
        # once, before the first chunk lands — and a run that keeps no
        # checkpoints says nothing, which reads as "there is nothing to watch".
        if store.usable:
            self.sink.record(
                type="checkpoint",
                doc_key=doc_key,
                path=str(store.dir_for(doc_key)),
            )
        if store.usable and not resume:
            store.clear(doc_key)
        previous_meta = store.load_meta(doc_key) if resume else {}
        chunk_states: list[Any] = [None] * len(chunks)
        if resume and store.usable:
            for i, chunk in enumerate(chunks):
                checkpoint = store.get(doc_key, chunk.id, chunk.hash())
                if checkpoint is not None:
                    # Skip checkpoints with empty translations — they should
                    # be re-translated rather than reused.
                    if not checkpoint.translated or not checkpoint.translated.strip():
                        logger.info(
                            "Skipping empty checkpoint for {}; will re-translate",
                            chunk.id,
                        )
                        continue
                    chunk_states[i] = self._from_checkpoint(chunk, checkpoint)
                    ui.advance()
        pending = [c for i, c in enumerate(chunks) if chunk_states[i] is None]
        resumed_all = not pending
        result.resumed_chunks = len(chunks) - len(pending)

        # One early record the web UI polls on a resumed run: how many chunks
        # it is about to replay from checkpoint, and whether the store had
        # anything to replay at all. Without it the resume's progress bar sits
        # at 0/N (replayed chunks leave no per-chunk events in this run's log)
        # and a resume whose checkpoints no longer match — model changed, file
        # touched — looks like it "started over" with nothing to explain it.
        self.sink.record(
            type="resume",
            resume_requested=bool(resume),
            resumed_chunks=result.resumed_chunks,
            checkpoint_usable=bool(store.usable),
        )

        if resumed_all:
            # Everything was already translated and consistency-patched; the
            # plan only needs to survive into the report.
            logger.info(
                "All {} chunk(s) resumed from checkpoint; skipping planning",
                len(chunks),
            )
            plan = DocumentPlan(
                summary=str(previous_meta.get("summary", "")),
                strategy=str(previous_meta.get("strategy", "") or ""),
                glossary_md="(none)",
                section_count=0,
            )
        else:
            # ---- document plan (ONE Planner request on the unified path) ----
            plan = self._plan(
                ctx=ctx,
                chunks=chunks,
                heuristic_terms=heuristic_terms,
                user_glossary=user_glossary,
                src=src,
                tgt=tgt,
                doc_text=doc_text,
                doc_title=doc.metadata.get("filename", "Document"),
            )
        result.planner_calls = planner.calls
        result.document_complexity = round(plan.complexity, 3)

        glossary_md = plan.glossary_md or "(none)"
        # What the prompts actually carry, not the Planner's own list: the
        # heuristic terms (and every user CSV entry) reach the translator too,
        # so a fallback plan used to report "0 terminology" while sending 65.
        result.glossary_terms = _glossary_entry_count(glossary_md)
        # ---- promoted skill (best-effort: a broken skill.md must not fail the
        # ---- run) — read once, then carried into every chunk's state ----
        try:
            skill = skill_opt.store.load(src, tgt)
        except Exception as exc:  # noqa: BLE001 - optimization is best-effort
            logger.warning("Skill load skipped: {}", exc)
            skill = None
        base_state: dict[str, Any] = {
            "src_lang": src,
            "tgt_lang": tgt,
            "strategy": plan.strategy,
            "few_shot": _merge_few_shot((skill or {}).get("few_shot") or [], plan.few_shot),
            "pitfalls": plan.pitfalls,
            "glossary_md": glossary_md,
            "skill": skill,
        }
        if store.usable:
            store.save_meta(
                doc_key,
                {
                    "file": str(file_path),
                    "src": src,
                    "tgt": tgt,
                    "model": translator_cfg.model,
                    "chunks": len(chunks),
                    "strategy": plan.strategy,
                    "summary": plan.summary,
                    "planner_used_llm": plan.used_llm,
                },
            )

        # ---- per-chunk routing inputs for the chunks still to translate ----
        with ui.step("analyse_chunks"):
            notes = {c.id: plan.note_for(c.section_path) for c in pending}
            per_chunk_state = {
                c.id: {
                    "complexity": score_chunk(
                        c.text,
                        token_count=c.token_count,
                        section_note=notes[c.id],
                        doc_complexity=plan.complexity,
                    ).value,
                    "section_note": notes[c.id],
                }
                for c in pending
            }

        # Chunks finished so far (resumed ones included), so an incremental write
        # always reflects everything complete and marks the rest as pending.
        completed: dict[str, str] = {
            chunk.id: st.get("translated", "")
            for chunk, st in zip(chunks, chunk_states, strict=True)
            if st is not None
        }

        def on_chunk_done(chunk: Any, state: dict[str, Any]) -> None:
            completed[chunk.id] = state.get("translated", "")
            if not app.partial_output or not want_md:
                return
            write_atomic(out_path, assemble_partial(chunks, completed, code_map))

        t0 = time.monotonic()
        if pending:
            pending_states = self._run_chunks(
                chunk_graph,
                pending,
                base_state,
                ui,
                ctx=ctx,
                per_chunk_state=per_chunk_state,
                store=store,
                doc_key=doc_key,
                on_chunk_done=on_chunk_done,
            )
            iterator = iter(pending_states)
            for i, chunk in enumerate(chunks):
                if chunk_states[i] is None:
                    chunk_states[i] = next(iterator)
        elapsed = time.monotonic() - t0

        chunk_states = [st for st in chunk_states if st is not None]
        by_id = {chunk.id: st for chunk, st in zip(chunks, chunk_states, strict=True)}
        translated: dict[str, str] = {
            chunk_id: st.get("translated", "") for chunk_id, st in by_id.items()
        }

        # ---- cross-chunk consistency (bounded, targeted patches) ----
        pairs: dict[str, str] = {p.term: p.translation for p in plan.glossary_pairs}
        pairs.update(user_glossary)  # the user's CSV always wins
        # Not gated on a glossary: the pass also votes on labels and repeated
        # paragraph openings, which need no planned target form.
        if pending and app.consistency_check and chunks:
            checker = ConsistencyChecker(
                ConsistencyAgent(config=self.agent_configs["REFINER"]),
                max_patches=app.consistency_max_patches,
                ui=ui,
                sink=self.sink,
            )
            with ui.step("consistency"):
                translated, report = checker.run(
                    chunks,
                    translated,
                    pairs,
                    context={
                        "src_lang": src,
                        "tgt_lang": tgt,
                        "strategy": plan.strategy,
                        "glossary_md": glossary_md,
                    },
                )
            # One mapping (the report's) fills the result fields, whose names
            # are the run record's keys: no hand-copied list to drift.
            for name, value in report.as_run_fields().items():
                setattr(result, name, value)
            # Keep the checkpoints in sync with what was actually written.
            for chunk in chunks:
                state = by_id.get(chunk.id)
                if state is None or not state.get("translated"):
                    continue
                if translated.get(chunk.id, "") == state.get("translated"):
                    continue
                state["translated"] = translated[chunk.id]
                if store.usable:
                    store.put(doc_key, self._to_checkpoint(chunk, state))

        final: dict[str, Any] = {
            "translated": translated,
            "traces": [st.get("trace", {}) for st in chunk_states],
        }

        # ---- assemble output (Markdown and/or the rebuilt PDF) ----
        with ui.step("assemble_output"):
            if want_md:
                write_atomic(out_path, assemble(chunks, translated, code_map))
            if want_pdf:
                try:
                    stats = rebuild_pdf(
                        file_path,
                        doc,
                        block_translations(chunks, translated),
                        pdf_path,
                        font_path=app.pdf_font_path or None,
                    )
                    result.pdf_path = str(pdf_path)
                    result.pdf_blocks = stats.placed
                    result.pdf_kept = stats.kept
                    result.pdf_failures = list(stats.failures)
                    for failure in stats.failures[:5]:
                        ui.add_error(f"pdf rebuild: {failure}")
                except Exception as exc:  # noqa: BLE001 - a failed rebuild must
                    # not discard a finished translation; the Markdown still is.
                    ui.add_error(f"PDF rebuild failed: {type(exc).__name__}: {exc}")
                    logger.warning("PDF rebuild failed: {}", exc)

        # ---- skillopt update (best-effort; never fails the translation) ----
        # Skipped on a pure resume: nothing new was translated to learn from.
        # Computed once: the SkillOpt update and the report ask the same
        # question of the same traces, and reading them twice was pure work.
        scores = _collect_scores(final)
        try:
            if pending:
                with ui.step("skillopt_update"):
                    for chunk, score in zip(chunks, scores, strict=False):
                        skill_opt.observe(
                            src,
                            tgt,
                            prompt=chunk.text[:200],
                            output=translated.get(chunk.id, "")[:200],
                            critic_score=score,
                        )
                    promoted = skill_opt.maybe_promote(src, tgt)
                    result.skill_promoted = promoted is not None
        except Exception as exc:  # noqa: BLE001 - optimization is best-effort
            logger.warning("SkillOpt update skipped: {}", exc)

        # ---- report ----
        paths = [str(st.get("path", "standard")) for st in chunk_states]
        result.output_path = str(out_path if want_md else pdf_path)
        result.chunks = len(chunks)
        result.avg_critic = (sum(scores) / len(scores)) if scores else 0.0
        result.traces_stored = sum(1 for st in chunk_states if st.get("stored"))
        result.fast_path_chunks = paths.count("fast")
        result.standard_path_chunks = paths.count("standard")
        result.deep_path_chunks = paths.count("deep")
        result.partial_output = bool(app.partial_output)
        result.elapsed_s = round(elapsed, 1)
        result.chunks_failed = sum(
            1 for chunk in chunks if not (translated.get(chunk.id) or "").strip()
        )
        result.fast_path_rejections = ctx.fast_path_rejections
        result.fast_path_top_reason = (
            ctx.fast_path_reasons.most_common(1)[0][0] if ctx.fast_path_reasons else ""
        )
        result.verify_skipped = ctx.verify_skips
        ui.set_stats(
            elapsed_s=round(elapsed, 1),
            avg_critic=round(result.avg_critic, 2),
            traces=result.traces_stored,
            fast_path=result.fast_path_chunks,
            skill_promoted=result.skill_promoted,
        )
        return result


def _merge_few_shot(skill_few_shot: list[str], plan_few_shot: list[str]) -> list[str]:
    """Skill examples first (they won promotion), deduped, cap 6 like the planner's."""
    merged: list[str] = []
    for item in list(skill_few_shot) + list(plan_few_shot or []):
        if item and item not in merged:
            merged.append(item)
    return merged[:6]


def _glossary_entry_count(glossary_md: str) -> int:
    """How many terms the glossary block handed to the prompts actually has.

    One bullet per term, whatever produced it — the user's CSV, the Planner or
    the heuristics. Counting the Planner's pairs instead reported zero on a
    fallback plan that still sent dozens of heuristic terms to every request.
    """
    return sum(1 for line in glossary_md.splitlines() if line.lstrip().startswith("- "))


def _collect_scores(final: dict[str, Any]) -> list[float]:
    """Extract per-chunk quality scores recorded in traces (0-10 scale)."""
    scores: list[float] = []
    for trace in final.get("traces", []):
        s = trace.get("score", 0.0)
        scores.append(float(s) * 10.0)
    return scores
