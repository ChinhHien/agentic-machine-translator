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
import json
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
from polyglot_lc.agents.verifier import TranslationRejected, VerificationError
from polyglot_lc.analysis import DocumentPlan, score_chunk
from polyglot_lc.analysis.unified_planner import (
    PLAN_CACHE_SCHEMA,
    plan_from_dict,
    plan_to_dict,
)
from polyglot_lc.config import (
    ConfigError,
    RetryConfig,
    get_agent_config,
    get_app_config,
    get_embed_config,
    get_retry_config,
)
from polyglot_lc.ingest import load_document
from polyglot_lc.ingest.ai_review import ReviewReport, review_with_llm
from polyglot_lc.llm.client import (
    LLMClient,
    TokenUsage,
    active_api_calls,
    provider_gate_stats,
)
from polyglot_lc.llm.resilience import (
    LLMError,
    NoBackupError,
    PermanentError,
    RetryStats,
    TruncatedError,
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
from polyglot_lc.pipeline.ingest_cache import IngestCache
from polyglot_lc.pipeline.pdf_output import rebuild_pdf
from polyglot_lc.reasoning_bank.embedder import Embedder
from polyglot_lc.reasoning_bank.store import ReasoningBank
from polyglot_lc.segment.chunker import Chunk, chunk_document
from polyglot_lc.segment.glossary import (
    GlossaryTerm,
    default_glossary,
    extract_glossary,
    glossary_budget,
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

#: How many times a chunk is re-run after its reply was cut at the provider's
#: output-token limit, before the halves are tried. The Translator already
#: escalates its own budget twice inside one ask, so by the time the error
#: reaches here the ladder is spent: re-running the whole graph asks the same
#: question with the same ceiling. This had no bound at all, so one chunk whose
#: budget could never fit held the run at its progress bar — paying for the
#: identical failed ask every time — for as long as the reader kept the page open.
_MAX_TRUNCATED_RETRIES = 2

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
    #: Which SkillOpt skill this run used (empty when none was applied), and
    #: what its translations actually scored. ``skill_promoted`` says whether
    #: the run's gradient replaced it at the end.
    skill_id: str = ""
    skill_version: int = 0
    skill_scope: str = ""
    skill_applied: bool = False
    skill_mean: float = 0.0
    skill_samples: int = 0
    skill_promoted: bool = False
    # --- adaptive / planning / durability metrics ---
    planner_calls: int = 0
    document_complexity: float = 0.0
    glossary_terms: int = 0
    fast_path_chunks: int = 0
    standard_path_chunks: int = 0
    deep_path_chunks: int = 0
    resumed_chunks: int = 0
    #: Chunks replayed from the content bank because this job had no checkpoint
    #: of its own — the translation was made under another model or endpoint.
    #: Kept separately from ``resumed_chunks``: those words are not this
    #: attempt's work and must not read as if they were.
    reused_foreign: int = 0
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
    #: Patch attempts allowed; semantic verification uses additional requests.
    consistency_patch_budget: int = 0
    #: True when the pass stopped after two patches in a row were refused — the
    #: rest of its budget was not spent, and the chunks it left are counted as
    #: skipped rather than silently looking consistent.
    consistency_stopped_early: bool = False
    #: Chunks the Verifier could not judge: their drafts are kept, flagged, and
    #: left out of the checkpoint so a resume asks the Verifier again.
    verify_skipped: int = 0
    #: True when this attempt reused the plan saved in its checkpoint instead
    #: of paying for another Planner request.
    plan_reused: bool = False
    #: Counters across every attempt of this translation. Filled at run end;
    #: empty until then. The per-attempt fields above keep their old meaning.
    cumulative: dict[str, Any] = field(default_factory=dict)
    #: Per-chunk scores observed by this attempt, merged into ``cumulative`` so
    #: a chunk retried later does not count twice in the mean.
    skill_scores: dict[str, float] = field(default_factory=dict)
    # Internal hand-off to ``_record_run_end``: prior counters and where to
    # write the merged ones. Excluded from repr because they are not a report.
    previous_stats: dict[str, Any] = field(default_factory=dict, repr=False)
    checkpoint_store: Any = field(default=None, repr=False)
    doc_key: str = field(default="", repr=False)
    partial_output: bool = False
    elapsed_s: float = 0.0
    chunks_failed: int = 0
    #: Chunks whose own request failed in a way no retry can fix (bad key,
    #: malformed prompt, a model name that does not exist). They are holes in the
    #: document, reported as incomplete rather than as a failed run.
    chunks_permanently_failed: int = 0
    chunks_rejected: int = 0
    #: Every chunk the plan expected, how many finished with a verified
    #: checkpoint, and which ones did not — the completion verdict.
    expected_chunks: int = 0
    verified_chunks: int = 0
    checkpointed_chunks: int = 0
    incomplete_chunks: list[dict[str, str]] = field(default_factory=list)
    completion: dict[str, Any] = field(default_factory=dict)
    #: Per-chunk review notes for best-effort translations that did not pass
    #: every gate. The records are deliberately structured: a dashboard can
    #: count causes without parsing prose, and a reviewer can find the exact
    #: chunk in the output.
    quality_review: list[dict[str, str]] = field(default_factory=list)
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


def _endpoint_of(base_url: str) -> str:
    """A URL fit to record: scheme, host, port and path, never a credential.

    The web UI pins this value on a resume, so it has to name the same server
    the checkpoints were keyed by; userinfo and query are dropped because they
    are where a key ends up when a provider wants one in the URL.
    """
    parts = urlparse(str(base_url or ""))
    if not parts.hostname:
        return ""
    netloc = parts.hostname
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    scheme = f"{parts.scheme}://" if parts.scheme else ""
    return f"{scheme}{netloc}{parts.path}".rstrip("/")


def _detect_lang(text: str) -> str:
    """Detect source language with langdetect (falls back to 'en')."""
    try:
        from langdetect import detect

        return detect(text[:2000]) or "en"
    except Exception:
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
        out: str | None = None,
        resume: bool | None = None,
        checkpoint_dir: str | Path | None = None,
        output_format: str | None = None,
        reuse_foreign: bool = True,
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
                    out,
                    ui,
                    result,
                    resume=self.resume if resume is None else bool(resume),
                    checkpoint_dir=checkpoint_dir or self.app.checkpoint_dir,
                    output_format=output_format or self.app.output_format,
                    reuse_foreign=reuse_foreign,
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
        # A crashed attempt never reaches the score pass, but the chunks it did
        # finish carry their scores in the checkpoint. Include those so the
        # cumulative mean covers every measured chunk, not just the last run's.
        if result.checkpoint_store is not None and result.doc_key:
            for chunk_id, checkpoint in result.checkpoint_store.completed(result.doc_key).items():
                result.skill_scores.setdefault(chunk_id, checkpoint.score)
        result.cumulative = _merge_cumulative(
            result.previous_stats, result, usage, stats, duration_ms
        )
        if result.checkpoint_store is not None and result.doc_key:
            result.checkpoint_store.save_stats(result.doc_key, result.cumulative)
        # "Failed" is a verdict on the deliverable, not on the road to it: a run
        # that waited out a provider outage and translated every chunk is done.
        # It reads as failed only when a chunk has no translation or the run
        # stopped on an error. Its problems are still counted below.
        failed = bool(result.chunks_failed or result.fatal_error)
        incomplete = bool(result.incomplete_chunks)
        status = "failed" if failed else ("incomplete" if incomplete else "done")
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
            status=status,
            active_api_calls=calls.get("active"),
            max_active_api_calls=calls.get("peak"),
            # What the per-provider queues settled on: a run that grew to 4
            # slots (or was halved by 429s) says so here instead of leaving it
            # to be guessed from timings.
            provider_slots=provider_gate_stats(),
            exit_code=(1 if failed else (2 if incomplete else 0)),
            duration_ms=duration_ms,
            tokens={
                "sent": usage["sent"],
                "received": usage["received"],
                "cached": usage.get("cached", 0),
                # Whether the provider reported prompt-cache use at all: a run
                # with large prompts and every call "unknown" cannot claim
                # caching works, and saying so is the first half of checking it.
                "cache_hits": usage.get("cache_hits", 0),
                "cache_misses": usage.get("cache_misses", 0),
                "cache_unknown": usage.get("cache_unknown", 0),
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
            chunks_done=max(
                0,
                result.chunks - result.chunks_failed - result.chunks_permanently_failed,
            ),
            chunks_failed=result.chunks_failed,
            chunks_permanently_failed=result.chunks_permanently_failed,
            chunks_rejected=result.chunks_rejected,
            completion=result.completion,
            incomplete_chunks=result.incomplete_chunks,
            fatal_error=result.fatal_error,
            resumed_chunks=result.resumed_chunks,
            reused_foreign=result.reused_foreign,
            plan_reused=result.plan_reused,
            # The consistency pass's own tally, on the record a reader opens
            # first: chunk left drifting and patch refused were only ever
            # visible as lines buried in the log.
            **result.consistency_fields(),
            verify_skipped=result.verify_skipped,
            skill={
                "id": result.skill_id,
                "version": result.skill_version,
                "scope": result.skill_scope,
                "applied": result.skill_applied,
                "mean": round(result.skill_mean, 3),
                "samples": result.skill_samples,
                "promoted": result.skill_promoted,
            },
            quality_review=result.quality_review,
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
            # Per-attempt fields above stay untouched for the supervisor and
            # for an old reader; this is the whole translation's running tally.
            cumulative=result.cumulative,
        )

    def _agent_summaries(self) -> list[dict[str, Any]]:
        """Who ran this run: model, **host**, endpoint, sampling — never a key.

        The host rather than the configured ``base_url`` keeps any credential
        and any path in it out of a file that gets copied, shared and read. The
        ``endpoint`` is the same URL without its userinfo or query, which is
        what a resumed run pins so it keeps talking to the very server that
        produced its checkpoints (the endpoint is part of their identity).
        """
        return [
            {
                "name": name.lower(),
                "model": cfg.model,
                "host": urlparse(cfg.base_url).hostname or "",
                "endpoint": _endpoint_of(cfg.base_url),
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
        except Exception as exc:
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
        core_glossary: dict[str, str],
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
            core_glossary=core_glossary,
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
        """Run independent chunks, isolating quality rejections but propagating fatal errors."""
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
        # A run must not ask a dead provider 76 times, and it must not throw
        # away 75 good chunks because three hit a blip either. The streak counts
        # consecutive "the provider is gone" failures; the config says how many
        # in a row mean the run itself is over.
        outage = {"streak": 0}
        outage_lock = threading.Lock()
        stop_after_failures = max(1, int(self.app.chunk_stop_after_failures))

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

        def wait_verify_retry(chunk: Any, attempt: int) -> None:
            """Wait before asking the Verifier again, without re-translating.

            A verifier outage says nothing about the draft, so the retry asks
            the same draft again. The wait doubles and is sliced, so the stop
            button cuts it short like every other wait in this loop.
            """
            delay = min(60.0, self.app.verify_retry_delay * (2 ** (attempt - 1)))
            where = ctx.last_step(chunk.id) if ctx is not None else ""
            logger.warning(
                "Chunk {} verification unavailable; re-checking the same draft in {:.0f}s",
                chunk.id,
                delay,
            )
            ui.retry(
                where,
                chunk.id,
                f"verification unavailable - re-checking in {delay:.0f}s",
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
            truncated_retries = 0
            verify_retries = 0
            attempt_limit = max(1, int(self.app.chunk_max_attempts))
            last_reason = ""
            while True:
                if self._stop.is_set():
                    # Another chunk hit something no retry can fix, so the run
                    # is over: this chunk is not asked again.
                    return {"translated": "", "chunk": chunk, "error": "run stopped"}, False, True
                if attempt >= attempt_limit:
                    # The provider has said no this many times: asking again is
                    # the same question, and waiting it out forever is how a
                    # 74-chunk run held a whole morning without writing a file.
                    reason = f"gave up after {attempt} attempt(s)"
                    if last_reason:
                        reason = f"{reason}: {last_reason}"
                    logger.warning(
                        "chunk={} {}; leaving it untranslated for this run "
                        "(CHUNK_MAX_ATTEMPTS={}) and out of the checkpoint",
                        chunk.id,
                        reason,
                        attempt_limit,
                    )
                    ui.add_error(f"{chunk.id}: {reason} - left untranslated")
                    return (
                        {"translated": "", "chunk": chunk, "error": reason},
                        True,
                        False,
                    )
                attempt += 1
                try:
                    out = translate_chunk(chunk)
                except LLMError as exc:
                    last_reason = f"{type(exc).__name__}: {exc}"
                    ui.add_error(f"{chunk.id}: {type(exc).__name__}: {exc}")
                    if isinstance(exc, PermanentError):
                        # This chunk's request is wrong in a way no retry fixes
                        # (bad key, malformed prompt, a model name that does not
                        # exist). It is one hole in the document, not an outage:
                        # leave it untranslated and out of the checkpoint, let
                        # the other chunks finish, and report the run incomplete
                        # (exit 2) so --resume fills exactly this gap. Stopping
                        # every worker here ended a 76-chunk run after 27 chunks.
                        logger.warning(
                            "chunk={} failed permanently ({}); leaving it untranslated "
                            "for this run and out of the checkpoint",
                            chunk.id,
                            last_reason,
                        )
                        return (
                            {
                                "translated": "",
                                "chunk": chunk,
                                "error": str(exc),
                                "permanent_failure": True,
                            },
                            True,
                            False,
                        )
                    if isinstance(exc, NoBackupError):
                        # The attempts are spent and there is no backup endpoint
                        # to switch to. One chunk losing that race is a hole the
                        # file can live with; a provider that keeps failing is
                        # an outage, and asking it 73 more times is not a retry,
                        # it is a decision to waste the rest of the run.
                        with outage_lock:
                            outage["streak"] += 1
                            streak = outage["streak"]
                        if streak >= stop_after_failures:
                            logger.error(
                                "chunk={} failed with no endpoint left ({} in a row); "
                                "stopping the run",
                                chunk.id,
                                streak,
                            )
                            self._stop.set()
                            return (
                                {"translated": "", "chunk": chunk, "error": str(exc)},
                                False,
                                True,
                            )
                        logger.warning(
                            "chunk={} has no endpoint left ({} in a row); leaving it "
                            "untranslated for this run and out of the checkpoint",
                            chunk.id,
                            streak,
                        )
                        return (
                            {
                                "translated": "",
                                "chunk": chunk,
                                "error": str(exc),
                                "permanent_failure": True,
                            },
                            True,
                            False,
                        )
                    if isinstance(exc, TruncatedError):
                        # The Translator's own budget ladder is spent by the time
                        # the error reaches here, so re-running the whole graph
                        # asks the same question with the same ceiling. A shorter
                        # question IS a different question: try the halves, and if
                        # they cannot cover the chunk either, leave a visible
                        # marker instead of asking forever.
                        if truncated_retries >= _MAX_TRUNCATED_RETRIES:
                            salvaged = salvage_by_split(chunk, depth)
                            if salvaged is not None:
                                return {"translated": salvaged, "chunk": chunk}, False, False
                            logger.warning(
                                "chunk={} reply kept being cut at the output-token "
                                "limit; leaving it untranslated for this run and out "
                                "of the checkpoint",
                                chunk.id,
                            )
                            ui.add_error(
                                f"{chunk.id}: reply cut at the output-token limit - "
                                "left untranslated"
                            )
                            return (
                                {"translated": "", "chunk": chunk, "error": str(exc)},
                                True,
                                False,
                            )
                        truncated_retries += 1
                    wait_and_retry(chunk, attempt, f"{type(exc).__name__}: {exc}")
                    if self._stop.is_set():
                        return {"translated": "", "chunk": chunk, "error": str(exc)}, False, True
                    continue
                translated = out.get("translated", "") or ""
                if translated.strip():
                    reason = coverage_error(chunk, translated) if measure_coverage else None
                    if reason is None:
                        # A verifier outage is retried here, on the same draft:
                        # translator_node reuses ``kept_draft``, so the only
                        # new request is the verdict that never arrived.
                        if (
                            out.get("verify_unavailable")
                            and verify_retries < self.app.verify_unavailable_retries
                        ):
                            verify_retries += 1
                            wait_verify_retry(chunk, verify_retries)
                            if self._stop.is_set():
                                return out, True, True
                            continue
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
                    # Verification reads the profile: half a reference list is
                    # still a reference list, not body prose.
                    profile=getattr(chunk, "profile", "body"),
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
            try:
                out, kept_partial, aborted = settle(chunk, 0)
            except TranslationRejected as exc:
                # TranslationRejected subclasses VerificationError: catch it
                # first, or a quality rejection would be mistaken for an
                # unavailable Verifier.
                ui.add_error(f"{chunk.id}: {exc}")
                # Keep the best draft even when the gate refused it. A reader
                # can review imperfect text; a marker cannot recover words.
                best = ctx.drafts.get(chunk.id, "") if ctx is not None else ""
                self._save_rejected(store, doc_key, chunk, best, str(exc))
                out = {
                    "translated": best,
                    "chunk": chunk,
                    "error": str(exc),
                    "quality_rejected": True,
                    "rejection_reason": str(exc),
                }
                kept_partial, aborted = True, False
            except VerificationError as exc:
                # A custom/test graph can raise where the production graph now
                # returns a skipped verdict. Treat it the same: keep the best
                # draft, flag it, and let the rest of the document finish.
                message = f"verification unavailable: {exc}"
                add_error = getattr(ui, "add_error", None)
                if callable(add_error):
                    add_error(f"{chunk.id}: {message}")
                best = ctx.drafts.get(chunk.id, "") if ctx is not None else ""
                if ctx is not None:
                    ctx.note_verify_skip(chunk.id)
                    ctx.record_error(step="verify", chunk_id=chunk.id, cause="verify", message=message)
                out = {
                    "translated": best,
                    "chunk": chunk,
                    "error": message,
                    "quality_rejected": False,
                    "verify_unavailable": True,
                }
                kept_partial, aborted = True, False
            if not aborted and (out.get("translated") or "").strip():
                # The provider answered: whatever streak it had is over.
                with outage_lock:
                    outage["streak"] = 0
            if aborted:
                return out
            # Unverified text is kept for the reader but never checkpointed:
            # a resume must ask the Verifier again rather than replay it.
            if out.get("verify_unavailable"):
                kept_partial = True
            # A translation — whole or salvaged — is worth keeping: a resume
            # never has to translate the chunk again. A partial one that
            # failed the coverage check is kept out of the checkpoint so a
            # later --resume can still recover the words it never got.
            if store is not None and store.usable and not kept_partial:
                store.put(doc_key, self._to_checkpoint(chunk, out))
            if (
                ctx is not None
                and not out.get("verify_unavailable")
                and not out.get("quality_rejected")
            ):
                # A chunk that eventually passed was not "never verified": the
                # retry worked, so the warning must not outlive it.
                ctx.clear_verify_skip(chunk.id)
            if on_chunk_done is not None:
                on_chunk_done(chunk, out)
            return out

        results: list[Any] = [None] * len(chunks)
        workers = max(1, self.app.max_concurrency)
        # The executor already has at most ``workers`` threads. This gate makes
        # the invariant explicit at the chunk boundary as well: a worker holds
        # one permit for the whole translate/verify/checkpoint lifecycle and
        # cannot pick up another chunk until ``run_one`` returns. Keeping this
        # local to one run also means completion sweeps and a later run can
        # never silently accumulate extra workers.
        chunk_slots = threading.BoundedSemaphore(workers)
        slots = get_retry_config().provider_concurrency
        if workers > slots:
            # Both knobs default to 1 and the settings dialog exposes both, so
            # raising only the worker count buys nothing and says nothing.
            logger.warning(
                "MAX_CONCURRENCY={} but LLM_PROVIDER_CONCURRENCY={}: {} chunk workers "
                "queue on {} provider slot(s), so the run is effectively serialised; "
                "raise LLM_PROVIDER_CONCURRENCY to the provider's real limit",
                workers,
                slots,
                workers,
                slots,
            )
        if workers == 1 or len(chunks) <= 1:
            for i, chunk in enumerate(chunks):
                results[i] = run_one(i, chunk)
                ui.advance()
        else:
            def run_in_slot(index: int, chunk: Any) -> dict[str, Any]:
                """Hold one chunk slot until this chunk is completely finished."""
                with chunk_slots:
                    return run_one(index, chunk)

            with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
                futures = {
                    pool.submit(run_in_slot, i, chunk): i for i, chunk in enumerate(chunks)
                }
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

        # ---- completion sweeps ----
        # A verifier outage that outlasted the per-chunk retries is retried
        # here at document level: the drafts are already in hand, so each round
        # re-verifies them (and only retranslates a chunk the Verifier actually
        # rejected). No sweep means no unbounded spending, and a chunk that is
        # still not verified is reported as incomplete rather than as done.
        def needs_completion(index: int, chunk: Any) -> bool:
            out = results[index] or {}
            if out.get("permanent_failure"):
                # Asked and answered: a bad request does not become a good one
                # by waiting, and the sweeps exist for outages, not for holes.
                return False
            if out.get("verify_unavailable") or out.get("quality_rejected"):
                return True
            if not (out.get("translated") or "").strip():
                return True
            if store is not None and store.usable:
                return store.get(doc_key, chunk.id, chunk.hash()) is None
            return False

        # A definitive rejection gets **one** extra try across the sweeps. A
        # verifier that rejected the same draft twice is unlikely to change its
        # mind just because time passed, and repeating it would only spend
        # translation calls. An unavailable verifier has no verdict yet, so it
        # is worth retrying in every round.
        extra_retry_used: set[str] = set()
        completion_started = time.monotonic()
        deadline = self.app.completion_deadline
        deadline_hit = False
        for round_no in range(1, max(0, self.app.completion_rounds) + 1):
            if self._stop.is_set():
                break
            if deadline > 0 and time.monotonic() - completion_started >= deadline:
                deadline_hit = True
                self.sink.info(
                    step="completion",
                    message=(
                        "completion deadline reached after "
                        f"{deadline:.0f}s; leaving the remaining chunks incomplete"
                    ),
                )
                logger.warning(
                    "Completion deadline reached after {:.0f}s; "
                    "leaving remaining chunks incomplete",
                    deadline,
                )
                break
            pending = [
                i
                for i, chunk in enumerate(chunks)
                if needs_completion(i, chunk)
                and not (
                    (results[i] or {}).get("quality_rejected")
                    and chunk.id in extra_retry_used
                )
            ]
            if not pending:
                break
            delay = min(120.0, self.app.completion_retry_delay * round_no)
            if delay > 0:
                self.sink.info(
                    step="completion",
                    message=(
                        f"completion sweep {round_no}: {len(pending)} chunk(s) chưa xác minh; "
                        f"thử lại sau {delay:.0f}s"
                    ),
                )
                waited = 0.0
                while waited < delay and not self._stop.is_set():
                    slice_s = min(_RETRY_TICK, delay - waited)
                    time.sleep(slice_s)
                    waited += slice_s
            if self._stop.is_set():
                break
            self.sink.record(
                type="completion_round",
                round=round_no,
                chunks=len(pending),
                ids=[chunks[i].id for i in pending[:50]],
            )
            for i in pending:
                if self._stop.is_set():
                    break
                if deadline > 0 and time.monotonic() - completion_started >= deadline:
                    deadline_hit = True
                    break
                if (results[i] or {}).get("quality_rejected"):
                    extra_retry_used.add(chunks[i].id)
                results[i] = run_one(i, chunks[i])
            if deadline_hit:
                self.sink.info(
                    step="completion",
                    message=(
                        "completion deadline reached after "
                        f"{deadline:.0f}s; leaving the remaining chunks incomplete"
                    ),
                )
                logger.warning(
                    "Completion deadline reached after {:.0f}s; "
                    "leaving remaining chunks incomplete",
                    deadline,
                )
                break
        return results

    @staticmethod
    def _save_rejected(
        store: CheckpointStore | None,
        doc_key: str,
        chunk: Any,
        translation: str,
        reason: str,
    ) -> None:
        """Keep a rejected draft on disk for review and later repair.

        The checkpoint intentionally stays empty: ``--resume`` must retry the
        chunk. This side file is evidence, not a resume shortcut.
        """
        if not translation.strip():
            return
        try:
            if store is not None:
                directory = store.dir_for(doc_key).parent / "rejected" / doc_key
            else:
                directory = Path("data") / "rejected" / (doc_key or "unknown")
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "chunk_id": chunk.id,
                "source": chunk.text,
                "translated": translation,
                "reason": reason,
                "ts": time.time(),
            }
            (directory / f"{chunk.id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Could not save rejected draft for {}: {}", chunk.id, exc)

    @staticmethod
    def _incomplete_chunks(
        chunks: list[Any],
        states: dict[str, dict[str, Any]],
        store: CheckpointStore | None,
        doc_key: str,
    ) -> list[dict[str, str]]:
        """Every planned chunk that is not finished and verified.

        A checkpoint is only written for a chunk the pipeline accepted, so its
        presence is the run's own receipt for "verified". When checkpointing is
        disabled there is no receipt to read; the chunk state's flags are then
        the only evidence, and they are still enough to refuse a false done.
        """
        missing: list[dict[str, str]] = []
        for chunk in chunks:
            state = states.get(chunk.id) or {}
            reason = ""
            if state.get("permanent_failure"):
                reason = "translation failed permanently: " + str(state.get("error") or "")[:300]
            elif not (state.get("translated") or "").strip():
                reason = "missing translation"
            elif state.get("quality_rejected"):
                reason = "verification rejected: " + (
                    str(state.get("rejection_reason") or "no acceptable translation")[:300]
                )
            elif state.get("verify_unavailable"):
                reason = "verification unavailable"
            elif store is not None and store.usable:
                if store.get(doc_key, chunk.id, chunk.hash()) is None:
                    reason = "no verified checkpoint"
            if reason:
                missing.append({"chunk": chunk.id, "reason": reason})
        return missing

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
    @staticmethod
    def _review_ingest(doc: Any) -> ReviewReport | None:
        """Best-effort AI repair of extracted blocks; never fails the run."""
        try:
            cfg = get_agent_config("TRANSLATOR").model_copy(
                update={"name": "INGEST_REVIEW", "temperature": 0.0}
            )
            app = get_app_config()
            report = review_with_llm(
                doc,
                LLMClient(cfg),
                max_blocks=app.ingest_ai_review_max_blocks,
                max_chars=app.ingest_ai_review_max_chars,
            )
        except Exception as exc:
            logger.warning("AI ingest review skipped: {}", exc)
            return None
        logger.info(
            "AI ingest review | candidates={} reviewed={} changed={} rejected={} failed={}",
            report.candidates,
            report.reviewed,
            report.changed,
            report.rejected,
            report.failed,
        )
        return report

    def _run_inner(
        self,
        file_path: str | Path,
        src: str,
        tgt: str,
        out: str | None,
        ui: SimpleUI,
        result: PipelineResult,
        *,
        resume: bool = False,
        checkpoint_dir: str | Path | None = None,
        output_format: str = "md",
        reuse_foreign: bool = True,
    ) -> PipelineResult:
        app = self.app
        fmt = (output_format or "md").strip().lower()
        want_md, want_pdf = fmt in {"md", "both"}, fmt in {"pdf", "both"}
        source_path = Path(file_path)

        # ---- ingest ----
        # A resume must not read the whole document again. Extraction is minutes
        # on a long book, and the optional AI review would rewrite block text —
        # changing every chunk hash and, with it, the checkpoint identity the
        # resume is about to look up. Both are cached by content (see
        # ``pipeline.ingest_cache``).
        ingest_cache = IngestCache(checkpoint_dir, enabled=checkpoint_dir is not None)
        sha256 = _file_sha256(source_path)
        cached = ingest_cache.load(sha256) if resume else None
        heuristic_terms: list[GlossaryTerm] = []
        with ui.step("ingest") as _:
            if cached is not None:
                doc, heuristic_terms = cached
                # The cache is keyed by content, not by name: the same bytes
                # under another path must still present themselves as this file.
                doc.metadata["source"] = str(source_path)
                doc.metadata["filename"] = source_path.name
                logger.info(
                    "Ingest cache: reusing {} block(s) extracted for {}",
                    len(doc.blocks),
                    source_path.name,
                )
                self.sink.info(
                    step="ingest",
                    message="dùng lại bản trích xuất đã lưu — không đọc lại tài liệu",
                )
                ui.set_stats(file=source_path.name, blocks=len(doc.blocks), cached_ingest=True)
            else:
                doc = load_document(file_path)
                ui.set_stats(file=doc.metadata.get("filename", ""), blocks=len(doc.blocks))
                if app.ingest_ai_review:
                    # The extractor has already attached page geometry to every
                    # block; review only rewrites block text in place and never
                    # re-runs extraction, so the rebuild path keeps its rectangles.
                    report = self._review_ingest(doc)
                    if report is not None:
                        ui.set_stats(
                            ai_review_changed=report.changed,
                            ai_review_rejected=report.rejected,
                        )

        # ---- language detection ----
        if src == "auto":
            with ui.step("detect_lang"):
                src = _detect_lang(doc.plain_text())
        ui.set_stats(src=src, tgt=tgt)

        # ---- chunking (pinned tokenizer, budget from CHUNK_MAX_TOKENS) ----
        translator_cfg = self.agent_configs["TRANSLATOR"]
        with ui.step("segment"):
            chunks = chunk_document(
                doc,
                model=translator_cfg.model,
                base_url=translator_cfg.base_url,
            )
            ui.set_stats(chunks=len(chunks))
        result.chunks = len(chunks)
        result.chunks_failed = len(chunks)
        result.partial_output = bool(app.partial_output)

        # ---- promoted skill (best-effort: a broken skill.md must not fail the
        # ---- run) — read before the run's first record, so every event and
        # ---- the report can name the skill that shaped this translation ----
        skill_store = SkillStore(app.skills_dir)
        try:
            skill = skill_store.load() if app.skill_apply else None
        except Exception as exc:
            logger.warning("Skill load skipped: {}", exc)
            skill = None
        result.skill_applied = skill is not None
        if skill is not None:
            result.skill_id = str(skill.get("skill_id", ""))
            result.skill_version = int(skill.get("version", 0) or 0)
            result.skill_scope = str(skill.get("scope", ""))

        # ---- what this run was asked to do, while it is still true ----
        source_path = Path(file_path)
        self.sink.record(
            type="run_start",
            file=source_path.name,
            sha256=sha256,
            bytes=source_path.stat().st_size if source_path.is_file() else 0,
            pages=doc.metadata.get("pages"),
            chunks=len(chunks),
            plan_hash=plan_hash(chunks),
            src=src,
            tgt=tgt,
            agents=self._agent_summaries(),
            max_tokens=app.chunk_max_tokens,
            flags={
                "resume": bool(resume),
                "format": fmt,
                "partial_output": bool(app.partial_output),
            },
            skill={
                "id": result.skill_id,
                "version": result.skill_version,
                "scope": result.skill_scope,
                "applied": result.skill_applied,
            },
            version=__version__,
        )

        # ---- terminology candidates (heuristic, local) + the pair's core terms ----
        # Read once: the same text feeds the extractor, the budget and the plan.
        doc_text = doc.plain_text()
        with ui.step("glossary"):
            # A 1000-page book deserves more than a memo's 40 terms; the budget
            # grows with the document (glossary_budget).
            if not heuristic_terms:
                heuristic_terms = extract_glossary(
                    doc_text, max_terms=glossary_budget(len(doc_text))
                )
            # The pair's own core terms first: every chunk prompt then carries
            # the same target form for them. The document's own candidates come
            # from the extractor above; there is no per-run override here.
            core_glossary: dict[str, str] = default_glossary(src, tgt)
        if cached is None:
            # Before anything can fail: the next attempt reuses this extraction
            # instead of paying for it again.
            ingest_cache.save(sha256, doc, heuristic_terms)

        # ---- subsystems ----
        subsystems = self._build_subsystems(ui)
        planner: PlannerAgent = subsystems["planner"]
        skill_opt = SkillOptimizer(skill_store, planner)
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
        # The glossary is part of the job, not a detail of the run: a chunk
        # translated against other terms is not the same deliverable. Hashing it
        # here makes `--resume` re-ask those chunks instead of replaying them.
        glossary_fingerprint = "\n".join(
            f"{term}\t{target}" for term, target in sorted(core_glossary.items())
        )
        doc_key = CheckpointStore.doc_key(
            file_path,
            src=src,
            tgt=tgt,
            model=translator_cfg.model,
            chunk_hashes=[c.hash() for c in chunks],
            glossary=glossary_fingerprint,
            skip_llm_verify=app.fast_path_skip_llm_verify,
            base_url=translator_cfg.base_url,
            skill_id=result.skill_id,
            skill_applied=result.skill_applied,
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
        # Provenance for every checkpoint this store writes: which model's words
        # a chunk carries is part of its record, not of the job's identity.
        store.model = translator_cfg.model
        store.base_url = translator_cfg.base_url
        # Everything the deliverable is, minus the model and endpoint: what the
        # content bank matches on when a chunk is carried over from another
        # model. A changed glossary or verification policy is *not* a match.
        store.content_key = CheckpointStore.content_key(
            file_path,
            src=src,
            tgt=tgt,
            chunk_hashes=[c.hash() for c in chunks],
            glossary=glossary_fingerprint,
            skip_llm_verify=app.fast_path_skip_llm_verify,
            skill_id=result.skill_id,
            skill_applied=result.skill_applied,
        )
        if store.usable and not resume:
            store.clear(doc_key)
        previous_meta = store.load_meta(doc_key) if resume else {}
        previous_stats = store.load_stats(doc_key) if resume else {}
        # Hand the prior counters and the checkpoint location to the run-end
        # recorder: it writes the merged totals after this attempt finishes.
        result.previous_stats = previous_stats
        result.checkpoint_store = store if store.usable else None
        result.doc_key = doc_key
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
        reused_foreign = 0
        if resume and store.usable and reuse_foreign:
            for i, chunk in enumerate(chunks):
                if chunk_states[i] is not None:
                    continue
                donor = store.bank_get(chunk.hash())
                if donor is None or donor.source_doc_key == doc_key:
                    continue
                # Another model or endpoint translated this exact text. Reusing
                # it is what pressing "Tiếp tục" is for; the chunk keeps its own
                # provenance so the report says whose words these are, and it is
                # adopted into this job's checkpoints so the completion verdict
                # (and a later PDF rebuild) sees it as finished.
                chunk_states[i] = self._from_checkpoint(chunk, donor)
                chunk_states[i]["from_other_model"] = True
                store.put(doc_key, donor, bank=False)
                ui.advance()
                reused_foreign += 1
                logger.info(
                    "chunk={} reused from the content bank (model {}); not asked again",
                    chunk.id,
                    donor.model or "unknown",
                )
        result.reused_foreign = reused_foreign
        pending = [c for i, c in enumerate(chunks) if chunk_states[i] is None]
        resumed_all = not pending
        result.resumed_chunks = len(chunks) - len(pending)
        result.chunks_failed = len(pending)

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
            reused_foreign=result.reused_foreign,
            checkpoint_usable=bool(store.usable),
        )

        cached_plan = (
            _load_cached_plan(previous_meta, chunks, src, tgt, glossary_fingerprint)
            if resume
            else None
        )
        if cached_plan is not None:
            # The same translation job keeps the plan its first attempt made:
            # the remaining chunks must see the same glossary and section notes,
            # and a resumed run should not pay for the Planner again.
            plan = cached_plan
            result.plan_reused = True
            logger.info("Resume reused the saved document plan; skipping planning")
            self.sink.record(type="plan", reused=True, planner_calls=0)
        elif resumed_all:
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
            self.sink.record(type="plan", reused=False, planner_calls=0)
        else:
            # ---- document plan (ONE Planner request on the unified path) ----
            plan = self._plan(
                ctx=ctx,
                chunks=chunks,
                heuristic_terms=heuristic_terms,
                core_glossary=core_glossary,
                src=src,
                tgt=tgt,
                doc_text=doc_text,
                doc_title=doc.metadata.get("filename", "Document"),
            )
            self.sink.record(type="plan", reused=False, planner_calls=planner.calls)
        result.planner_calls = planner.calls
        result.document_complexity = round(plan.complexity, 3)

        glossary_md = plan.glossary_md or "(none)"
        # What the prompts actually carry, not the Planner's own list: the
        # heuristic terms (and the pair's core terms) reach the translator too,
        # so a fallback plan used to report "0 terminology" while sending 65.
        result.glossary_terms = _glossary_entry_count(glossary_md)
        # ---- the skill read before the run's first record: carried into every
        # ---- chunk's state, unchanged for the whole run ----
        base_state: dict[str, Any] = {
            "src_lang": src,
            "tgt_lang": tgt,
            "strategy": plan.strategy,
            "few_shot": _merge_few_shot((skill or {}).get("few_shot") or [], plan.few_shot),
            "pitfalls": plan.pitfalls,
            "glossary_md": glossary_md,
            "glossary_pairs": {
                **{p.term: p.translation for p in plan.glossary_pairs},
                **core_glossary,
            },
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
                    # Never replace a real cached plan with the report-only stub
                    # used by an old fully-resumed checkpoint.
                    **(
                        {}
                        if resumed_all and cached_plan is None
                        else {
                            "plan": {
                                "schema": PLAN_CACHE_SCHEMA,
                                "plan_hash": plan_hash(chunks),
                                "src": src,
                                "tgt": tgt,
                                "glossary": glossary_fingerprint,
                                # Also part of ``doc_key``; kept here so a
                                # reader of meta.json can see which skill
                                # shaped the cached plan.
                                "skill_id": result.skill_id,
                                "skill_applied": result.skill_applied,
                                "data": plan_to_dict(plan),
                            }
                        }
                    ),
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

        rejected: set[str] = set()
        # ``on_chunk_done`` runs on every worker thread. Guard the shared maps,
        # and write partial snapshots in generation order so a slow older write
        # cannot overwrite a newer one.
        progress_lock = threading.Lock()
        partial_lock = threading.Lock()
        partial_generation = 0
        last_partial_generation = 0

        def on_chunk_done(chunk: Any, state: dict[str, Any]) -> None:
            nonlocal partial_generation, last_partial_generation
            with progress_lock:
                completed[chunk.id] = state.get("translated", "")
                if state.get("quality_rejected"):
                    rejected.add(chunk.id)
                else:
                    # A later completion pass may have verified the chunk; the
                    # rejection was this chunk's earlier attempt, not its result.
                    rejected.discard(chunk.id)
                partial_generation += 1
                generation = partial_generation
                snapshot = dict(completed)
            if not app.partial_output or not want_md:
                return
            with partial_lock:
                if generation < last_partial_generation:
                    return
                try:
                    write_atomic(out_path, assemble_partial(chunks, snapshot, code_map))
                except OSError as exc:
                    # A live preview is a convenience; failing to write it must
                    # not abort a run whose final assembly can still succeed.
                    logger.warning("partial output write skipped: {}", exc)
                    self.sink.info(
                        step="partial_output",
                        message=f"could not write partial output: {exc}",
                    )
                    return
                last_partial_generation = generation

        # ---- preflight ----
        # A wrong key, a model that does not exist, an endpoint that is down:
        # each used to surface at the chunk that happened to hit it, after the
        # plan and dozens of translations were paid for (chunk-073 discovered a
        # bad model name). Three short calls buy that answer before the run
        # spends anything real.
        if pending and app.preflight:
            from polyglot_lc.pipeline.preflight import run_preflight

            with ui.step("preflight") as _:
                check = run_preflight(self.agent_configs)
            self.sink.record(
                type="step",
                step="preflight",
                status="ok" if check.ok else "fail",
                error=check.summary(),
                checks=check.as_dicts(),
            )
            if not check.ok:
                result.chunks_failed = len(chunks)
                result.fatal_error = check.summary()
                ui.add_error(check.summary())
                logger.error("Preflight failed; no chunk was translated | {}", check.summary())
                return result

        t0 = time.monotonic()
        try:
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
        finally:
            result.elapsed_s = round(time.monotonic() - t0, 1)
            # A chunk whose request failed permanently is a hole in the
            # document, not a broken run: it is counted as incomplete below, so
            # the file still ships (exit 2) and ``--resume`` fills exactly that
            # gap. Counting it here made one bad chunk end a whole run with 1.
            permanent = {
                str(getattr(state.get("chunk"), "id", ""))
                for state in chunk_states
                if state is not None and state.get("permanent_failure")
            }
            result.chunks_permanently_failed = len(permanent)
            result.chunks_failed = sum(
                1
                for chunk in chunks
                if not (completed.get(chunk.id) or "").strip() and chunk.id not in permanent
            )
            result.chunks_rejected = len(rejected)
            # Unique chunks, not attempts: a verification retry is the same
            # chunk asking twice, and counting both would overstate the gap.
            result.verify_skipped = sum(
                1 for st in chunk_states if st is not None and st.get("verify_unavailable")
            )
            result.fast_path_rejections = ctx.fast_path_rejections
            result.fast_path_top_reason = (
                ctx.fast_path_reasons.most_common(1)[0][0] if ctx.fast_path_reasons else ""
            )

        chunk_states = [st for st in chunk_states if st is not None]
        by_id = {chunk.id: st for chunk, st in zip(chunks, chunk_states, strict=True)}
        result.quality_review = [
            {
                "chunk": chunk.id,
                "severity": str(getattr(st.get("verify_result"), "severity", "major")),
                "reason": str(st.get("rejection_reason", "")),
            }
            for chunk, st in zip(chunks, chunk_states, strict=True)
            if st.get("quality_rejected")
        ]
        translated: dict[str, str] = {
            chunk_id: st.get("translated", "") for chunk_id, st in by_id.items()
        }

        # ---- cross-chunk consistency (bounded, targeted patches) ----
        pairs: dict[str, str] = {p.term: p.translation for p in plan.glossary_pairs}
        pairs.update(core_glossary)  # the pair's core terms always win
        # Not gated on a glossary: the pass also votes on labels and repeated
        # paragraph openings, which need no planned target form.
        if pending and app.consistency_check and chunks:
            checker = ConsistencyChecker(
                ConsistencyAgent(config=self.agent_configs["REFINER"]),
                ctx.verifier,
                max_patches=app.consistency_max_patches,
                ui=ui,
                sink=self.sink,
            )
            with ui.step("consistency"):
                full_translations = dict(translated)
                eligible = [chunk for chunk in chunks if chunk.id not in rejected]
                consistency_translations = {
                    chunk.id: translated[chunk.id]
                    for chunk in eligible
                    if translated.get(chunk.id, "").strip()
                }
                patched, report = checker.run(
                    eligible,
                    consistency_translations,
                    pairs,
                    context={
                        "src_lang": src,
                        "tgt_lang": tgt,
                        "strategy": plan.strategy,
                        "glossary_md": glossary_md,
                    },
                )
                translated = full_translations
                translated.update(patched)
            # One mapping (the report's) fills the result fields, whose names
            # are the run record's keys: no hand-copied list to drift.
            for name, value in report.as_run_fields().items():
                setattr(result, name, value)
            # Keep the checkpoints in sync with what was actually written.
            for chunk in chunks:
                state = by_id.get(chunk.id)
                if chunk.id in rejected:
                    continue
                if state is None or not state.get("translated"):
                    continue
                if translated.get(chunk.id, "") == state.get("translated"):
                    continue
                state["translated"] = translated[chunk.id]
                if store.usable:
                    store.put(doc_key, self._to_checkpoint(chunk, state))

        # ---- completion verdict ----
        # Every planned chunk must have its own verified checkpoint. This is
        # the gate the reader sees: a run with a draft for everything but a
        # verifier outage on part of it is incomplete, not done.
        result.expected_chunks = len(chunks)
        result.incomplete_chunks = self._incomplete_chunks(chunks, by_id, store, doc_key)
        result.verified_chunks = max(0, len(chunks) - len(result.incomplete_chunks))
        result.checkpointed_chunks = (
            len(chunks) - len(store.missing(doc_key, chunks)) if store is not None and store.usable else 0
        )
        result.completion = {
            "expected": result.expected_chunks,
            "verified": result.verified_chunks,
            "checkpointed": result.checkpointed_chunks,
            "reused_from_other_model": result.reused_foreign,
            "incomplete": result.incomplete_chunks,
        }

        # ---- assemble output (Markdown and/or the rebuilt PDF) ----
        pdf_ok = True
        pdf_error = ""
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
                    pdf_ok = pdf_path.is_file()
                    if pdf_ok:
                        result.pdf_path = str(pdf_path)
                        result.pdf_blocks = stats.placed
                        result.pdf_kept = stats.kept
                    else:
                        pdf_error = "rebuild produced no PDF file"
                    result.pdf_failures = list(stats.failures)
                    for failure in stats.failures[:5]:
                        ui.add_error(f"pdf rebuild: {failure}")
                except Exception as exc:
                    # not discard a finished translation; the Markdown still is.
                    pdf_ok = False
                    pdf_error = f"{type(exc).__name__}: {exc}"
                    ui.add_error(f"PDF rebuild failed: {pdf_error}")
                    logger.warning("PDF rebuild failed: {}", exc)
            if want_pdf and not want_md and not pdf_ok:
                # PDF was the only deliverable. Reporting the run as complete
                # with no file (or a stale one) would be a lie.
                result.fatal_error = f"PDF rebuild failed: {pdf_error or 'no output file'}"
                result.output_path = ""

        # ---- skillopt update (best-effort; never fails the translation) ----
        # A chunk with no trace score is not a zero: it was not measured, and
        # averaging it in would drag every mean down as a provider fails. Use
        # the scores actually observed, keyed by chunk, for both the report and
        # the optimizer. Resumed chunks are not this run's evidence; only the
        # chunks translated now are observed.
        scores = _collect_scores(chunks, by_id)
        if scores:
            result.skill_mean = sum(scores.values()) / len(scores)
            result.skill_samples = len(scores)
        # The SkillOpt gate reads the retry counters; capture them before the
        # update so ``retranslate_rate`` is this run's, not the zero default
        # that ``run()`` only fills in after ``_run_inner`` returns.
        current_stats = RetryStats.snapshot()
        result.chunk_retries = current_stats["chunk_retries"]
        result.retries = (
            current_stats["rate_limit_retries"]
            + current_stats["transient_retries"]
            + current_stats["chunk_retries"]
        )
        try:
            # SKILL_APPLY=off is the control arm: no skill went into the prompt,
            # so nothing here can be attributed to one. The run still reports
            # its own mean, which is what a with-skill run is compared against.
            if pending and app.skill_apply:
                with ui.step("skillopt_update"):
                    for chunk in pending:
                        score = scores.get(chunk.id)
                        if score is None or not translated.get(chunk.id):
                            continue
                        skill_opt.observe(
                            src,
                            tgt,
                            prompt=chunk.text[:200],
                            output=translated.get(chunk.id, "")[:200],
                            critic_score=score,
                        )
                    promoted = skill_opt.maybe_promote(
                        src,
                        tgt,
                        run_id=getattr(self.sink, "run_id", ""),
                        doc=source_path.name,
                        verify_fail_rate=(result.verify_skipped / len(chunks)) if chunks else 0.0,
                        retranslate_rate=(result.chunk_retries / len(chunks)) if chunks else 0.0,
                        applied=skill,
                    )
                    result.skill_promoted = promoted is not None
            elif pending and scores:
                # The control arm still leaves a row: otherwise its runs are
                # invisible in history and a with-skill mean has nothing to be
                # compared against.
                skill_store.record_run(
                    run_id=getattr(self.sink, "run_id", ""),
                    doc=source_path.name,
                    pair=f"{src}-{tgt}",
                    used_skill_id="",
                    used_version=0,
                    scores=list(scores.values()),
                    verify_fail_rate=(result.verify_skipped / len(chunks)) if chunks else 0.0,
                    retranslate_rate=(result.chunk_retries / len(chunks)) if chunks else 0.0,
                    promoted=False,
                    reason="SKILL_APPLY=off",
                )
        except Exception as exc:
            logger.warning("SkillOpt update skipped: {}", exc)

        # ---- report ----
        paths = [str(st.get("path", "standard")) for st in chunk_states]
        result.output_path = (
            str(out_path) if want_md else (str(pdf_path) if pdf_path.is_file() else "")
        )
        result.skill_scores = dict(scores)
        result.avg_critic = (sum(scores.values()) / len(scores)) if scores else 0.0
        result.traces_stored = sum(1 for st in chunk_states if st.get("stored"))
        result.fast_path_chunks = paths.count("fast")
        result.standard_path_chunks = paths.count("standard")
        result.deep_path_chunks = paths.count("deep")
        ui.set_stats(
            elapsed_s=result.elapsed_s,
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

    One bullet per term, whatever produced it — the pair's core terms, the
    Planner or the heuristics. Counting the Planner's pairs instead reported zero on a
    fallback plan that still sent dozens of heuristic terms to every request.
    """
    return sum(1 for line in glossary_md.splitlines() if line.lstrip().startswith("- "))


def _load_cached_plan(
    meta: dict[str, Any],
    chunks: list[Any],
    src: str,
    tgt: str,
    glossary_fingerprint: str,
) -> DocumentPlan | None:
    """The saved plan for this exact job, or ``None`` when it cannot be used.

    ``doc_key`` already partitions checkpoints by file, chunk hashes, model,
    endpoint, glossary and skill; the extra checks catch a truncated or
    hand-edited ``meta.json`` and an old cache written before this schema.
    """
    cached = meta.get("plan")
    if not isinstance(cached, dict):
        return None
    try:
        schema = int(cached.get("schema", 0) or 0)
    except (TypeError, ValueError):
        return None
    if schema != PLAN_CACHE_SCHEMA:
        return None
    if cached.get("plan_hash") != plan_hash(chunks):
        return None
    if str(cached.get("src", "")) != src or str(cached.get("tgt", "")) != tgt:
        return None
    if str(cached.get("glossary", "")) != glossary_fingerprint:
        return None
    return plan_from_dict(cached.get("data"))


def _merge_cumulative(
    previous: dict[str, Any],
    result: PipelineResult,
    usage: dict[str, int],
    stats: dict[str, int],
    duration_ms: int,
) -> dict[str, Any]:
    """Add one attempt's counters to the translation's running tally.

    Token and retry counters are additive; chunk state is a snapshot of the
    whole document and must not be summed; scores are keyed by chunk so a chunk
    translated again replaces its old score instead of inflating the mean.
    """
    prev = previous if isinstance(previous, dict) else {}
    previous_tokens = prev.get("tokens") if isinstance(prev.get("tokens"), dict) else {}
    scores: dict[str, float] = {}
    raw_scores = prev.get("skill_scores")
    if isinstance(raw_scores, dict):
        for chunk_id, value in raw_scores.items():
            try:
                scores[str(chunk_id)] = float(value)
            except (TypeError, ValueError):
                continue
    scores.update(result.skill_scores)
    mean = (sum(scores.values()) / len(scores)) if scores else 0.0

    def add(name: str, current: int) -> int:
        try:
            return int(prev.get(name, 0) or 0) + int(current or 0)
        except (TypeError, ValueError):
            return int(current or 0)

    return {
        "attempts": add("attempts", 1),
        "tokens": {
            "sent": int(previous_tokens.get("sent", 0) or 0) + int(usage.get("sent", 0) or 0),
            "received": int(previous_tokens.get("received", 0) or 0)
            + int(usage.get("received", 0) or 0),
            "cached": int(previous_tokens.get("cached", 0) or 0)
            + int(usage.get("cached", 0) or 0),
        },
        "calls": add("calls", usage.get("calls", 0)),
        "cache_hits": add("cache_hits", usage.get("cache_hits", 0)),
        "cache_misses": add("cache_misses", usage.get("cache_misses", 0)),
        "retries": add("retries", result.retries),
        "chunk_retries": add("chunk_retries", result.chunk_retries),
        "fallbacks": add("fallbacks", stats.get("fallbacks", 0)),
        "duration_ms": add("duration_ms", duration_ms),
        "planner_calls": add("planner_calls", result.planner_calls),
        "chunks_done": max(
            0, result.chunks - result.chunks_failed - result.chunks_permanently_failed
        ),
        "chunks_rejected": result.chunks_rejected,
        "verify_skipped": result.verify_skipped,
        "skill": {"mean": round(mean, 3), "samples": len(scores)},
        "skill_scores": scores,
    }


def _collect_scores(
    chunks: list[Any], by_id: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """Map each measured chunk id to its trace score (0-10 scale).

    A missing score means "not measured", not zero: including it turned a
    provider failure into a bad translation and quietly biased every mean.
    """
    scores: dict[str, float] = {}
    for chunk in chunks:
        state = by_id.get(chunk.id)
        trace = (state or {}).get("trace") or {}
        if "score" not in trace:
            continue
        try:
            value = float(trace["score"])
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue
        scores[chunk.id] = value * 10.0
    return scores
