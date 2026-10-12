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

import hashlib
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
    get_agent_config,
    get_app_config,
    get_embed_config,
)
from polyglot_lc.ingest import load_document
from polyglot_lc.llm.client import TokenUsage
from polyglot_lc.llm.resilience import LLMError, RetryStats
from polyglot_lc.orchestrator.graph import (
    GraphContext,
    make_chunk_graph,
    make_document_plan,
    plan_document,
)
from polyglot_lc.pipeline.assembler import (
    assemble,
    assemble_partial,
    block_translations,
    output_paths,
    write_atomic,
)
from polyglot_lc.pipeline.checkpoint import CheckpointStore, ChunkCheckpoint
from polyglot_lc.pipeline.consistency import ConsistencyChecker
from polyglot_lc.pipeline.pdf_output import rebuild_pdf
from polyglot_lc.reasoning_bank.embedder import Embedder
from polyglot_lc.reasoning_bank.store import ReasoningBank
from polyglot_lc.segment.chunker import chunk_document
from polyglot_lc.segment.glossary import (
    MAX_USER_GLOSSARY_TERMS,
    extract_glossary,
    glossary_to_md,
    load_glossary_csv,
    render_user_glossary,
)
from polyglot_lc.segment.summary_tree import summarize_with_planner
from polyglot_lc.skillopt.optimizer import SkillOptimizer
from polyglot_lc.skillopt.skill_store import SkillStore
from polyglot_lc.ui.events import EventSink
from polyglot_lc.ui.simple import SimpleUI
from polyglot_lc.ui.trace import plan_hash


@dataclass
class PipelineResult:
    """Final outcome of one translation run."""

    output_path: str = ""
    #: The rebuilt, layout-preserving PDF, when OUTPUT_FORMAT asked for one.
    pdf_path: str = ""
    pdf_blocks: int = 0
    chunks: int = 0
    avg_critic: float = 0.0
    retries: int = 0
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
    consistency_terms: int = 0
    consistency_patches: int = 0
    partial_output: bool = False
    elapsed_s: float = 0.0
    chunks_failed: int = 0
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

    # ------------------------------------------------------------------
    def run(
        self,
        file_path: str | Path,
        src: str = "auto",
        tgt: str = "vi",
        domain: str = "general",
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
                    domain,
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
                logger.error("Config error: {}", exc)
                raise
            except Exception as exc:
                ui.add_error(f"{type(exc).__name__}: {exc}")
                result.errors = list(ui.errors)
                raise
            finally:
                # Retries come from the resilience layer's own counters, which
                # are the only place a retry actually happens.
                stats = RetryStats.snapshot()
                result.retries = stats["rate_limit_retries"] + stats["transient_retries"]
                result.log_path = str(self.sink.path or "")
                # Written from the `finally` so a run that raised still leaves a
                # record of how it ended — that is the run most worth reading.
                self._record_run_end(result, stats, round((time.monotonic() - started) * 1000))
                # A chunk abandoned after provider errors (an exhausted rate
                # limit) does not raise: the run continues and its place in the
                # output is marked. The report still has to carry it, or a lost
                # section looks like a clean run.
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
        self.sink.record(
            type="run_end",
            status="failed" if result.errors else "done",
            exit_code=1 if result.errors else 0,
            duration_ms=duration_ms,
            tokens={
                "sent": usage["sent"],
                "received": usage["received"],
                "cached": usage.get("cached", 0),
            },
            calls=usage["calls"],
            cache_hits=usage.get("cache_hits", 0),
            cache_misses=usage.get("cache_misses", 0),
            retries=result.retries,
            fallbacks=stats.get("fallbacks", 0),
            chunks_done=max(0, result.chunks - result.chunks_failed),
            chunks_failed=result.chunks_failed,
            resumed_chunks=result.resumed_chunks,
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
        ui: SimpleUI,
        subsystems: dict[str, Any],
        chunks: list[Any],
        heuristic_terms: list[Any],
        user_glossary: dict[str, str],
        src: str,
        tgt: str,
        domain: str,
        doc_text: str,
        doc_title: str,
    ) -> DocumentPlan:
        """Produce the document plan — one Planner request on the unified path."""
        if self.app.unified_planner:
            return make_document_plan(
                ctx,
                chunks=chunks,
                src_lang=src,
                tgt_lang=tgt,
                domain=domain,
                doc_text=doc_text,
                doc_title=doc_title,
                heuristic_terms=heuristic_terms,
                user_glossary=user_glossary,
            )

        # Legacy path: summary tree (one request per section + document) + plan.
        planner = subsystems["planner"]
        with ui.step("summary_tree"):
            tree = summarize_with_planner(chunks, planner, doc_title=doc_title)
            doc_summary = tree.summary or tree.render()
        glossary_md = glossary_to_md(heuristic_terms)
        rendered_user = render_user_glossary(user_glossary)
        if rendered_user:
            glossary_md += "\n" + rendered_user
        legacy = plan_document(
            ctx,
            doc_summary=doc_summary,
            src_lang=src,
            tgt_lang=tgt,
            domain=domain,
            glossary_md=glossary_md,
        )
        return DocumentPlan(
            summary=doc_summary,
            strategy=legacy["strategy"],
            few_shot=legacy["few_shot"],
            pitfalls=legacy["pitfalls"],
            glossary_pairs=[],
            glossary_md=legacy["glossary_md"],
            complexity=legacy["complexity"],
            section_notes=legacy["section_notes"],
            used_llm=True,
        )

    # ------------------------------------------------------------------
    def _run_chunks(
        self,
        chunk_graph: Any,
        chunks: list[Any],
        base_state: dict[str, Any],
        ui: SimpleUI,
        *,
        per_chunk_state: dict[str, dict[str, Any]] | None = None,
        store: CheckpointStore | None = None,
        doc_key: str = "",
        on_chunk_done: Callable[[Any, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Run each chunk's graph, bounded-parallel across independent chunks.

        Each chunk keeps its internal Translate->[Critic->Refiner]->Verify order;
        only *different* chunks run concurrently (``MAX_CONCURRENCY``). Results
        are returned in the original chunk order regardless of completion order.
        A provider failure on one chunk is recorded and skipped instead of
        discarding a whole book run: the chunk is left uncheckpointed (so
        ``--resume`` retries it) and the output marks it as untranslated.
        """
        per_chunk_state = per_chunk_state or {}

        def translate_chunk(chunk: Any) -> dict[str, Any]:
            state = {**base_state, **per_chunk_state.get(chunk.id, {}), "chunk": chunk}
            out = chunk_graph.invoke(state, config={"recursion_limit": 100})
            # ---- Retry empty translations (up to 2 extra attempts) ----
            max_empty_retries = 2
            for retry in range(max_empty_retries):
                translated = out.get("translated", "") or out.get("draft", "")
                if translated and translated.strip():
                    break
                logger.warning(
                    "Chunk {} produced empty translation (attempt {}/{}); retrying",
                    chunk.id,
                    retry + 1,
                    max_empty_retries,
                )
                out = chunk_graph.invoke(state, config={"recursion_limit": 100})
            return out

        def run_one(index: int, chunk: Any) -> dict[str, Any]:
            try:
                out = translate_chunk(chunk)
            except LLMError as exc:
                ui.add_error(f"{chunk.id}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "Chunk {} abandoned after provider errors ({}); continuing with the rest",
                    chunk.id,
                    exc,
                )
                return {"translated": "", "chunk": chunk, "error": str(exc)}
            # ---- Only checkpoint non-empty translations ----
            translated = out.get("translated", "")
            if store is not None and store.usable and translated and translated.strip():
                store.put(doc_key, self._to_checkpoint(chunk, out))
            elif store is not None and store.usable and not (translated and translated.strip()):
                logger.warning(
                    "Chunk {} has empty translation after retries; not checkpointing",
                    chunk.id,
                )
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
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()  # raise the first failure
                ui.advance()
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
        domain: str,
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
        ui.set_stats(src=src, tgt=tgt, domain=domain)

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
            domain=domain,
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
        with ui.step("glossary"):
            heuristic_terms = extract_glossary(doc.plain_text())
            user_glossary: dict[str, str] = {}
            if glossary:
                user_glossary = load_glossary_csv(glossary)
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
        doc_key = CheckpointStore.doc_key(
            file_path,
            src=src,
            tgt=tgt,
            domain=domain,
            model=translator_cfg.model,
            chunk_hashes=[c.hash() for c in chunks],
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
                ui=ui,
                subsystems=subsystems,
                chunks=chunks,
                heuristic_terms=heuristic_terms,
                user_glossary=user_glossary,
                src=src,
                tgt=tgt,
                domain=domain,
                doc_text=doc.plain_text(),
                doc_title=doc.metadata.get("filename", "Document"),
            )
        result.planner_calls = planner.calls
        result.document_complexity = round(plan.complexity, 3)
        result.glossary_terms = len(plan.glossary_pairs)

        glossary_md = plan.glossary_md or "(none)"
        base_state: dict[str, Any] = {
            "src_lang": src,
            "tgt_lang": tgt,
            "domain": domain,
            "strategy": plan.strategy,
            "few_shot": plan.few_shot,
            "pitfalls": plan.pitfalls,
            "glossary_md": glossary_md,
        }
        if store.usable:
            store.save_meta(
                doc_key,
                {
                    "file": str(file_path),
                    "src": src,
                    "tgt": tgt,
                    "domain": domain,
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
        if pending and app.consistency_check and pairs:
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
            result.consistency_terms = report.inconsistencies
            result.consistency_patches = report.patched_chunks
            # Keep the checkpoints in sync with what was actually written.
            for chunk in chunks:
                state = by_id.get(chunk.id)
                if state is None or not state.get("translated"):
                    continue
                if translated.get(chunk.id, "") == state.get("translated"):
                    continue
                state["translated"] = translated[chunk.id]
                if store.usable:
                    checkpoint = self._to_checkpoint(chunk, state)
                    checkpoint.patched = True
                    store.put(doc_key, checkpoint)

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
        try:
            if pending:
                with ui.step("skillopt_update"):
                    scores_raw = _collect_scores(final)
                    for chunk, score in zip(chunks, scores_raw, strict=False):
                        skill_opt.observe(
                            src,
                            tgt,
                            domain,
                            prompt=chunk.text[:200],
                            output=translated.get(chunk.id, "")[:200],
                            critic_score=score,
                        )
                    promoted = skill_opt.maybe_promote(src, tgt, domain)
                    result.skill_promoted = promoted is not None
        except Exception as exc:  # noqa: BLE001 - optimization is best-effort
            logger.warning("SkillOpt update skipped: {}", exc)

        # ---- report ----
        scores = _collect_scores(final)
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
        ui.set_stats(
            elapsed_s=round(elapsed, 1),
            avg_critic=round(result.avg_critic, 2),
            traces=result.traces_stored,
            fast_path=result.fast_path_chunks,
            skill_promoted=result.skill_promoted,
        )
        return result


def _collect_scores(final: dict[str, Any]) -> list[float]:
    """Extract per-chunk quality scores recorded in traces (0-10 scale)."""
    scores: list[float] = []
    for trace in final.get("traces", []):
        s = trace.get("score", 0.0)
        scores.append(float(s) * 10.0)
    return scores
