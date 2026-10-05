"""End-to-end pipeline: ingest -> segment -> plan -> graph -> assemble -> save.

Owns subsystem construction (agents, reasoning bank, skillopt, UI) and the
final report. The heavy lifting per chunk happens inside the LangGraph.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from polyglot_lc.config import (
    ConfigError,
    get_agent_config,
    get_app_config,
    get_embed_config,
)
from polyglot_lc.agents import (
    CriticAgent,
    PlannerAgent,
    RefinerAgent,
    TranslatorAgent,
    VerifierAgent,
)
from polyglot_lc.ingest import load_document
from polyglot_lc.ingest.md import restore_code
from polyglot_lc.orchestrator.graph import GraphContext, make_chunk_graph, plan_document
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


@dataclass
class PipelineResult:
    """Final outcome of one translation run."""

    output_path: str = ""
    chunks: int = 0
    avg_critic: float = 0.0
    retries: int = 0
    traces_stored: int = 0
    log_path: str = ""
    errors: list[str] = field(default_factory=list)
    skill_promoted: bool = False


def _detect_lang(text: str) -> str:
    """Detect source language with langdetect (falls back to 'en')."""
    try:
        from langdetect import detect

        return detect(text[:2000]) or "en"
    except Exception:  # noqa: BLE001 - detection is best-effort
        return "en"


class TranslationPipeline:
    """ Orchestrates one full document translation."""

    def __init__(self, quiet: bool = False) -> None:
        self.app = get_app_config()
        self.quiet = quiet
        # Validate agent configs up-front so missing keys hit the Errors panel.
        self.agent_configs = {name: get_agent_config(name) for name in
                              ("PLANNER", "TRANSLATOR", "CRITIC", "REFINER", "VERIFIER")}
        self.embed_config = get_embed_config()
        self.sink = EventSink(self.app.logs_dir, enabled=self.app.log_jsonl)

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
    ) -> PipelineResult:
        """Translate *file_path* from *src* to *tgt*; returns a report."""
        result = PipelineResult()

        with SimpleUI(self.sink, quiet=self.quiet) as ui:
            try:
                return self._run_inner(
                    file_path, src, tgt, domain, glossary, out, max_tokens,
                    ui, result,
                )
            except ConfigError as exc:
                ui.add_error(str(exc))
                result.errors = list(ui.errors)
                logger.error("Config error: {}", exc)
                raise
            except Exception as exc:  # noqa: BLE001
                ui.add_error(f"{type(exc).__name__}: {exc}")
                result.errors = list(ui.errors)
                raise
            finally:
                result.retries = ui.retries
                result.log_path = str(self.sink.path or "")

    # ------------------------------------------------------------------
    def _run_chunks(
        self,
        chunk_graph: Any,
        chunks: list[Any],
        base_state: dict[str, Any],
        ui: SimpleUI,
    ) -> list[dict[str, Any]]:
        """Run each chunk's graph, bounded-parallel across independent chunks.

        Each chunk keeps its internal Translate→Critic→Refiner→Verify order;
        only *different* chunks run concurrently (``MAX_CONCURRENCY``). Results
        are returned in the original chunk order regardless of completion order.
        A chunk failure propagates (matching the previous sequential behaviour).
        """

        def run_one(chunk: Any) -> dict[str, Any]:
            state = {**base_state, "chunk": chunk}
            return chunk_graph.invoke(state, config={"recursion_limit": 100})

        results: list[Any] = [None] * len(chunks)
        workers = max(1, self.app.max_concurrency)
        if workers == 1 or len(chunks) <= 1:
            for i, chunk in enumerate(chunks):
                results[i] = run_one(chunk)
                ui.advance()
            return results

        with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
            futures = {pool.submit(run_one, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()  # raise the first failure
                ui.advance()
        return results

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
    ) -> PipelineResult:
        app = self.app

        # ---- ingest ----
        with ui.step("ingest") as h:
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
            chunks = chunk_document(doc, max_tokens=max_tokens,
                                    model=translator_cfg.model)
            ui.set_stats(chunks=len(chunks))

        # ---- glossary ----
        with ui.step("glossary"):
            terms = extract_glossary(doc.plain_text())
            glossary_md = glossary_to_md(terms)
            user_glossary: dict[str, str] = {}
            if glossary:
                user_glossary = load_glossary_csv(glossary)
                if len(user_glossary) > MAX_USER_GLOSSARY_TERMS:
                    logger.warning(
                        "Glossary has {} entries; keeping the first {} to bound context",
                        len(user_glossary), MAX_USER_GLOSSARY_TERMS,
                    )
                rendered = render_user_glossary(user_glossary)
                if rendered:
                    glossary_md += "\n" + rendered

        # ---- subsystems ----
        planner = PlannerAgent(self.agent_configs["PLANNER"])
        translator = TranslatorAgent(self.agent_configs["TRANSLATOR"])
        critic = CriticAgent(self.agent_configs["CRITIC"])
        refiner = RefinerAgent(self.agent_configs["REFINER"])
        verifier = VerifierAgent(self.agent_configs["VERIFIER"])

        reasoning_bank: ReasoningBank | None = None
        try:
            embedder = Embedder(self.embed_config)
            reasoning_bank = ReasoningBank(app.reasoning_bank_path, embedder)
        except Exception as exc:  # noqa: BLE001 - RB is best-effort
            ui.add_error(f"reasoning bank unavailable: {exc}")
            logger.warning("Reasoning bank unavailable: {}", exc)

        skill_opt = SkillOptimizer(SkillStore(app.skills_dir), planner)

        # ---- summary tree (Planner LLM) ----
        with ui.step("summary_tree"):
            tree = summarize_with_planner(
                chunks, planner, doc_title=doc.metadata.get("filename", "Document")
            )
            doc_summary = tree.summary or tree.render()

        # ---- graph ----
        ctx = GraphContext(
            planner=planner,
            translator=translator,
            critic=critic,
            refiner=refiner,
            verifier=verifier,
            ui=ui,
            sink=self.sink,
            reasoning_bank=reasoning_bank,
        )
        chunk_graph = make_chunk_graph(ctx)
        ui.set_total_chunks(len(chunks))

        # ---- document-level plan (once, never concurrent) ----
        plan = plan_document(
            ctx,
            doc_summary=doc_summary,
            src_lang=src,
            tgt_lang=tgt,
            domain=domain,
            glossary_md=glossary_md,
        )
        base_state: dict[str, Any] = {
            "src_lang": src,
            "tgt_lang": tgt,
            "domain": domain,
            "strategy": plan["strategy"],
            "few_shot": plan["few_shot"],
            "pitfalls": plan["pitfalls"],
            "glossary_md": glossary_md,
        }

        t0 = time.monotonic()
        chunk_states = self._run_chunks(chunk_graph, chunks, base_state, ui)
        elapsed = time.monotonic() - t0

        translated = {
            chunk.id: st.get("translated", "")
            for chunk, st in zip(chunks, chunk_states, strict=True)
        }
        traces = [st.get("trace", {}) for st in chunk_states]
        final: dict[str, Any] = {"translated": translated, "traces": traces}

        # ---- assemble output ----
        with ui.step("assemble_output"):
            code_map = doc.metadata.get("code_map", {})
            parts: list[str] = []
            ordered = sorted(final["translated"].items(), key=lambda kv: kv[0])
            for _, translated in ordered:
                parts.append(restore_code(translated, code_map))
            output_text = "\n\n".join(parts)

            out_path = Path(out) if out else Path(
                f"{Path(file_path).stem}.{tgt}{Path(file_path).suffix or '.txt'}"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_text, encoding="utf-8")

        # ---- skillopt update (best-effort; never fails the translation) ----
        try:
            with ui.step("skillopt_update"):
                scores_raw = _collect_scores(final)
                for chunk, score in zip(chunks, scores_raw, strict=False):
                    skill_opt.observe(src, tgt, domain, prompt=chunk.text[:200],
                                      output=final["translated"].get(chunk.id, "")[:200],
                                      critic_score=score)
                promoted = skill_opt.maybe_promote(src, tgt, domain)
                result.skill_promoted = promoted is not None
        except Exception as exc:  # noqa: BLE001 - optimization is best-effort
            logger.warning("SkillOpt update skipped: {}", exc)

        # ---- report ----
        scores = _collect_scores(final)
        result.output_path = str(out_path)
        result.chunks = len(chunks)
        result.avg_critic = (sum(scores) / len(scores)) if scores else 0.0
        result.traces_stored = len(final.get("traces", []))
        ui.set_stats(
            elapsed_s=round(elapsed, 1),
            avg_critic=round(result.avg_critic, 2),
            traces=result.traces_stored,
            skill_promoted=result.skill_promoted,
        )
        return result


def _collect_scores(final: dict[str, Any]) -> list[float]:
    """Extract per-chunk critic scores recorded in traces (0-10 scale)."""
    scores: list[float] = []
    for trace in final.get("traces", []):
        s = trace.get("score", 0.0)
        scores.append(float(s) * 10.0)
    return scores
