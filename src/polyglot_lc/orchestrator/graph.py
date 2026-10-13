"""Translation graph: document-level planning + per-chunk adaptive graph.

The document-level step (:func:`make_document_plan`) runs once and is never
concurrent; it is a single Planner request (see
:mod:`polyglot_lc.analysis.unified_planner`). Each independent chunk then flows
through its own compiled graph::

    translator -> [fast: complexity low + draft clean] -> verifier
               -> [full: critic -> [refiner -> critic]* ] -> verifier
    verifier -> [retranslate -> verifier] (max 1) -> collect -> save_trace -> END

Chunks keep this internal order; ``MAX_CONCURRENCY`` only parallelizes *different*
chunks (see :meth:`polyglot_lc.pipeline.coordinator.TranslationPipeline._run_chunks`).
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger

from polyglot_lc import emphasis
from polyglot_lc.agents import (
    CriticAgent,
    PlannerAgent,
    RefinerAgent,
    TranslatorAgent,
    VerifierAgent,
)
from polyglot_lc.agents.verifier import VerifyResult, hard_checks
from polyglot_lc.analysis.complexity import draft_signals
from polyglot_lc.analysis.unified_planner import DocumentPlan, analyze_document
from polyglot_lc.config import get_app_config
from polyglot_lc.orchestrator.router import (
    ROUTE_CRITIC,
    ROUTE_VERIFY,
    after_critic,
    after_translate,
    after_verifier,
)
from polyglot_lc.orchestrator.state import ChunkState

#: Route label recorded in each chunk's state.
PATH_FAST = "fast"
PATH_STANDARD = "standard"
PATH_DEEP = "deep"


def settle_emphasis(chunk_text: str, translation: str) -> tuple[str, bool]:
    """Drop the chunk's emphasis tokens unless they match the source exactly.

    The tokens mark spans, not content, so a mismatch is not worth a
    re-translation — but a lopsided pair left in place would put a ``⟦`` on the
    page, so it is removed at the last point that sees the whole chunk.

    Module level, and not a closure in ``collect_node``, so the rule can be
    tested without standing up a graph.
    """
    if emphasis.agrees_with(chunk_text, translation):
        return translation, False
    return emphasis.strip(translation), True


class GraphContext:
    """Shared services the graph nodes need (agents, RB, UI, events)."""

    def __init__(
        self,
        planner: PlannerAgent,
        translator: TranslatorAgent,
        critic: CriticAgent,
        refiner: RefinerAgent,
        verifier: VerifierAgent,
        ui: Any = None,
        sink: Any = None,
        reasoning_bank: Any = None,
    ) -> None:
        self.planner = planner
        self.translator = translator
        self.critic = critic
        self.refiner = refiner
        self.verifier = verifier
        self.ui = ui
        self.sink = sink
        self.reasoning_bank = reasoning_bank
        self.app = get_app_config()
        # Reasoning Bank access is serialized: chunk graphs run concurrently.
        self.rb_lock = threading.Lock()
        # Decision counts for the end-of-run report. Concurrent chunks record
        # from several threads, so the counters get their own lock.
        self.decision_lock = threading.Lock()
        self.decision_counts: dict[str, int] = {}
        self.fast_path_rejections = 0
        self.fast_path_reasons: Counter[str] = Counter()

    def step(self, name: str, chunk_id: str = "") -> Any:
        """UI step context manager (no-op-ish when ui is None)."""
        if self.ui is not None:
            return self.ui.step(name, chunk_id)
        from contextlib import nullcontext

        return nullcontext()

    def record_decision(
        self, *, kind: str, chosen: bool, reasons: list[str], **fields: Any
    ) -> None:
        """Write one routing decision, and keep the counts the report needs."""
        payload = {"kind": kind, "chosen": bool(chosen), "reasons": list(reasons), **fields}
        with self.decision_lock:
            self.decision_counts[kind] = self.decision_counts.get(kind, 0) + 1
            if kind == "fast_path" and not chosen:
                self.fast_path_rejections += 1
                if reasons:
                    # Only the decisive reason (the first one) is tallied: the
                    # rest are context, and counting them would let a shared
                    # "this gate passed" line outrank the real cause.
                    self.fast_path_reasons[str(reasons[0])] += 1
        if self.sink is not None:
            self.sink.record(type="decision", **payload)

    def record_error(
        self, *, cause: str, step: str = "", chunk_id: str = "", message: str = ""
    ) -> None:
        """Write one problem with its cause, kept inside the closed set.

        An unknown cause is replaced by ``unknown`` rather than written as-is:
        the file's guarantee is that every cause is one of twelve values, and a
        reader that trusts it must not be able to find a twelfth.
        """
        from polyglot_lc.ui.trace import CAUSES

        if cause not in CAUSES:
            logger.warning("trace: unknown error cause {!r}; recording it as 'unknown'", cause)
            cause = "unknown"
        if self.sink is None:
            return
        self.sink.record(type="error", step=step, chunk_id=chunk_id, cause=cause, message=message)


def retrieve_traces(ctx: GraphContext, query: str) -> list[Any]:
    """Best-effort Reasoning Bank lookup (never blocks a run)."""
    if ctx.reasoning_bank is None:
        return []
    try:
        with ctx.rb_lock:
            return ctx.reasoning_bank.retrieve(query, top_k=ctx.app.rb_top_k)
    except Exception as exc:  # noqa: BLE001 - RB is best-effort
        logger.warning("Reasoning Bank retrieval failed ({}); continuing", exc)
        return []


def make_document_plan(
    ctx: GraphContext,
    *,
    chunks: list[Any],
    src_lang: str,
    tgt_lang: str,
    domain: str,
    doc_text: str = "",
    doc_summary: str = "",
    doc_title: str = "Document",
    heuristic_terms: list[Any] | None = None,
    user_glossary: dict[str, str] | None = None,
    extra_glossary_md: str = "",
) -> DocumentPlan:
    """Document-level planning (runs once): RB lookup + ONE Planner request."""
    with ctx.step("plan"):
        traces = retrieve_traces(ctx, f"{domain} {(doc_summary or doc_text)[:200]}")
        return analyze_document(
            chunks,
            ctx.planner,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            domain=domain,
            doc_text=doc_text,
            doc_summary=doc_summary,
            doc_title=doc_title,
            heuristic_terms=heuristic_terms,
            user_glossary=user_glossary,
            rb_traces=traces,
            extra_glossary_md=extra_glossary_md,
        )


def plan_document(
    ctx: GraphContext,
    *,
    doc_summary: str,
    src_lang: str,
    tgt_lang: str,
    domain: str,
    glossary_md: str = "",
) -> dict[str, Any]:
    """Document-level planning without chunks (single Planner request).

    Backwards-compatible entry point kept for callers that only have a document
    summary; the pipeline uses :func:`make_document_plan` so the Planner can see
    the section outline and terminology candidates too. The Reasoning Bank is
    best-effort: a retrieval failure degrades to ``(no traces)``.
    """
    plan = make_document_plan(
        ctx,
        chunks=[],
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        domain=domain,
        doc_summary=doc_summary,
        extra_glossary_md=glossary_md,
    )
    return {
        "strategy": plan.strategy,
        "few_shot": plan.few_shot,
        "pitfalls": plan.pitfalls,
        "summary": plan.summary,
        "glossary_md": plan.glossary_md,
        "complexity": plan.complexity,
        "section_notes": plan.section_notes,
        "glossary_pairs": plan.glossary_pairs,
    }


def should_take_fast_path(state: ChunkState, app_cfg: Any) -> bool:
    """Fast path gate: simple chunk, clean draft, high deterministic score.

    Disabled when ``ADAPTIVE_QUALITY=0`` or when the caller did not supply a
    complexity score (an unknown chunk is treated as a hard one).
    """
    return fast_path_verdict(state, app_cfg)[0]


def fast_path_verdict(state: ChunkState, app_cfg: Any) -> tuple[bool, list[str]]:
    """The fast-path gate, plus the reasons it went the way it did.

    The reasons name the comparison that decided it, so a trace can answer "why
    did this chunk cost four requests?" without re-deriving the numbers. They
    are built here rather than at the call site so the gate and its explanation
    can never disagree.

    **The first reason is the decisive one**; any that follow are context. The
    rejection tally in the run report depends on that order.
    """
    if not app_cfg.adaptive_quality:
        return False, ["ADAPTIVE_QUALITY off"]
    complexity = state.get("complexity")
    if complexity is None:
        return False, ["complexity unknown for this chunk"]
    limit = float(app_cfg.fast_path_max_complexity)
    passed = f"complexity {float(complexity):.2f} <= {limit:.2f}"
    if complexity > limit:
        return False, [f"complexity {float(complexity):.2f} > {limit:.2f}"]
    issues = list(state.get("draft_issues") or [])
    if issues:
        return False, [f"draft issues: {', '.join(map(str, issues))}", passed]
    score = float(state.get("draft_score", 0.0))
    min_score = float(app_cfg.fast_path_min_score)
    if score < min_score:
        return False, [f"draft score {score:.2f} < {min_score:.2f}", passed]
    return True, [passed, "draft clean"]


def make_chunk_graph(ctx: GraphContext):
    """Build and compile the per-chunk translation StateGraph."""
    app_cfg = ctx.app

    def translator_node(state: ChunkState) -> dict[str, Any]:
        chunk = state["chunk"]
        section_note = state.get("section_note", "")
        with ctx.step("translate", chunk.id):
            prompt = ctx.translator.build_prompt(
                chunk_text=chunk.text,
                src_lang=state["src_lang"],
                tgt_lang=state["tgt_lang"],
                strategy=state["strategy"],
                few_shot=state.get("few_shot"),
                pitfalls=state.get("pitfalls"),
                glossary_md=state.get("glossary_md", ""),
                extra_critique="",
                section_note=section_note,
            )
            draft = ctx.translator.translate(prompt, chunk_id=chunk.id)
        # Free, local self-assessment: decides the route without a second call.
        signals = draft_signals(chunk.text, draft)
        decision: dict[str, Any] = {
            "draft": draft,
            "critique": None,
            "refine_round": 0,
            "retranslate_count": 0,
            "draft_score": signals.score,
            "draft_issues": signals.issues,
        }
        fast, reasons = fast_path_verdict({**state, **decision}, app_cfg)
        decision["fast_path"] = fast
        decision["fast_path_reasons"] = reasons
        decision["skip_llm_verify"] = bool(fast and app_cfg.fast_path_skip_llm_verify)
        # The gate already computed why; until now that answer was thrown away
        # one line after it was used.
        ctx.record_decision(
            chunk_id=chunk.id,
            kind="fast_path",
            chosen=bool(fast),
            reasons=reasons,
            numbers={
                "complexity": state.get("complexity"),
                "max_complexity": app_cfg.fast_path_max_complexity,
                "draft_score": signals.score,
                "min_score": app_cfg.fast_path_min_score,
            },
        )
        return decision

    def critic_node(state: ChunkState) -> dict[str, Any]:
        chunk = state["chunk"]
        with ctx.step("critic", chunk.id):
            critique = ctx.critic.review(
                source=chunk.text,
                draft=state["draft"],
                src_lang=state["src_lang"],
                tgt_lang=state["tgt_lang"],
                glossary_md=state.get("glossary_md", ""),
                strategy=state.get("strategy", ""),
                pitfalls=state.get("pitfalls"),
                section_note=state.get("section_note", ""),
            )
            return {"critique": critique}

    def refiner_node(state: ChunkState) -> dict[str, Any]:
        chunk = state["chunk"]
        with ctx.step("refine", chunk.id):
            critique = state["critique"]
            refined = ctx.refiner.refine(
                source=chunk.text,
                draft=state["draft"],
                critique_text=critique.feedback_text,
                src_lang=state["src_lang"],
                tgt_lang=state["tgt_lang"],
                strategy=state.get("strategy", ""),
                pitfalls=state.get("pitfalls"),
                glossary_md=state.get("glossary_md", ""),
                section_note=state.get("section_note", ""),
            )
            return {"draft": refined, "refine_round": state.get("refine_round", 0) + 1}

    def verifier_node(state: ChunkState) -> dict[str, Any]:
        chunk = state["chunk"]
        with ctx.step("verify", chunk.id):
            # Repair dropped ⟦CODE_n⟧ markers *before* checking: the text being
            # verified should be the text that ships, and a marker the model lost
            # is deterministically recoverable, so it must not cost a retranslate.
            # Imported here: the pipeline package imports this module.
            from polyglot_lc.pipeline.placeholders import repair_code_placeholders

            draft, repaired = repair_code_placeholders(chunk.text, state["draft"])
            if repaired:
                logger.warning(
                    "chunk={} translation dropped {} code marker(s) ({}); "
                    "re-inserted at the source position",
                    chunk.id,
                    len(repaired),
                    ", ".join(repaired),
                )
                ctx.record_error(
                    step="verify",
                    chunk_id=chunk.id,
                    cause="placeholder",
                    message=f"{len(repaired)} code marker(s) re-inserted: {', '.join(repaired)}",
                )
            if state.get("skip_llm_verify"):
                # Fast verify: the deterministic hard checks already passed,
                # so the (paid) LLM verdict would add nothing.
                problems = hard_checks(chunk.text, draft)
                ctx.record_decision(
                    chunk_id=chunk.id,
                    kind="verify_mode",
                    chosen=True,
                    reasons=[
                        "fast path: deterministic checks only"
                        if not problems
                        else f"deterministic checks failed: {'; '.join(problems[:3])}"
                    ],
                    numbers={"hard_check_problems": len(problems)},
                )
                return {
                    "draft": draft,
                    "verify_result": VerifyResult(
                        ok=not problems,
                        reasons=problems or ["deterministic checks passed"],
                    ),
                }
            ctx.record_decision(
                chunk_id=chunk.id,
                kind="verify_mode",
                chosen=False,
                reasons=["LLM verifier plus deterministic checks"],
                numbers={},
            )
            vr = ctx.verifier.verify(
                source=chunk.text,
                translation=draft,
                src_lang=state["src_lang"],
                tgt_lang=state["tgt_lang"],
                glossary_md=state.get("glossary_md", ""),
            )
            return {"draft": draft, "verify_result": vr}

    def retranslate_node(state: ChunkState) -> dict[str, Any]:
        chunk = state["chunk"]
        with ctx.step("retranslate", chunk.id):
            vr = state["verify_result"]
            reasons = "; ".join(getattr(vr, "reasons", [])) or "verification failed"
            prompt = ctx.translator.build_prompt(
                chunk_text=chunk.text,
                src_lang=state["src_lang"],
                tgt_lang=state["tgt_lang"],
                strategy=state["strategy"],
                few_shot=state.get("few_shot"),
                pitfalls=state.get("pitfalls"),
                glossary_md=state.get("glossary_md", ""),
                extra_critique=reasons,
                section_note=state.get("section_note", ""),
            )
            draft = ctx.translator.translate(prompt, chunk_id=chunk.id)
            return {
                "draft": draft,
                "verify_result": None,
                "retranslate_count": state.get("retranslate_count", 0) + 1,
            }

    def collect_node(state: ChunkState) -> dict[str, Any]:
        if state.get("fast_path"):
            path = PATH_FAST
        elif state.get("refine_round", 0) > 0:
            path = PATH_DEEP
        else:
            path = PATH_STANDARD
        # A dropped ⟦CODE_n⟧ marker would silently delete that code block, so
        # re-insert any the model lost before the text is checkpointed.
        # Imported here: the pipeline package imports this module.
        from polyglot_lc.pipeline.placeholders import repair_code_placeholders

        translated, repaired = repair_code_placeholders(state["chunk"].text, state["draft"])
        if repaired:
            logger.warning(
                "chunk={} translation dropped {} code marker(s) ({}); "
                "re-inserted at the source position",
                state["chunk"].id,
                len(repaired),
                ", ".join(repaired),
            )
            ctx.record_error(
                step="collect",
                chunk_id=state["chunk"].id,
                cause="placeholder",
                message=f"{len(repaired)} code marker(s) re-inserted before saving",
            )
        # Emphasis tokens mark spans, not content: a chunk whose pairs do not
        # match the source is printed plain rather than translated again (spec
        # section 2.1), but the mismatch is recorded so a reader can see it.
        translated, dropped_emphasis = settle_emphasis(state["chunk"].text, translated)
        if dropped_emphasis:
            logger.warning(
                "chunk={} emphasis tokens did not match the source; printing this chunk plain",
                state["chunk"].id,
            )
            ctx.record_error(
                step="collect",
                chunk_id=state["chunk"].id,
                cause="emphasis",
                message="emphasis tokens dropped: counts do not match the source",
            )
        return {"translated": translated, "path": path}

    def save_trace_node(state: ChunkState) -> dict[str, Any]:
        from polyglot_lc.reasoning_bank.schema import ReasoningTrace

        chunk = state["chunk"]
        with ctx.step("save_trace", chunk.id):
            critique = state.get("critique")
            if critique is None:
                # Fast path: no Critic ran, so the deterministic draft score is
                # the only honest quality estimate for the trace.
                raw_score = float(state.get("draft_score", 5.0))
                detail = "draft signals: " + ("; ".join(state.get("draft_issues", [])) or "clean")
            else:
                raw_score = float(getattr(critique, "score", 5.0) or 0.0)
                detail = getattr(critique, "feedback_text", "")[:120]
            trace = ReasoningTrace(
                domain=state["domain"],
                src_lang=state["src_lang"],
                tgt_lang=state["tgt_lang"],
                strategy=state["strategy"],
                few_shot=state.get("few_shot", [])[:3],
                pitfalls=state.get("pitfalls", [])[:3] + [f"{chunk.id}: {detail}"],
                score=round(max(0.0, min(1.0, raw_score / 10.0)), 3),
            )
            stored = False
            if ctx.reasoning_bank is not None:
                try:
                    with ctx.rb_lock:
                        stored = ctx.reasoning_bank.upsert(trace)
                except Exception as exc:  # noqa: BLE001 - RB is best-effort
                    logger.warning("Reasoning Bank upsert failed ({}); continuing", exc)
            return {"trace": trace.model_dump(), "stored": stored}

    g = StateGraph(ChunkState)
    g.add_node("translator", translator_node)
    g.add_node("critic", critic_node)
    g.add_node("refiner", refiner_node)
    g.add_node("verifier", verifier_node)
    g.add_node("retranslate", retranslate_node)
    g.add_node("collect", collect_node)
    g.add_node("save_trace", save_trace_node)

    g.set_entry_point("translator")
    g.add_conditional_edges(
        "translator",
        after_translate,
        {ROUTE_CRITIC: "critic", ROUTE_VERIFY: "verifier"},
    )

    def critic_edge(state: ChunkState) -> str:
        """Route after the Critic, recording the gate's numbers and reasons."""
        route = after_critic(
            state, app_cfg.critic_threshold, app_cfg.refine_rounds, app_cfg.refine_min_score
        )
        critique = state.get("critique")
        score = float(getattr(critique, "score", 10.0) or 0.0)
        threshold = float(app_cfg.critic_threshold)
        rounds_used = int(state.get("refine_round", 0) or 0)
        reasons = [
            f"critic score {score:.2f} {'<' if score < threshold else '>='} threshold {threshold:.2f}"
        ]
        if app_cfg.refine_min_score is not None and score >= app_cfg.refine_min_score:
            reasons.append(f"score >= refine_min_score {app_cfg.refine_min_score:.2f}")
        if rounds_used >= app_cfg.refine_rounds:
            reasons.append(f"refine rounds exhausted ({rounds_used}/{app_cfg.refine_rounds})")
        ctx.record_decision(
            chunk_id=state["chunk"].id,
            kind="refine_gate",
            chosen=route == "refine",
            reasons=reasons,
            numbers={
                "critic_score": score,
                "threshold": threshold,
                "refine_min_score": app_cfg.refine_min_score,
                "refine_round": rounds_used,
            },
        )
        return route

    def verifier_edge(state: ChunkState) -> str:
        """Route after the Verifier, recording the verdict and any abandonment."""
        route = after_verifier(state, max_retries=1)
        verdict = state.get("verify_result")
        ok = bool(verdict) if verdict is not None else True
        rejections = [str(r) for r in (getattr(verdict, "reasons", None) or [])]
        ctx.record_decision(
            chunk_id=state["chunk"].id,
            kind="retranslate",
            chosen=route == "retranslate",
            reasons=(
                [f"verifier rejected the draft: {r}" for r in rejections[:3]]
                if not ok
                else ["verifier accepted the draft"]
            ),
            numbers={
                "ok": ok,
                "retranslate_count": int(state.get("retranslate_count", 0) or 0),
            },
        )
        if not ok and route != "retranslate":
            # Out of retranslate budget: this chunk ships unverified, and that is
            # a problem worth a cause rather than a silent pass.
            ctx.record_error(
                step="verify",
                chunk_id=state["chunk"].id,
                cause="verify",
                message="; ".join(rejections)
                or "verification failed and the retranslate budget is used up",
            )
        return route

    g.add_conditional_edges(
        "critic",
        critic_edge,
        {"refine": "refiner", "verify": "verifier"},
    )
    g.add_edge("refiner", "critic")
    g.add_conditional_edges(
        "verifier",
        verifier_edge,
        {"retranslate": "retranslate", "next": "collect"},
    )
    g.add_edge("retranslate", "verifier")
    g.add_edge("collect", "save_trace")
    g.add_edge("save_trace", END)

    return g.compile()
