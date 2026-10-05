"""Translation graph: document-level planning + per-chunk graph.

The document-level step (:func:`plan_document`) runs once and is never
concurrent. Each independent chunk then flows through its own compiled graph::

    translator -> critic -> [refiner -> critic] (max REFINE_ROUNDS)
               -> verifier -> [retranslate -> verifier] (max 1)
               -> collect -> save_trace -> END

Chunks keep this internal dependency order; ``MAX_CONCURRENCY`` only parallelizes
*different* chunks (see :meth:`TranslationPipeline._run_chunks`).
"""

from __future__ import annotations

import threading
from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger

from polyglot_lc.agents import (
    CriticAgent,
    PlannerAgent,
    RefinerAgent,
    TranslatorAgent,
    VerifierAgent,
)
from polyglot_lc.config import get_app_config
from polyglot_lc.orchestrator.router import after_critic, after_verifier
from polyglot_lc.orchestrator.state import ChunkState


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

    def step(self, name: str, chunk_id: str = "") -> Any:
        """UI step context manager (no-op-ish when ui is None)."""
        if self.ui is not None:
            return self.ui.step(name, chunk_id)
        from contextlib import nullcontext

        return nullcontext()


def plan_document(
    ctx: GraphContext,
    *,
    doc_summary: str,
    src_lang: str,
    tgt_lang: str,
    domain: str,
    glossary_md: str,
) -> dict[str, Any]:
    """Document-level planning (runs once): reasoning-bank lookup + Planner.

    The Reasoning Bank is best-effort: a retrieval failure degrades to
    ``(no traces)`` and never blocks the run.
    """
    app_cfg = ctx.app
    with ctx.step("plan"):
        traces: list[Any] = []
        if ctx.reasoning_bank is not None:
            try:
                query = f"{domain} {doc_summary[:200]}"
                with ctx.rb_lock:
                    traces = ctx.reasoning_bank.retrieve(query, top_k=app_cfg.rb_top_k)
            except Exception as exc:  # noqa: BLE001 - RB is best-effort
                logger.warning("Reasoning Bank retrieval failed ({}); continuing", exc)
        plan = ctx.planner.plan(
            doc_summary=doc_summary,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            domain=domain,
            glossary_md=glossary_md,
            rb_traces=traces,
        )
        return {
            "strategy": plan["strategy"],
            "few_shot": plan["few_shot"],
            "pitfalls": plan["pitfalls"],
        }


def make_chunk_graph(ctx: GraphContext):
    """Build and compile the per-chunk translation StateGraph."""
    app_cfg = ctx.app

    def translator_node(state: ChunkState) -> dict[str, Any]:
        chunk = state["chunk"]
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
            )
            draft = ctx.translator.translate(prompt, chunk_id=chunk.id)
            return {"draft": draft, "critique": None, "refine_round": 0,
                    "retranslate_count": 0}

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
            )
            return {"draft": refined, "refine_round": state.get("refine_round", 0) + 1}

    def verifier_node(state: ChunkState) -> dict[str, Any]:
        chunk = state["chunk"]
        with ctx.step("verify", chunk.id):
            vr = ctx.verifier.verify(
                source=chunk.text,
                translation=state["draft"],
                src_lang=state["src_lang"],
                tgt_lang=state["tgt_lang"],
            )
            return {"verify_result": vr}

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
            )
            draft = ctx.translator.translate(prompt, chunk_id=chunk.id)
            return {"draft": draft, "verify_result": None,
                    "retranslate_count": state.get("retranslate_count", 0) + 1}

    def collect_node(state: ChunkState) -> dict[str, Any]:
        return {"translated": state["draft"]}

    def save_trace_node(state: ChunkState) -> dict[str, Any]:
        from polyglot_lc.reasoning_bank.schema import ReasoningTrace

        chunk = state["chunk"]
        with ctx.step("save_trace", chunk.id):
            critique = state.get("critique")
            score = (getattr(critique, "score", 5.0) or 0.0) / 10.0
            trace = ReasoningTrace(
                domain=state["domain"],
                src_lang=state["src_lang"],
                tgt_lang=state["tgt_lang"],
                strategy=state["strategy"],
                few_shot=state.get("few_shot", [])[:3],
                pitfalls=state.get("pitfalls", [])[:3]
                + [f"{chunk.id}: {getattr(critique, 'feedback_text', '')[:120]}"],
                score=round(score, 3),
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
    g.add_edge("translator", "critic")
    g.add_conditional_edges(
        "critic",
        lambda s: after_critic(
            s, app_cfg.critic_threshold, app_cfg.refine_rounds, app_cfg.refine_min_score
        ),
        {"refine": "refiner", "verify": "verifier"},
    )
    g.add_edge("refiner", "critic")
    g.add_conditional_edges(
        "verifier",
        lambda s: after_verifier(s, max_retries=1),
        {"retranslate": "retranslate", "next": "collect"},
    )
    g.add_edge("retranslate", "verifier")
    g.add_edge("collect", "save_trace")
    g.add_edge("save_trace", END)

    return g.compile()
