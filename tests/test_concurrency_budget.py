"""Bounded-concurrency, request-budget and best-effort-subsystem tests.

These exercise the real orchestration (mock LLM mode) rather than mocking the
pipeline, so a future accidental extra LLM call changes the counts and fails.
"""

from __future__ import annotations

import random
import time

import pytest

from polyglot_lc.llm.client import TokenUsage, clear_mock_replies, register_mock_reply
from polyglot_lc.llm.resilience import RetryStats

# Heading-free text so deterministic verifier hard checks pass (LLM verifier
# runs). At max_tokens=15 this chunks into exactly 3 chunks.
PLAIN = (
    "Neural machine translation systems process text in segments.\n\n"
    "Each segment is translated with consistent terminology.\n\n"
    "Quality improves when glossary terms are respected.\n"
)
MAX_TOKENS = 15
EXPECTED_CHUNKS = 3


@pytest.fixture(autouse=True)
def _clean_mock_replies():
    clear_mock_replies()
    RetryStats.reset()
    TokenUsage.reset()
    yield
    clear_mock_replies()
    RetryStats.reset()
    TokenUsage.reset()


def _write(tmp_path, text, name="doc.pdf"):
    """Input is PDF-only, so a document has to be rendered into a real one."""
    from pdfdoc import write_pdf

    return write_pdf(tmp_path / name, text)


def _expected_chunks(path):
    from polyglot_lc.ingest import load_document
    from polyglot_lc.segment.chunker import chunk_document
    from polyglot_lc.segment.summary_tree import build_sections

    doc = load_document(path)
    chunks = chunk_document(doc, max_tokens=MAX_TOKENS, model="mock-model")
    return chunks, build_sections(chunks)


def _run(mock_env, monkeypatch, path, *, concurrency=None, replies=None, out=None):
    if replies:
        for marker, reply in replies.items():
            register_mock_reply(marker, reply)
    if concurrency is not None:
        monkeypatch.setenv("MAX_CONCURRENCY", str(concurrency))
    # Measure each run independently.
    TokenUsage.reset()
    RetryStats.reset()
    from polyglot_lc.pipeline import TranslationPipeline

    result = TranslationPipeline(quiet=True).run(
        path,
        src="en",
        tgt="vi",
        max_tokens=MAX_TOKENS,
        out=out or str(path.with_suffix(".vi.md")),
    )
    return result, TokenUsage.snapshot(), TokenUsage.snapshot_by_agent()


def _calls(by_agent, name):
    return by_agent.get(name, {}).get("calls", 0)


# --------------------------- MAX_CONCURRENCY config -------------------------
def test_max_concurrency_default_and_override(mock_env, monkeypatch):
    from polyglot_lc.config import get_app_config

    # The default is 1: a free tier serves one request at a time, so the
    # sequential run is the honest default and parallelism is opted into.
    assert get_app_config().max_concurrency == 1
    monkeypatch.setenv("MAX_CONCURRENCY", "5")
    assert get_app_config().max_concurrency == 5


def test_max_concurrency_out_of_range(mock_env, monkeypatch):
    from polyglot_lc.config import ConfigError, get_app_config

    monkeypatch.setenv("MAX_CONCURRENCY", "0")
    with pytest.raises(ConfigError, match="MAX_CONCURRENCY|max_concurrency"):
        get_app_config()


# ------------------------------- request budgets ----------------------------
def test_request_budget_adaptive_fast_path(mock_env, tmp_path, monkeypatch):
    """Default adaptive run: simple, cleanly-translated chunks skip the Critic.

    Each chunk costs translate + verify (2 requests) instead of the full
    translate + critic + verify (3), and the document plan costs ONE Planner
    request rather than one per section plus one.
    """
    path = _write(tmp_path, PLAIN)
    chunks, _sections = _expected_chunks(path)
    assert len(chunks) == EXPECTED_CHUNKS

    result, snap, by = _run(mock_env, monkeypatch, path, concurrency=1)

    n = len(chunks)
    assert _calls(by, "TRANSLATOR") == n
    assert _calls(by, "CRITIC") == 0  # fast path skips critic + refiner
    assert _calls(by, "REFINER") == 0
    assert _calls(by, "VERIFIER") == n
    # one document plan + one SkillOpt gradient call
    assert _calls(by, "PLANNER") == 2
    assert snap["calls"] == 2 * n + 2
    assert result.fast_path_chunks == n
    assert result.planner_calls == 1


def test_request_budget_full_quality_path(mock_env, tmp_path, monkeypatch):
    """ADAPTIVE_QUALITY=0 restores Translator -> Critic -> Verifier per chunk.

    The merged planner still applies, so this is strictly cheaper than the old
    per-section summary tree: 3*n + 2 requests instead of 3*n + sections + 3.
    """
    monkeypatch.setenv("ADAPTIVE_QUALITY", "0")
    path = _write(tmp_path, PLAIN)
    chunks, sections = _expected_chunks(path)

    result, snap, by = _run(mock_env, monkeypatch, path, concurrency=1)

    n = len(chunks)
    assert _calls(by, "TRANSLATOR") == n
    assert _calls(by, "CRITIC") == n
    assert _calls(by, "REFINER") == 0  # default score 8.5 >= threshold
    assert _calls(by, "VERIFIER") == n
    assert _calls(by, "PLANNER") == 2
    assert snap["calls"] == 3 * n + 2
    # The old pipeline paid one Planner request per section plus one.
    assert snap["calls"] < 3 * n + len(sections) + 3
    assert result.standard_path_chunks == n
    assert result.fast_path_chunks == 0


def test_request_budget_critic_requires_refine(mock_env, tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_QUALITY", "0")
    path = _write(tmp_path, PLAIN)
    chunks, _sections = _expected_chunks(path)

    replies = {"translation critic": '{"score": 3.0, "issues": ["x"], "suggestions": []}'}
    _result, _snap, by = _run(mock_env, monkeypatch, path, concurrency=1, replies=replies)

    n = len(chunks)
    # translate(1) + critic(3) + refiner(2) + verify(1) per chunk
    assert _calls(by, "TRANSLATOR") == n
    assert _calls(by, "CRITIC") == 3 * n
    assert _calls(by, "REFINER") == 2 * n
    assert _calls(by, "VERIFIER") == n
    # Scores below the threshold drove the deep path for every chunk.
    assert _result.deep_path_chunks == n
    assert _result.fast_path_chunks == 0


def test_request_budget_semantic_retry(mock_env, tmp_path, monkeypatch):
    """A verifier failure retranslates even on the fast path (no critique needed)."""
    path = _write(tmp_path, PLAIN)
    chunks, _sections = _expected_chunks(path)

    replies = {"translation QA verifier": '{"pass": false, "reasons": ["dropped"]}'}
    _result, _snap, by = _run(mock_env, monkeypatch, path, concurrency=1, replies=replies)

    n = len(chunks)
    # verifier fails -> retranslate -> verify again (bounded to one retry)
    assert _calls(by, "TRANSLATOR") == 2 * n
    assert _calls(by, "VERIFIER") == 2 * n
    assert _calls(by, "CRITIC") == 0
    assert _calls(by, "REFINER") == 0


def test_verifier_deterministic_failure_skips_llm(mock_env, tmp_path, monkeypatch):
    # A heading that the (mock) translation cannot preserve -> hard check fails
    # before any LLM verifier call.
    path = _write(tmp_path, "# Title\n\nA body paragraph long enough to translate.\n")
    _result, snap, by = _run(mock_env, monkeypatch, path, concurrency=1)

    assert _calls(by, "VERIFIER") == 0
    assert snap["calls"] > 0


# ----------------------------- adaptive refiner -----------------------------
def test_adaptive_refiner_skips_refine_for_high_score(mock_env, tmp_path, monkeypatch):
    # Exercises the refiner gate on the full path (the fast path bypasses it).
    monkeypatch.setenv("ADAPTIVE_QUALITY", "0")
    path = _write(tmp_path, PLAIN)
    chunks, _ = _expected_chunks(path)
    monkeypatch.setenv("CRITIC_THRESHOLD", "9.0")
    monkeypatch.setenv("REFINE_MIN_SCORE", "7.0")  # 8.0 >= 7.0 -> skip
    replies = {"translation critic": '{"score": 8.0, "issues": [], "suggestions": []}'}

    _result, _snap, by = _run(mock_env, monkeypatch, path, concurrency=1, replies=replies)
    assert _calls(by, "REFINER") == 0
    assert _calls(by, "VERIFIER") == len(chunks)


def test_adaptive_refiner_refines_below_min_score(mock_env, tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_QUALITY", "0")
    path = _write(tmp_path, PLAIN)
    chunks, _ = _expected_chunks(path)
    monkeypatch.setenv("CRITIC_THRESHOLD", "9.0")
    monkeypatch.setenv("REFINE_MIN_SCORE", "9.0")  # 8.0 < 9.0 -> refine
    replies = {"translation critic": '{"score": 8.0, "issues": [], "suggestions": []}'}

    _result, _snap, by = _run(mock_env, monkeypatch, path, concurrency=1, replies=replies)
    assert _calls(by, "REFINER") == 2 * len(chunks)


def test_adaptive_refiner_router_direct():
    from polyglot_lc.orchestrator.router import after_critic

    class C:
        def __init__(self, s):
            self.score = s

    # score >= refine_min_score -> verify (skip refiner)
    assert after_critic({"critique": C(8.0), "refine_round": 0}, 9.0, 2, 7.0) == "verify"
    # score < refine_min_score and < threshold -> refine
    assert after_critic({"critique": C(8.0), "refine_round": 0}, 9.0, 2, 9.0) == "refine"
    # rounds exhausted -> verify regardless
    assert after_critic({"critique": C(1.0), "refine_round": 2}, 9.0, 2, 9.0) == "verify"


# ------------------------------- concurrency --------------------------------
def test_concurrency_matches_sequential_budget(mock_env, tmp_path, monkeypatch):
    path = _write(tmp_path, PLAIN)
    chunks, _ = _expected_chunks(path)

    _r1, seq_snap, seq_by = _run(mock_env, monkeypatch, path, concurrency=1)
    _r4, par_snap, par_by = _run(mock_env, monkeypatch, path, concurrency=4)

    # Concurrency must not increase the number of requests (token counts may
    # differ because run 2 sees reasoning-bank traces from run 1).
    assert seq_snap["calls"] == par_snap["calls"]
    seq_calls = {k: v["calls"] for k, v in seq_by.items()}
    par_calls = {k: v["calls"] for k, v in par_by.items()}
    assert seq_calls == par_calls
    assert _calls(par_by, "TRANSLATOR") == len(chunks)


def test_output_order_preserved_under_concurrency(mock_env, tmp_path, monkeypatch):
    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.translator import TranslatorAgent

    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")  # isolate ordering semantics
    path = _write(tmp_path, PLAIN)
    chunks, _ = _expected_chunks(path)

    def fake_translate(self, prompt, chunk_id=""):
        time.sleep(random.uniform(0.0, 0.03))  # shuffle completion order
        return "T::" + extract_source(prompt)

    monkeypatch.setattr(TranslatorAgent, "translate", fake_translate)
    out = tmp_path / "ordered.vi.md"
    _run(mock_env, monkeypatch, path, concurrency=4, out=str(out))

    # Pipeline assembles chunks in original order regardless of completion order.
    expected = "\n\n".join(f"T::{c.text}" for c in chunks)
    assert out.read_text(encoding="utf-8") == expected


def test_chunk_failure_propagates_under_concurrency(mock_env, tmp_path, monkeypatch):
    from polyglot_lc.agents.translator import TranslatorAgent

    path = _write(tmp_path, PLAIN)
    out = tmp_path / "should_not_exist.vi.md"

    def boom(self, prompt, chunk_id=""):
        raise RuntimeError("chunk boom")

    monkeypatch.setattr(TranslatorAgent, "translate", boom)
    with pytest.raises(RuntimeError, match="chunk boom"):
        _run(mock_env, monkeypatch, path, concurrency=4, out=str(out))
    assert not out.exists()  # a failed run writes no partial output file


def test_the_run_end_record_carries_the_api_call_concurrency(mock_env, tmp_path, monkeypatch):
    """The Report tab's "Api calls" reads the run's own
    records; for a finished run the last one to carry the count is ``run_end``.
    The counter itself is a process-wide reading (pinned in
    ``tests/test_llm_client.py``), so what is pinned here is the other half: the
    coordinator writes down the snapshot it read rather than dropping it, which
    is what keeps the page off its "chưa đo được" fallback for a fresh run.
    """

    import json
    from pathlib import Path

    from polyglot_lc.pipeline import coordinator

    monkeypatch.setenv("LOG_JSONL", "1")
    monkeypatch.setattr(coordinator, "active_api_calls", lambda: {"active": 2, "peak": 3})
    path = _write(tmp_path, PLAIN)
    result, _snap, _by = _run(mock_env, monkeypatch, path)

    records = [
        json.loads(line) for line in Path(result.log_path).read_text(encoding="utf-8").splitlines()
    ]
    end = [record for record in records if record.get("type") == "run_end"][-1]
    assert (end["active_api_calls"], end["max_active_api_calls"]) == (2, 3)


# ------------------- LLM_PROVIDER_CONCURRENCY: provider-boundary queue ------
# These mock-mode tests exercise LLM_PROVIDER_CONCURRENCY directly (no real
# provider). Worker pool (MAX_CONCURRENCY) and provider queue (LLM_PROVIDER_
# CONCURRENCY) stay separate by design: the queue bounds how many requests are
# in flight at the provider; the worker pool may run more workers in parallel.


def test_provider_concurrency_respects_configured_limit(mock_env, monkeypatch):
    """LLM_PROVIDER_CONCURRENCY stands as configured.

    The user sets a larger limit for a paid/self-hosted provider and the
    config honours it; the pipeline can then run as many requests in flight
    as the limit allows. The default (no explicit value) stays at the safe 1.
    """
    from polyglot_lc.config import get_retry_config

    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY", "4")
    assert get_retry_config().provider_concurrency == 4
    monkeypatch.delenv("LLM_PROVIDER_CONCURRENCY", raising=False)
    assert get_retry_config().provider_concurrency == 1


def test_provider_concurrency_survives_pipeline_run(mock_env, tmp_path, monkeypatch):
    """The queue limit is a config value, not a behaviour gate.

    Ingest a 3-chunk document and run it with concurrency=4 (workers > chunks
    is legal); the provider queue runs at its configured limit regardless, so
    the run succeeds end-to-end. The queue did not change the number of
    requests made — workers can still overlap their work; only the
    provider-boundary concurrency is bound.
    """
    from polyglot_lc.config import get_retry_config

    monkeypatch.setenv("LLM_PROVIDER_CONCURRENCY", "2")
    assert get_retry_config().provider_concurrency == 2
    path = _write(tmp_path, PLAIN)
    chunks, _ = _expected_chunks(path)
    assert len(chunks) == EXPECTED_CHUNKS  # ingest worked; 3 chunks
    result, _snap, _by = _run(mock_env, monkeypatch, path, concurrency=4)
    assert result.chunks == EXPECTED_CHUNKS
    assert _calls(_by, "TRANSLATOR") == EXPECTED_CHUNKS


# --------------------------- best-effort subsystems -------------------------
class _BoomRB:
    def retrieve(self, *a, **k):
        raise RuntimeError("rb retrieval down")

    def upsert(self, *a, **k):
        raise RuntimeError("rb upsert down")


def _ctx(mock_env, rb):
    from polyglot_lc.agents import (
        CriticAgent,
        PlannerAgent,
        RefinerAgent,
        TranslatorAgent,
        VerifierAgent,
    )
    from polyglot_lc.config import get_agent_config
    from polyglot_lc.orchestrator.graph import GraphContext

    return GraphContext(
        planner=PlannerAgent(get_agent_config("PLANNER")),
        translator=TranslatorAgent(get_agent_config("TRANSLATOR")),
        critic=CriticAgent(get_agent_config("CRITIC")),
        refiner=RefinerAgent(get_agent_config("REFINER")),
        verifier=VerifierAgent(get_agent_config("VERIFIER")),
        reasoning_bank=rb,
    )


def test_chunk_survives_rb_upsert_failure(mock_env, tmp_path):
    from polyglot_lc.orchestrator.graph import make_chunk_graph

    path = _write(tmp_path, PLAIN)
    chunks, _ = _expected_chunks(path)
    ctx = _ctx(mock_env, _BoomRB())
    graph = make_chunk_graph(ctx)
    out = graph.invoke(
        {
            "chunk": chunks[0],
            "src_lang": "en",
            "tgt_lang": "vi",
            "strategy": "literal",
            "few_shot": [],
            "pitfalls": [],
            "glossary_md": "",
        },
        config={"recursion_limit": 100},
    )
    assert out["translated"]
    assert out["stored"] is False  # RB failure swallowed


def test_a_chunk_is_hard_checked_once_per_draft(mock_env, tmp_path, monkeypatch):
    """Hard checks ran up to three times over the same text; now once.

    The translator scored the draft with ``draft_signals`` (one scan), the fast
    verify path scanned it again, and ``VerifierAgent.verify`` scanned it a
    third time. Every draft is prepared once now, and its problems travel with
    the state to both verify paths.
    """
    from polyglot_lc.orchestrator import graph as graph_module
    from polyglot_lc.orchestrator.graph import make_chunk_graph

    calls: list[str] = []
    real = graph_module.hard_checks

    def counting(source, translation):
        calls.append(source)
        return real(source, translation)

    monkeypatch.setattr(graph_module, "hard_checks", counting)
    path = _write(tmp_path, PLAIN)
    chunks, _ = _expected_chunks(path)
    out = make_chunk_graph(_ctx(mock_env, None)).invoke(
        {
            "chunk": chunks[0],
            "src_lang": "en",
            "tgt_lang": "vi",
            "strategy": "literal",
            "few_shot": [],
            "pitfalls": [],
            "glossary_md": "",
        },
        config={"recursion_limit": 100},
    )
    assert out["translated"]
    assert len(calls) == 1


def test_skillopt_failure_does_not_fail_run(mock_env, tmp_path, monkeypatch):
    from polyglot_lc.skillopt.optimizer import SkillOptimizer

    def boom(self, *a, **k):
        raise RuntimeError("skillopt down")

    monkeypatch.setattr(SkillOptimizer, "maybe_promote", boom)
    path = _write(tmp_path, PLAIN)
    result, _snap, _by = _run(mock_env, monkeypatch, path, concurrency=2)
    assert result.skill_promoted is False
    assert result.output_path and __import__("pathlib").Path(result.output_path).exists()


def test_a_run_promotes_the_skill_to_skill_md(mock_env, tmp_path, monkeypatch):
    """A finished run with good critic scores writes the one skill.md file
    (frontmatter + markdown body), acceptance criterion 1."""
    path = _write(tmp_path, PLAIN)
    result, _snap, _by = _run(mock_env, monkeypatch, path, concurrency=2)
    assert result.skill_promoted is True  # mock critic scores are 8.5
    skill_file = tmp_path / "skills" / "en-vi" / "skill.md"
    assert skill_file.exists()
    text = skill_file.read_text(encoding="utf-8")
    assert text.startswith("---") and "mean_score:" in text
    assert "\nInstruction\n" in text and "\nRationale\n" in text


def test_promoted_skill_reaches_every_translator_prompt(mock_env, tmp_path, monkeypatch):
    """The read loop: a skill.md on disk puts its instruction into every
    Translator prompt of the next run — main site and retranslate site alike
    (the verifier-failure reply forces the retry). Acceptance criterion 2."""
    skill_dir = tmp_path / "skills" / "en-vi"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.md").write_text(
        "---\nsrc_lang: en\ntgt_lang: vi\nmean_score: 8.42\n---\n"
        "Instruction\nNever invent names.\n\nFew-shot\nHai => Two\n\nRationale\nr\n",
        encoding="utf-8",
    )
    seen: list[str] = []
    seen_few_shot: list[list[str]] = []
    from polyglot_lc.agents.translator import TranslatorAgent

    orig = TranslatorAgent.build_prompt

    def spy(self, *args, **kwargs):
        seen.append(str(kwargs.get("skill_instruction", "")))
        seen_few_shot.append(list(kwargs.get("few_shot") or []))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(TranslatorAgent, "build_prompt", spy)
    path = _write(tmp_path, PLAIN)
    replies = {"translation QA verifier": '{"pass": false, "reasons": ["dropped"]}'}
    _run(mock_env, monkeypatch, path, concurrency=1, replies=replies)
    assert seen, "build_prompt never ran"
    assert all(s == "Never invent names." for s in seen), seen
    # The skill's few-shot is merged into every prompt's examples too
    # (acceptance criterion 2, second half).
    assert all("Hai => Two" in fs for fs in seen_few_shot), seen_few_shot


def test_corrupt_skill_md_does_not_fail_the_run(mock_env, tmp_path, monkeypatch):
    """A hand-damaged skill.md reads as no skill; the translation still ends."""
    skill_dir = tmp_path / "skills" / "en-vi"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.md").write_text("garbage without frontmatter", encoding="utf-8")
    path = _write(tmp_path, PLAIN)
    result, _snap, _by = _run(mock_env, monkeypatch, path, concurrency=2)
    assert result.output_path and __import__("pathlib").Path(result.output_path).exists()


# --------------------------- glossary context bound -------------------------
def test_render_user_glossary_is_capped():
    from polyglot_lc.segment.glossary import MAX_USER_GLOSSARY_TERMS, render_user_glossary

    mapping = {f"term{i}": f"value{i}" for i in range(MAX_USER_GLOSSARY_TERMS + 50)}
    rendered = render_user_glossary(mapping)
    assert rendered.count("\n") + 1 == MAX_USER_GLOSSARY_TERMS
    assert "term0" in rendered and f"term{MAX_USER_GLOSSARY_TERMS}" not in rendered


# --------------------------- cache fingerprint ------------------------------
def test_cache_fingerprint_stable_and_sensitive():
    from polyglot_lc.agents.context import stable_prefix_fingerprint

    msgs = [{"role": "system", "content": "shared prefix text"}]
    fp = stable_prefix_fingerprint(msgs, model="m1")
    assert fp == stable_prefix_fingerprint(msgs, model="m1")  # stable
    assert fp != stable_prefix_fingerprint(msgs, model="m2")  # model-sensitive
    assert fp != stable_prefix_fingerprint(
        [{"role": "system", "content": "other prefix"}], model="m1"
    )
    assert "shared prefix" not in fp  # only a digest is exposed, never raw text


# ----------------------- emphasis mismatch at the chunk level ---------------
def test_settle_emphasis_drops_a_lopsided_pair():
    from polyglot_lc.orchestrator.graph import settle_emphasis

    text, dropped = settle_emphasis("A ⟦BOLD⟧B⟦/BOLD⟧ C", "A B ⟦BOLD⟧C")
    assert text == "A B C"
    assert dropped is True


def test_settle_emphasis_keeps_a_matching_pair():
    from polyglot_lc.orchestrator.graph import settle_emphasis

    text, dropped = settle_emphasis("A ⟦BOLD⟧B⟦/BOLD⟧ C", "A ⟦BOLD⟧B⟦/BOLD⟧ C")
    assert text == "A ⟦BOLD⟧B⟦/BOLD⟧ C"
    assert dropped is False


def test_a_lopsided_emphasis_pair_is_recorded_as_an_emphasis_error(mock_env, tmp_path, monkeypatch):
    """Không tốn lượt dịch lại vì kiểu chữ, nhưng phải thấy được trong trace."""
    import json

    from pdfdoc import write_styled_pdf

    from polyglot_lc.llm.client import LLMResponse
    from polyglot_lc.orchestrator.graph import make_chunk_graph
    from polyglot_lc.ui.events import EventSink

    path = write_styled_pdf(tmp_path / "styled.pdf", "Alpha ", "Beta", " Gamma")
    chunks, _ = _expected_chunks(path)
    assert "⟦BOLD⟧" in chunks[0].text, "nguồn thật phải mang token"
    sink = EventSink(tmp_path, enabled=True)
    ctx = _ctx(mock_env, _BoomRB())
    ctx.sink = sink
    # Bản dịch mock làm mất token đóng → lệch → phải bị gỡ và ghi lại.
    monkeypatch.setattr(
        ctx.translator,
        "chat",
        lambda *a, **k: LLMResponse(text="Alpha ⟦BOLD⟧Beta Gamma", model="m"),
    )
    out = make_chunk_graph(ctx).invoke(
        {
            "chunk": chunks[0],
            "src_lang": "en",
            "tgt_lang": "vi",
            "strategy": "literal",
            "few_shot": [],
            "pitfalls": [],
            "glossary_md": "",
        },
        config={"recursion_limit": 100},
    )

    assert "⟦" not in out["translated"]
    records = [json.loads(line) for line in sink.path.read_text(encoding="utf-8").splitlines()]
    assert any(r.get("cause") == "emphasis" for r in records), records
