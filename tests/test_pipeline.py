"""End-to-end pipeline tests (mock LLM) + router unit tests."""

from __future__ import annotations

import json

import pytest

from polyglot_lc.orchestrator.router import after_critic, after_verifier, has_next_chunk


# ----------------------------- router rules ---------------------------------
class _C:
    def __init__(self, score: float) -> None:
        self.score = score


def test_router_critic_below_threshold_routes_to_refine():
    state = {"critique": _C(5.0), "refine_round": 0}
    assert after_critic(state, threshold=7.0, max_rounds=2) == "refine"


def test_router_critic_max_rounds_goes_to_verify():
    state = {"critique": _C(5.0), "refine_round": 2}
    assert after_critic(state, threshold=7.0, max_rounds=2) == "verify"


def test_router_critic_above_threshold_goes_to_verify():
    state = {"critique": _C(8.0), "refine_round": 0}
    assert after_critic(state, threshold=7.0, max_rounds=2) == "verify"


def test_router_verifier_fail_retranslates_once():
    state = {"verify_result": False, "retranslate_count": 0}
    assert after_verifier(state) == "retranslate"
    state2 = {"verify_result": False, "retranslate_count": 1}
    assert after_verifier(state2) == "next"


def test_router_next_chunk():
    assert has_next_chunk({"current_chunk": 1, "total_chunks": 3}) == "translate"
    assert has_next_chunk({"current_chunk": 3, "total_chunks": 3}) == "assemble"


# ------------------------------ pipeline ------------------------------------
def test_pipeline_end_to_end_mock(mock_env, sample_md_file):
    from polyglot_lc.pipeline import TranslationPipeline

    p = TranslationPipeline(quiet=True)
    result = p.run(sample_md_file, src="en", tgt="vi", domain="tech")
    assert result.chunks >= 1
    assert result.avg_critic > 0
    assert result.output_path.endswith(".vi.md")
    out = open(result.output_path, encoding="utf-8").read()
    assert out.strip()


def test_pipeline_writes_jsonl_log(mock_env, sample_md_file):
    import os

    os.environ["LOG_JSONL"] = "1"
    from polyglot_lc.pipeline import TranslationPipeline

    p = TranslationPipeline(quiet=True)
    result = p.run(sample_md_file, src="en", tgt="vi")
    assert result.log_path
    lines = [json.loads(l) for l in open(result.log_path, encoding="utf-8")]
    assert lines, "log must contain events"
    required = {"ts", "step", "chunk_id", "status", "latency", "error"}
    assert required.issubset(set(lines[0].keys()))
    steps = {l["step"] for l in lines}
    assert "ingest" in steps and "translate" in steps


def test_pipeline_missing_config_error_panel(monkeypatch, sample_md_file):
    # No env vars set -> ConfigError with a clear message.
    for name in ("PLANNER", "TRANSLATOR", "CRITIC", "REFINER", "VERIFIER",
                 "EMBED"):
        for suffix in ("BASE_URL", "MODEL", "API_KEY", "TEMPERATURE"):
            monkeypatch.delenv(f"{name}_{suffix}", raising=False)
    from polyglot_lc.pipeline import TranslationPipeline

    with pytest.raises(Exception, match="PLANNER_BASE_URL"):
        TranslationPipeline(quiet=True)


def test_pipeline_out_path_override(mock_env, sample_md_file, tmp_path):
    from polyglot_lc.pipeline import TranslationPipeline

    out = tmp_path / "sub" / "out.md"
    p = TranslationPipeline(quiet=True)
    result = p.run(sample_md_file, src="en", tgt="vi", out=str(out))
    assert result.output_path == str(out)
    assert out.exists()
