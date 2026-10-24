"""Tests for the Reasoning Bank: schema gate, retrieve/upsert, persistence."""

from __future__ import annotations

import pytest

from polyglot_lc.config import get_embed_config
from polyglot_lc.reasoning_bank.embedder import Embedder
from polyglot_lc.reasoning_bank.schema import MIN_UPSERT_SCORE, ReasoningTrace
from polyglot_lc.reasoning_bank.store import ReasoningBank


@pytest.fixture()
def bank(rb_dir, mock_env):
    return ReasoningBank(rb_dir, Embedder(get_embed_config()))


def test_upsert_score_gate(bank):
    good = ReasoningTrace(strategy="keep acronyms", score=0.9)
    assert bank.upsert(good) is True
    bad = ReasoningTrace(strategy="weak trace", score=MIN_UPSERT_SCORE - 0.1)
    assert bank.upsert(bad) is False
    assert bank.count() == 1


def test_boundary_score_accepted(bank):
    edge = ReasoningTrace(strategy="exactly at gate", score=MIN_UPSERT_SCORE)
    assert bank.upsert(edge) is True


def test_retrieve_returns_similar_first(bank):
    bank.upsert(
        ReasoningTrace(
            src_lang="en",
            tgt_lang="vi",
            strategy="Keep product names unchanged",
            score=0.95,
        )
    )
    bank.upsert(
        ReasoningTrace(
            src_lang="en",
            tgt_lang="vi",
            strategy="Use formal register for contracts",
            score=0.9,
        )
    )
    hits = bank.retrieve("product names in tech manuals", top_k=1)
    assert len(hits) == 1
    # The strategy comes back as the embedded document "en->vi: <strategy>".
    assert "Keep product names unchanged" in hits[0].strategy


def test_retrieve_empty_bank(bank):
    assert bank.retrieve("anything") == []


def test_trace_metadata_roundtrip(bank):
    t = ReasoningTrace(
        src_lang="en",
        tgt_lang="vi",
        strategy="strategy text",
        few_shot=["a -> b", "c -> d"],
        pitfalls=["p1", "p2"],
        score=0.8,
    )
    bank.upsert(t)
    hit = bank.retrieve("strategy text", top_k=1)[0]
    assert hit.few_shot == ["a -> b", "c -> d"]
    assert hit.pitfalls == ["p1", "p2"]
    assert abs(hit.score - 0.8) < 1e-6


def test_persistence_across_instances(rb_dir, mock_env):
    emb = Embedder(get_embed_config())
    b1 = ReasoningBank(rb_dir, emb)
    b1.upsert(ReasoningTrace(strategy="persist me", score=0.9))
    b2 = ReasoningBank(rb_dir, emb)
    assert b2.count() == 1
    hits = b2.retrieve("persist me", top_k=1)
    # Strategy is embedded as the document text "src->tgt: <strategy>".
    assert hits and "persist me" in hits[0].strategy


def test_embedder_hash_fallback_deterministic(mock_env):
    v1 = Embedder._hash_vector("same text")
    v2 = Embedder._hash_vector("same text")
    assert v1 == v2
    assert len(v1) == 256


def test_embedder_falls_back_on_api_error(mock_env, monkeypatch):
    emb = Embedder(get_embed_config())

    def boom(inputs):
        raise RuntimeError("api down")

    monkeypatch.setattr(emb.client, "embed", boom)
    vecs = emb.embed_texts(["hello"])
    assert len(vecs[0]) == 256
