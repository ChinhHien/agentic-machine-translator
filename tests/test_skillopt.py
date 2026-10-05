"""Tests for SkillOpt: promotion logic, score history, textual gradient."""

from __future__ import annotations

import pytest

from polyglot_lc.agents import PlannerAgent
from polyglot_lc.config import get_agent_config
from polyglot_lc.skillopt.optimizer import SkillOptimizer
from polyglot_lc.skillopt.skill_store import SkillStore


@pytest.fixture()
def store(tmp_path):
    return SkillStore(tmp_path / "skills")


@pytest.fixture()
def optimizer(store, mock_env):
    return SkillOptimizer(store, PlannerAgent(get_agent_config("PLANNER")))


def test_first_promotion_when_no_skill(optimizer, store):
    for s in (6.0, 6.5, 7.0):
        optimizer.observe("en", "vi", "tech", "prompt", "output", s)
    skill = optimizer.maybe_promote("en", "vi", "tech")
    assert skill is not None
    assert skill["mean_score"] == pytest.approx(6.5)
    saved = store.load("en", "vi", "tech")
    assert saved is not None and saved["instruction"]


def test_no_promotion_without_history(optimizer, store):
    assert optimizer.maybe_promote("en", "vi", "tech") is None
    assert store.load("en", "vi", "tech") is None


def test_regression_not_promoted(optimizer, store):
    # First promote a good skill.
    for s in (8.0, 8.5, 9.0):
        optimizer.observe("en", "vi", "tech", "p", "o", s)
    first = optimizer.maybe_promote("en", "vi", "tech")
    assert first is not None
    saved = store.load("en", "vi", "tech")
    # Then worse scores -> must NOT replace.
    optimizer2 = SkillOptimizer(store, optimizer.planner)
    for s in (3.0, 3.5, 4.0):
        optimizer2.observe("en", "vi", "tech", "p", "o", s)
    optimizer2.maybe_promote("en", "vi", "tech")
    assert store.load("en", "vi", "tech") == saved


def test_improvement_promotes_new_skill(optimizer, store):
    for s in (4.0, 4.5):
        optimizer.observe("en", "vi", "tech", "p", "o", s)
    first = optimizer.maybe_promote("en", "vi", "tech")
    assert first is not None and first["mean_score"] < 5.0
    optimizer2 = SkillOptimizer(store, optimizer.planner)
    for s in (9.0, 9.5):
        optimizer2.observe("en", "vi", "tech", "p", "o", s)
    second = optimizer2.maybe_promote("en", "vi", "tech")
    assert second is not None
    assert second["mean_score"] > first["mean_score"]
    assert store.load("en", "vi", "tech")["mean_score"] == second["mean_score"]


def test_history_window_keeps_last_50(optimizer, store):
    for i in range(60):
        optimizer.observe("en", "vi", "legal", "p", "o", float(i % 10))
    scores = store.recent_scores("en", "vi", "legal", n=50)
    assert len(scores) == 50


def test_mean_recent_excludes_sentinels(optimizer, store):
    optimizer.observe("en", "vi", "d", "p", "o", 8.0)
    store.append_score("en", "vi", "d", float("nan"))
    assert store.mean_recent("en", "vi", "d") == pytest.approx(8.0)


def test_domains_isolated(optimizer, store):
    for s in (6.0, 6.5, 7.0):
        optimizer.observe("en", "vi", "tech", "p", "o", s)
    optimizer.maybe_promote("en", "vi", "tech")
    assert store.load("en", "vi", "tech") is not None
    assert store.load("en", "vi", "legal") is None


def test_gradient_uses_llm(optimizer):
    for s in (6.0, 7.0, 7.5):
        optimizer.observe("en", "vi", "tech", "prompt text", "output text", s)
    gradient = optimizer.propose_gradient("en", "vi", "tech")
    assert gradient is not None
    assert gradient["instruction"]
    assert isinstance(gradient["few_shot"], list)
