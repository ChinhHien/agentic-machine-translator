"""Tests for SkillOpt: the skill.md store, promotion logic, textual gradient."""

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


# ------------------------------- skill.md ----------------------------------


def test_skill_md_round_trip(store):
    path = store.save(
        "en", "vi", instruction="Be terse.", few_shot=["a => b"], rationale="why", mean_score=8.42
    )
    assert path == store.dir / "en-vi" / "skill.md"
    loaded = store.load("en", "vi")
    assert loaded["instruction"] == "Be terse."
    assert loaded["few_shot"] == ["a => b"]
    assert loaded["rationale"] == "why"
    assert loaded["mean_score"] == pytest.approx(8.42)
    assert loaded["src_lang"] == "en" and loaded["tgt_lang"] == "vi"


def test_record_score_clips_to_50(store):
    store.save("en", "vi", instruction="x", few_shot=[], rationale="r", mean_score=0.0)
    for i in range(60):
        store.record_score("en", "vi", float(i % 10))
    assert len(store.load("en", "vi")["scores"]) == 50


def test_mean_recent_uses_last_20(store):
    store.save("en", "vi", instruction="x", few_shot=[], rationale="r", mean_score=0.0)
    for i in range(25):
        store.record_score("en", "vi", 1.0 if i < 5 else 9.0)
    assert store.mean_recent("en", "vi") == pytest.approx(9.0)
    assert store.mean_recent("en", "vi", n=50) == pytest.approx((5 * 1.0 + 20 * 9.0) / 25)


def test_mean_recent_drops_nan_and_missing(store):
    store.save("en", "vi", instruction="x", few_shot=[], rationale="r", mean_score=0.0)
    store.record_score("en", "vi", 8.0)
    store.record_score("en", "vi", float("nan"))
    assert store.mean_recent("en", "vi") == pytest.approx(8.0)
    assert store.load("en", "fr") is None


def test_save_writes_atomically_with_no_tmp_left(store):
    store.save("en", "vi", instruction="x", few_shot=[], rationale="r", mean_score=1.0)
    assert not list(store.dir.rglob("*.tmp"))


def test_a_locked_skill_is_replaced_after_a_brief_retry(store, monkeypatch):
    """A reader holding the file open on Windows is transient, not fatal.

    ``os.replace`` fails with WinError 5 while another process reads the target
    without delete sharing; a retry a moment later succeeds.
    """
    from pathlib import Path

    original = Path.replace
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "Access is denied")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    store.save("en", "vi", instruction="Be terse.", few_shot=[], rationale="r", mean_score=8.0)
    assert calls["n"] == 3
    assert store.load("en", "vi")["instruction"] == "Be terse."
    assert not list(store.dir.rglob("*.tmp"))


def test_a_skill_held_open_is_still_written(store, monkeypatch):
    """The fallback writes the target in place rather than losing the skill."""
    from pathlib import Path

    def denied(self, target):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "replace", denied)
    path = store.save(
        "en", "vi", instruction="Keep this.", few_shot=[], rationale="r", mean_score=8.0
    )
    assert "Keep this." in path.read_text(encoding="utf-8")
    assert store.load("en", "vi")["instruction"] == "Keep this."
    assert not list(store.dir.rglob("*.tmp"))


def test_load_corrupt_file_is_none(store):
    path = store.dir / "en-vi"
    path.mkdir(parents=True)
    (path / "skill.md").write_text("not a skill at all", encoding="utf-8")
    assert store.load("en", "vi") is None


def test_record_score_before_first_promotion_keeps_scores(store):
    store.record_score("en", "vi", 7.0)
    store.record_score("en", "vi", 9.0)
    assert store.load("en", "vi") is None  # no instruction => not a skill yet
    assert store.mean_recent("en", "vi") == pytest.approx(8.0)


def test_save_and_record_score_do_not_clobber_each_other(store):
    store.save(
        "en", "vi", instruction="Keep me.", few_shot=["a => b"], rationale="r", mean_score=5.0
    )
    store.record_score("en", "vi", 9.0)
    assert store.load("en", "vi")["instruction"] == "Keep me."
    assert store.load("en", "vi")["scores"] == [9.0]
    store.save("en", "vi", instruction="Replaced.", few_shot=[], rationale="r2", mean_score=6.0)
    assert store.load("en", "vi")["scores"] == [9.0]  # scores survive a promotion


# ---------------------------- promotion logic ------------------------------


def test_first_promotion_when_no_skill(optimizer, store):
    for s in (6.0, 6.5, 7.0):
        optimizer.observe("en", "vi", "prompt", "output", s)
    skill = optimizer.maybe_promote("en", "vi")
    assert skill is not None
    assert skill["mean_score"] == pytest.approx(6.5)
    saved = store.load("en", "vi")
    assert saved is not None and saved["instruction"]


def test_no_promotion_without_history(optimizer, store):
    assert optimizer.maybe_promote("en", "vi") is None
    assert store.load("en", "vi") is None


def test_regression_not_promoted(optimizer, store):
    # First promote a good skill.
    for s in (8.0, 8.5, 9.0):
        optimizer.observe("en", "vi", "p", "o", s)
    first = optimizer.maybe_promote("en", "vi")
    assert first is not None
    saved = store.load("en", "vi")
    # Then worse scores -> the skill content must NOT be replaced (the score
    # history legitimately grows in the same file).
    optimizer2 = SkillOptimizer(store, optimizer.planner)
    for s in (3.0, 3.5, 4.0):
        optimizer2.observe("en", "vi", "p", "o", s)
    assert optimizer2.maybe_promote("en", "vi") is None
    later = store.load("en", "vi")
    assert later["instruction"] == saved["instruction"]
    assert later["few_shot"] == saved["few_shot"]
    assert later["mean_score"] == saved["mean_score"]


def test_improvement_promotes_new_skill(optimizer, store):
    for s in (4.0, 4.5):
        optimizer.observe("en", "vi", "p", "o", s)
    first = optimizer.maybe_promote("en", "vi")
    assert first is not None and first["mean_score"] < 5.0
    optimizer2 = SkillOptimizer(store, optimizer.planner)
    for s in (9.0, 9.5):
        optimizer2.observe("en", "vi", "p", "o", s)
    second = optimizer2.maybe_promote("en", "vi")
    assert second is not None
    assert second["mean_score"] > first["mean_score"]
    assert store.load("en", "vi")["mean_score"] == second["mean_score"]


def test_history_window_keeps_last_50(optimizer, store):
    store.save("en", "vi", instruction="x", few_shot=[], rationale="r", mean_score=0.0)
    for i in range(60):
        optimizer.observe("en", "vi", "p", "o", float(i % 10))
    assert len(store.load("en", "vi")["scores"]) == 50


def test_mean_recent_excludes_sentinels(optimizer, store):
    optimizer.observe("en", "vi", "p", "o", 8.0)
    store.record_score("en", "vi", float("nan"))
    assert store.mean_recent("en", "vi") == pytest.approx(8.0)


def test_pairs_isolated(optimizer, store):
    for s in (6.0, 6.5, 7.0):
        optimizer.observe("en", "vi", "p", "o", s)
    optimizer.maybe_promote("en", "vi")
    assert store.load("en", "vi") is not None
    assert store.load("en", "de") is None


def test_gradient_uses_llm(optimizer):
    for s in (6.0, 7.0, 7.5):
        optimizer.observe("en", "vi", "prompt text", "output text", s)
    gradient = optimizer.propose_gradient("en", "vi")
    assert gradient is not None
    assert gradient["instruction"]
    assert isinstance(gradient["few_shot"], list)


# ---------------------------- the read loop --------------------------------


def test_few_shot_merge_puts_skill_first_deduped_and_capped():
    from polyglot_lc.pipeline.coordinator import _merge_few_shot

    merged = _merge_few_shot(
        ["a => A", "b => B"],
        ["a => A", "c => C"] + [f"p{i}" for i in range(8)],
    )
    assert merged[:2] == ["a => A", "b => B"]  # skill examples win the front
    assert merged.count("a => A") == 1  # deduped
    assert len(merged) == 6  # capped like the planner's few-shot


def test_skill_line_does_not_touch_the_shared_prefix(mock_env):
    """The skill line lives in the dynamic user part; messages[0] (the
    cross-agent read-source prefix the provider caches) must not change."""
    from polyglot_lc.agents.translator import TranslatorAgent

    tr = TranslatorAgent(get_agent_config("TRANSLATOR"))
    base = dict(chunk_text="Hello.", src_lang="en", tgt_lang="vi", strategy="literal")
    without = tr.build_prompt(**base)
    with_skill = tr.build_prompt(**base, skill_instruction="Be terse.")
    assert with_skill[0] == without[0]
    assert with_skill[1]["content"].startswith("Style learned over previous runs: Be terse.")
    assert "Style learned" not in without[1]["content"]
