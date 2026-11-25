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
def optimizer(store, mock_env, monkeypatch):
    # Three chunks are enough here: the gate itself is tested with its own
    # values, and a 20-chunk run per test would only make the suite slower.
    monkeypatch.setenv("SKILL_MIN_SAMPLES", "3")
    return SkillOptimizer(store, PlannerAgent(get_agent_config("PLANNER")))


# ------------------------------- skill.md ----------------------------------


def test_skill_md_round_trip(store):
    path = store.save(
        instruction="Be terse.", few_shot=["a => b"], rationale="why", mean_score=8.42
    )
    assert path == store.path
    loaded = store.load()
    assert loaded["instruction"] == "Be terse."
    assert loaded["few_shot"] == ["a => b"]
    assert loaded["rationale"] == "why"
    assert loaded["mean_score"] == pytest.approx(8.42)
    assert loaded["scope"] == "system"
    assert loaded["schema"] == 2 and loaded["version"] == 1
    assert loaded["measured_runs"] == 1
    assert len(loaded["skill_id"]) == 8
    # There is exactly one skill file; no old snapshot is written beside it.
    assert not (store.dir / "versions").exists()


def test_schema_one_file_still_loads(store):
    """The old format (no schema/version lines) reads as a v1 baseline."""
    folder = store.dir
    folder.mkdir(parents=True)
    (folder / "skill.md").write_text(
        "---\nmean_score: 9.8\nscores: [8.0, 10.0]\nskill_id: old-format\n---\n"
        "Instruction\nBe terse.\n\nFew-shot\na => b\n\nRationale\nwhy\n",
        encoding="utf-8",
    )
    loaded = store.load()
    assert loaded["schema"] == 1 and loaded["version"] == 1
    assert loaded["baseline_mean"] == pytest.approx(9.8)
    assert loaded["measured_runs"] == 0
    assert loaded["instruction"] == "Be terse."


def test_heading_levels_and_bullets_are_content_free(store):
    folder = store.dir
    folder.mkdir(parents=True)
    (folder / "skill.md").write_text(
        "---\nschema: 2\n---\n## Instruction\nKeep placeholders.\n"
        "### Few-shot\n- a => b\n* c => d\n### Rationale\nbecause\n",
        encoding="utf-8",
    )
    loaded = store.load()
    assert loaded["instruction"] == "Keep placeholders."
    assert loaded["few_shot"] == ["a => b", "c => d"]
    assert loaded["rationale"] == "because"


def test_commented_lines_are_not_content(store):
    """A line wrapped in an HTML comment is switched off, not one more rule."""
    import hashlib

    folder = store.dir
    folder.mkdir(parents=True)
    (folder / "skill.md").write_text(
        "---\nschema: 2\n---\nInstruction\nKeep placeholders.\n\n"
        "Few-shot\na => b\n<!-- c => d -->\n\nRationale\nbecause\n",
        encoding="utf-8",
    )
    loaded = store.load()
    assert loaded["few_shot"] == ["a => b"]
    expected = hashlib.sha1("Keep placeholders.\na => b".encode("utf-8")).hexdigest()[:8]
    assert loaded["skill_id"] == expected
    assert store.validate() == []


def test_save_flattens_and_caps_few_shot(store):
    store.save(
        instruction="x",
        few_shot=["- a => b\nsecond line", "c => d", "e => f", "g => h"],
        rationale="r",
    )
    loaded = store.load()
    assert loaded["few_shot"] == ["a => b second line", "c => d", "e => f"]


def test_save_refuses_empty_instruction(store):
    with pytest.raises(ValueError):
        store.save(instruction="  ", few_shot=[], rationale="r")


def test_record_score_clips_to_50(store):
    store.save(instruction="x", few_shot=[], rationale="r", mean_score=0.0)
    for i in range(60):
        store.record_score(float(i % 10))
    assert len(store.load()["scores"]) == 50


def test_mean_recent_uses_last_20(store):
    store.save(instruction="x", few_shot=[], rationale="r", mean_score=0.0)
    for i in range(25):
        store.record_score(1.0 if i < 5 else 9.0)
    assert store.mean_recent() == pytest.approx(9.0)
    assert store.mean_recent(n=50) == pytest.approx((5 * 1.0 + 20 * 9.0) / 25)


def test_mean_recent_drops_nan_and_missing(store):
    store.save(instruction="x", few_shot=[], rationale="r", mean_score=0.0)
    store.record_score(8.0)
    store.record_score(float("nan"))
    assert store.mean_recent() == pytest.approx(8.0)


def test_save_writes_atomically_with_no_tmp_left(store):
    store.save(instruction="x", few_shot=[], rationale="r", mean_score=1.0)
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
    store.save(instruction="Be terse.", few_shot=[], rationale="r", mean_score=8.0)
    # The single skill file goes through ``replace``; the retries belong to
    # that write, not to the save as a whole.
    assert calls["n"] >= 3
    assert store.load()["instruction"] == "Be terse."
    assert not list(store.dir.rglob("*.tmp"))


def test_a_skill_held_open_is_still_written(store, monkeypatch):
    """The fallback writes the target in place rather than losing the skill."""
    from pathlib import Path

    def denied(self, target):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "replace", denied)
    path = store.save(
        instruction="Keep this.", few_shot=[], rationale="r", mean_score=8.0
    )
    assert "Keep this." in path.read_text(encoding="utf-8")
    assert store.load()["instruction"] == "Keep this."
    assert not list(store.dir.rglob("*.tmp"))


def test_load_corrupt_file_is_none(store):
    path = store.dir
    path.mkdir(parents=True)
    (path / "skill.md").write_text("not a skill at all", encoding="utf-8")
    assert store.load() is None


def test_record_score_before_first_promotion_keeps_scores(store):
    store.record_score(7.0)
    store.record_score(9.0)
    assert store.load() is None  # no instruction => not a skill yet
    assert store.mean_recent() == pytest.approx(8.0)


def test_save_and_record_score_do_not_clobber_each_other(store):
    store.save(
        instruction="Keep me.", few_shot=["a => b"], rationale="r", mean_score=5.0
    )
    store.record_score(9.0)
    assert store.load()["instruction"] == "Keep me."
    assert store.load()["scores"] == [9.0]
    store.save(instruction="Replaced.", few_shot=[], rationale="r2", mean_score=6.0)
    assert store.load()["scores"] == [9.0]  # scores survive a promotion


# ---------------------------- promotion logic ------------------------------


def test_first_promotion_when_no_skill(optimizer, store):
    for s in (6.0, 6.5, 7.0):
        optimizer.observe("en", "vi", "prompt", "output", s)
    skill = optimizer.maybe_promote("en", "vi")
    assert skill is not None
    # The run was translated by the previous (absent) skill: the new version
    # has no measurement of its own yet, and says so.
    assert skill["mean_score"] is None
    assert skill["measured_runs"] == 0
    assert skill["baseline_mean"] == pytest.approx(0.0)
    saved = store.load()
    assert saved is not None and saved["instruction"]


def test_no_promotion_without_history(optimizer, store):
    assert optimizer.maybe_promote("en", "vi") is None
    assert store.load() is None


def test_a_new_version_is_measured_before_it_is_judged(optimizer, store):
    for s in (8.0, 8.5, 9.0):
        optimizer.observe("en", "vi", "p", "o", s)
    first = optimizer.maybe_promote("en", "vi")
    assert first is not None and first["version"] == 1
    # The next run uses v1 and measures it; no second promotion on that same
    # measurement, however high it is.
    optimizer2 = SkillOptimizer(store, optimizer.planner)
    for s in (8.0, 8.5, 9.0):
        optimizer2.observe("en", "vi", "p", "o", s)
    assert optimizer2.maybe_promote("en", "vi") is None
    measured = store.load()
    assert measured["version"] == 1
    assert measured["mean_score"] == pytest.approx(8.5)
    assert measured["measured_runs"] == 1


def test_regression_not_promoted(optimizer, store):
    store.save(
        instruction="Be terse.", few_shot=[], rationale="r", mean_score=8.5
    )
    for s in (3.0, 3.5, 4.0):
        optimizer.observe("en", "vi", "p", "o", s)
    assert optimizer.maybe_promote("en", "vi") is None
    later = store.load()
    assert later["instruction"] == "Be terse."
    assert later["version"] == 1
    # Measured, not promoted: the run still counts as evidence about v1.
    assert later["mean_score"] == pytest.approx(3.5)


def test_improvement_promotes_new_skill(optimizer, store):
    store.save(
        instruction="Be terse.", few_shot=[], rationale="r", mean_score=4.0
    )
    for s in (9.0, 9.5, 9.0):
        optimizer.observe("en", "vi", "p", "o", s)
    second = optimizer.maybe_promote("en", "vi")
    assert second is not None and second["version"] == 2
    assert second["baseline_mean"] == pytest.approx(4.0)
    assert second["mean_score"] is None
    assert store.load()["mean_score"] is None
    assert store.load()["version"] == 2
    assert not (store.dir / "versions").exists()


def test_min_samples_blocks_promotion(optimizer, store, monkeypatch):
    monkeypatch.setenv("SKILL_MIN_SAMPLES", "5")
    for s in (9.0, 9.5):
        optimizer.observe("en", "vi", "p", "o", s)
    assert optimizer.maybe_promote("en", "vi") is None
    assert store.load() is None
    assert "chưa đủ mẫu" in store.history()[-1]["reason"]


def test_min_gain_blocks_a_tie(optimizer, store, monkeypatch):
    store.save(instruction="Be terse.", few_shot=[], rationale="r", mean_score=8.0)
    monkeypatch.setenv("SKILL_MIN_GAIN", "0.5")
    for s in (8.1, 8.2, 8.3):
        optimizer.observe("en", "vi", "p", "o", s)
    assert optimizer.maybe_promote("en", "vi") is None
    assert store.load()["version"] == 1
    assert "chưa vượt baseline" in store.history()[-1]["reason"]


def test_observing_does_not_write_the_skill_file(store, mock_env, monkeypatch):
    """One write per run, not one per chunk: 400 chunks must not mean 400 writes."""
    optimizer = SkillOptimizer(store, PlannerAgent(get_agent_config("PLANNER")))
    writes = {"n": 0}
    original = store._write

    def counting(path, meta, body):
        writes["n"] += 1
        return original(path, meta, body)

    monkeypatch.setattr(store, "_write", counting)
    for i in range(40):
        optimizer.observe("en", "vi", "p", "o", 8.0)
    assert writes["n"] == 0


def test_record_scores_appends_in_one_write_and_keeps_last_50(store):
    store.save(instruction="x", few_shot=[], rationale="r", mean_score=0.0)
    store.record_scores([float(i % 10) for i in range(60)])
    loaded = store.load()
    assert len(loaded["scores"]) == 50
    assert loaded["instruction"] == "x"  # body untouched by a score write


def test_record_run_measures_and_logs(optimizer, store):
    store.save(instruction="x", few_shot=[], rationale="r", mean_score=7.0)
    skill = store.load()
    row = store.record_run(
        run_id="r1",
        doc="d.pdf",
        used_skill_id=skill["skill_id"],
        used_version=1,
        scores=[9.0, 9.5],
        verify_fail_rate=0.25,
        retranslate_rate=0.5,
    )
    assert row["mean"] == pytest.approx(9.25) and row["n"] == 2
    assert store.load()["mean_score"] == pytest.approx(9.25)
    logged = store.history()[-1]
    assert logged["run_id"] == "r1" and logged["doc"] == "d.pdf"
    assert logged["version"] == 1 and logged["verify_fail_rate"] == pytest.approx(0.25)


def test_promoting_run_does_not_measure_the_new_version(store):
    store.record_run(
        run_id="r1",
        used_skill_id="old",
        used_version=1,
        scores=[9.0, 10.0],
        promoted=True,
    )
    assert store.load() is None  # no skill was written: only measured


def test_a_promotion_overwrites_the_single_skill_file(store):
    """There is no old-version archive: the next update replaces skill.md."""
    store.save(instruction="first", few_shot=[], rationale="r")
    store.save(instruction="second", few_shot=[], rationale="r")

    assert store.load()["instruction"] == "second"
    assert store.load()["version"] == 2
    assert [p.name for p in store.dir.glob("*.md")] == ["skill.md"]


def test_the_repository_skill_is_valid(store):
    from pathlib import Path

    repo_skills = SkillStore(Path(__file__).resolve().parents[1] / "skills")
    assert repo_skills.validate() == []


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
