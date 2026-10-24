# SkillOpt: skill.md store + close the learning loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SkillOpt stops being write-only: the promoted skill is stored as one human-readable `skills/<src>-<tgt>/skill.md` (no domain) and is read back every run into the Translator prompt.

**Architecture:** `SkillStore` is rewritten around a single `skill.md` file (YAML-ish frontmatter + markdown body, hand-parsed — no new deps); `SkillOptimizer` loses every `domain` parameter; the coordinator loads the skill once per run after planning, merges its few-shot into the plan's (skill first, dedupe, cap 6), carries `skill` in each chunk's state, and `translator_node` passes `skill_instruction` into `TranslatorAgent.build_prompt`.

**Tech Stack:** Python 3.11, loguru, pytest. No new dependencies.

**Spec:** the task message "Nhiệm vụ: Hoàn thiện SkillOpt — đóng vòng lặp học và chuyển sang file skill.md đơn giản" (in the conversation; acceptance criteria 1–5 are the contract).

## Global Constraints

- SkillOpt is best-effort: every load/save/promote failure logs a warning and continues; it must never fail a translation (keep the existing `except → warning` style at `coordinator.py:905-921`).
- No changes to Reasoning Bank, TM, router, checkpoint.
- `--domain` stays for Planner + Reasoning Bank; only SkillOpt drops it. `ChunkState.domain`, `doc_key(domain=...)`, `ui.set_stats(domain=...)` stay.
- No new dependency: PyYAML is NOT in `pyproject.toml`, so frontmatter is parsed by a minimal hand-rolled parser.
- LLM budget: skill work adds exactly the existing 1 gradient call per run; loading/reading the skill adds 0 LLM calls (`result.planner_calls` must stay 1 in the budget tests).
- Style: loguru `logger`, type hints, short English docstrings/comments — like the rest of the codebase.
- Full `pytest tests` suite green.

## Review Focus

Inputs/failure modes the tasks' tests do not all exercise, most likely first:

1. **Corrupt `skill.md` mid-run** — a hand-edited/garbage file must yield `load() -> None` and a run with zero `skill_instruction`, not a crash. Pinned in Task 2 by running a full translation with a garbage `skill.md`.
2. **Scores-only file** (record_score ran before the first promotion) must not read as a promoted skill, and run-1 scores must survive into `mean_recent`. Pinned in Task 1 (`test_record_score_before_first_promotion_keeps_scores`).
3. **save/record_score clobbering each other** (spec B3): `record_score` after `save` keeps the instruction; `save` after `record_score` keeps the scores. Pinned in Task 1 (`test_save_and_record_score_do_not_clobber_each_other`).
4. **The retranslate prompt site** (`graph.py:452-461`, the semantic-retry path) must also carry `skill_instruction`, not just the main site at 303. Pinned in Task 2 by forcing the retry marker (reuse the trigger from `test_request_budget_semantic_retry`) and asserting every captured `skill_instruction`.
5. **Shared cache prefix**: `messages[0]` (the cross-agent read-source prefix) must be byte-identical with and without a skill; the skill line lives in `messages[1]`. Pinned in Task 2 (`test_skill_line_does_not_touch_the_shared_prefix`).

---

### Task 1: Domain-free `SkillStore` on `skill.md` + `SkillOptimizer` API + coordinator call sites

**Files:**
- Modify: `src/polyglot_lc/skillopt/skill_store.py` (rewrite)
- Modify: `src/polyglot_lc/skillopt/optimizer.py`
- Modify: `src/polyglot_lc/pipeline/coordinator.py:905-921` (observe/promote call sites)
- Test: `tests/test_skillopt.py` (rewrite)
- Test: `tests/test_concurrency_budget.py` (add full-run promote test)

**Interfaces:**
- Produces (Task 2 consumes `SkillStore.load` and `SkillOptimizer.store`):
  - `SkillStore.load(src: str, tgt: str) -> dict | None` — promoted skill dict `{"instruction", "few_shot", "rationale", "mean_score", "src_lang", "tgt_lang", "promoted_at", "updated_at", "scores"}` (fields present only when in the file); `None` when absent, unparsable, or the `Instruction` section is missing/blank (scores-only files are not skills).
  - `SkillStore.record_score(src: str, tgt: str, score: float) -> None` — append to frontmatter `scores`, keep last 50, drop non-float/NaN entries, atomic write; creates the file (frontmatter only) when missing; preserves instruction/few-shot/rationale when present.
  - `SkillStore.mean_recent(src: str, tgt: str, n: int = 20) -> float` — mean of last `n` numeric `scores` from frontmatter; `0.0` when none.
  - `SkillStore.save(src: str, tgt: str, *, instruction: str, few_shot: list[str], rationale: str, mean_score: float) -> Path` — merge into existing file (scores preserved), sets `src_lang/tgt_lang/mean_score/promoted_at/updated_at`, rebuilds the body, atomic (`tmp` + `Path.replace`).
  - `SkillOptimizer.observe(src: str, tgt: str, prompt: str, output: str, critic_score: float) -> None`
  - `SkillOptimizer.propose_gradient(src: str, tgt: str) -> dict | None` — gradient prompt says `Language pair: {src}->{tgt}` only (no domain line); falls back to `Translate faithfully and fluently.` when the store has no instruction.
  - `SkillOptimizer.maybe_promote(src: str, tgt: str) -> dict | None` — unchanged rule: promote when no current skill, or `new_mean > old_mean`; returns the skill dict with `mean_score` (tests read `skill["mean_score"]`).
- File format (spec, verbatim):

```
---
src_lang: en
tgt_lang: vi
mean_score: 8.42
promoted_at: 1791623869.69
updated_at: 1792333594.66
scores: [8.5, 7.5]
---
Instruction
<1-4 sentences for the Translator>

Few-shot
source => target
source => target

Rationale
<why it was promoted>
```

Section markers are the bare lines `Instruction`, `Few-shot`, `Rationale`; body content runs until the next marker (or EOF). Parser is best-effort: malformed frontmatter, unparsable `scores`, or I/O errors log-and-return `None`/`0.0`/no-op — never raise from `load`.

- [ ] **Step 1: Write the failing tests (store level)** in `tests/test_skillopt.py`, replacing the domain-based ones:

```python
def test_skill_md_round_trip(store):
    path = store.save("en", "vi", instruction="Be terse.", few_shot=["a => b"], rationale="why", mean_score=8.42)
    assert path == store.dir / "en-vi" / "skill.md"
    loaded = store.load("en", "vi")
    assert loaded["instruction"] == "Be terse."
    assert loaded["few_shot"] == ["a => b"]
    assert loaded["rationale"] == "why"
    assert loaded["mean_score"] == pytest.approx(8.42)

def test_record_score_clips_to_50(store):
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

def test_load_corrupt_file_is_none(store):
    p = store.dir / "en-vi"; p.mkdir(parents=True)
    (p / "skill.md").write_text("not a skill at all", encoding="utf-8")
    assert store.load("en", "vi") is None

def test_record_score_before_first_promotion_keeps_scores(store):
    store.record_score("en", "vi", 7.0)
    store.record_score("en", "vi", 9.0)
    assert store.load("en", "vi") is None          # no instruction => not a skill
    assert store.mean_recent("en", "vi") == pytest.approx(8.0)

def test_save_and_record_score_do_not_clobber_each_other(store):
    store.save("en", "vi", instruction="Keep me.", few_shot=["a => b"], rationale="r", mean_score=5.0)
    store.record_score("en", "vi", 9.0)
    assert store.load("en", "vi")["instruction"] == "Keep me."
    assert store.load("en", "vi")["scores"] == [9.0]
```

Optimizer tests: same promotion-logic tests as today with the `domain` argument removed (`optimizer.observe("en", "vi", "prompt", "output", s)`, `optimizer.maybe_promote("en", "vi")`, `store.load("en", "vi")`); rename `test_domains_isolated` → `test_pairs_isolated` asserting `store.load("en", "de") is None` while `store.load("en", "vi")` is set; `test_history_window_keeps_last_50` drives `optimizer.observe` 60× and asserts `len(store.load("en", "vi")["scores"]) == 50`.

- [ ] **Step 2: Run `pytest tests/test_skillopt.py -q`** — Expected: FAIL (`TypeError: load() missing 1 required positional argument: 'domain'` and friends).

- [ ] **Step 3: Rewrite `src/polyglot_lc/skillopt/skill_store.py`** with the four methods above: `_path(src, tgt)` → `self.dir / f"{src}-{tgt}" / "skill.md"`; `_read_meta(path) -> dict | None` (hand parser: `---` fences, `key: value` on each line, `scores: [a, b]` inline list, every value coerced best-effort); `_write(path, meta, instruction, few_shot, rationale)` (serialize meta in the spec's order, body sections only when instruction non-empty; `tmp = path.with_name(path.name + ".tmp")` then `tmp.replace(path)`). Drop `_path(domain)`, `history_path`, `append_score`, `recent_scores`, `recent_scores_all`, `save(..., domain, dict)`.

- [ ] **Step 4: Run `pytest tests/test_skillopt.py -q`** — store tests PASS; optimizer tests still FAIL on old signatures.

- [ ] **Step 5: Adapt `src/polyglot_lc/skillopt/optimizer.py`**: drop `domain` from `observe`/`propose_gradient`/`maybe_promote`; flat `_buffer` filtered by `(src, tgt)` only; `self.store.record_score(src, tgt, score)` in `observe`; gradient prompt's `Language pair: {src}->{tgt}; domain: {domain}` → `Language pair: {src}->{tgt}`; `current = self.store.load(src, tgt) or {...}` keeps the `Translate faithfully and fluently.` fallback (use `current.get("instruction") or <fallback>`); `maybe_promote` calls `self.store.save(src, tgt, instruction=..., few_shot=..., rationale=..., mean_score=round(new_mean, 3))` and returns the skill dict (with `promoted_at`, `src_lang`, `tgt_lang` as before, no `domain` key).

- [ ] **Step 6: Update coordinator call sites** at `coordinator.py:912-920`: remove the positional `domain` argument from `skill_opt.observe(...)` and `skill_opt.maybe_promote(src, tgt)`; everything else in the block stays.

- [ ] **Step 7: Add the full-run promote test** (acceptance 1) to `tests/test_concurrency_budget.py`:

```python
def test_a_run_promotes_the_skill_to_skill_md(mock_env, tmp_path, monkeypatch):
    path = _write(tmp_path, PLAIN)
    _run(mock_env, monkeypatch, path, concurrency=2)
    skill_file = tmp_path / "skills" / "en-vi" / "skill.md"
    assert skill_file.exists()
    text = skill_file.read_text(encoding="utf-8")
    assert text.startswith("---") and "Instruction" in text and "mean_score:" in text
```

(Mock critic replies return `score 8.5` by default, so the first run's mean beats "no current skill".)

- [ ] **Step 8: Run `pytest tests/test_skillopt.py tests/test_concurrency_budget.py -q`** — Expected: PASS. If the promote test fails, debug the real cause (never relax the assertion).

- [ ] **Step 9: Commit** (or leave uncommitted if the partner declined commits):

```bash
git add src/polyglot_lc/skillopt/ src/polyglot_lc/pipeline/coordinator.py tests/test_skillopt.py tests/test_concurrency_budget.py
git commit -m "feat(skillopt): store one domain-free skill.md and drop the domain key"
```

---

### Task 2: Read the skill back — state → Translator prompt

**Files:**
- Modify: `src/polyglot_lc/orchestrator/state.py` (`ChunkState` doc-level block)
- Modify: `src/polyglot_lc/pipeline/coordinator.py` (load after planning, ~line 755; `base_state` ~line 761; helper next to `_collect_scores`)
- Modify: `src/polyglot_lc/orchestrator/graph.py:303-310` and `:452-461` (both `build_prompt` calls)
- Modify: `src/polyglot_lc/agents/translator.py` (`build_prompt`)
- Test: `tests/test_concurrency_budget.py`, `tests/test_skillopt.py`

**Interfaces:**
- Consumes: `SkillStore.load(src, tgt) -> dict | None` (Task 1); `skill_opt.store` already exists in the coordinator.
- Produces:
  - `ChunkState.skill: dict | None` — declared key (LangGraph drops undeclared keys).
  - `coordinator._merge_few_shot(skill_few_shot: list[str], plan_few_shot: list[str]) -> list[str]` — skill entries first, exact-string dedupe, `[:6]`.
  - `TranslatorAgent.build_prompt(..., skill_instruction: str = "") -> list[dict[str, str]]` — when non-empty, prepends to `user_parts` the part `Style learned over previous runs: <instruction>`; `messages[0]` untouched.
  - `state["skill"]` is the whole skill dict; nodes pass `skill_instruction=str((state.get("skill") or {}).get("instruction") or "")`.

- [ ] **Step 1: Write the failing tests**:

```python
# tests/test_skillopt.py — prefix stability (Review Focus 5)
def test_skill_line_does_not_touch_the_shared_prefix(mock_env):
    from polyglot_lc.agents import TranslatorAgent
    tr = TranslatorAgent(get_agent_config("TRANSLATOR"))
    base = dict(chunk_text="Hello.", src_lang="en", tgt_lang="vi", strategy="literal")
    without = tr.build_prompt(**base)
    with_skill = tr.build_prompt(**base, skill_instruction="Be terse.")
    assert with_skill[0] == without[0]
    assert with_skill[1]["content"].startswith("Style learned over previous runs: Be terse.")
    assert "Style learned" not in without[1]["content"]
```

```python
# tests/test_concurrency_budget.py — the read loop cannot be lost again (acceptance 2)
def test_promoted_skill_reaches_every_translator_prompt(mock_env, tmp_path, monkeypatch):
    skill_dir = tmp_path / "skills" / "en-vi"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.md").write_text(
        "---\nsrc_lang: en\ntgt_lang: vi\nmean_score: 8.42\n---\n"
        "Instruction\nNever invent names.\n\nFew-shot\nHai => Two\n\nRationale\nr\n",
        encoding="utf-8",
    )
    seen: list[str] = []
    from polyglot_lc.agents.translator import TranslatorAgent
    orig = TranslatorAgent.build_prompt
    def spy(self, *args, **kwargs):
        seen.append(str(kwargs.get("skill_instruction", "")))
        return orig(self, *args, **kwargs)
    monkeypatch.setattr(TranslatorAgent, "build_prompt", spy)
    path = _write(tmp_path, PLAIN)
    _run(mock_env, monkeypatch, path, concurrency=2, replies=RETRY_MARKER_REPLY)  # force the retranslate site too
    assert seen, "build_prompt never ran"
    assert all(s == "Never invent names." for s in seen)
```

(`RETRY_MARKER_REPLY`: copy the marker/reply pair from `test_request_budget_semantic_retry` so the retry call site at `graph.py:452` runs too — Review Focus 4.)

```python
# tests/test_concurrency_budget.py — corrupt skill.md is harmless (Review Focus 1)
def test_corrupt_skill_md_does_not_fail_the_run(mock_env, tmp_path, monkeypatch):
    skill_dir = tmp_path / "skills" / "en-vi"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.md").write_text("garbage without frontmatter", encoding="utf-8")
    path = _write(tmp_path, PLAIN)
    result, _snap, _by = _run(mock_env, monkeypatch, path, concurrency=2)
    assert result.output_path and __import__("pathlib").Path(result.output_path).exists()
```

- [ ] **Step 2: Run the three tests** — Expected: FAIL (`skill_instruction` unexpected kwarg / prompts equal).

- [ ] **Step 3: Implement.** (a) `state.py`: add `skill: dict[str, Any] | None` with a `#: promoted SkillOpt skill copied into every chunk (document-level)` comment. (b) `coordinator.py` between the plan resolve (`glossary_md = plan.glossary_md or "(none)"`) and `base_state`: load best-effort —

```python
try:
    skill = skill_opt.store.load(src, tgt)
except Exception as exc:  # noqa: BLE001 - a broken skill must not fail the run
    logger.warning("Skill load skipped: {}", exc)
    skill = None
```

then `base_state["skill"] = skill` and `"few_shot": _merge_few_shot((skill or {}).get("few_shot") or [], plan.few_shot)` (module-level helper next to `_collect_scores`). (c) `translator.py`: append `skill_instruction: str = ""` to `build_prompt`'s signature; first `user_parts` entry guarded by `if skill_instruction:` with the exact copy `Style learned over previous runs: ` prefix. (d) `graph.py`: add `skill_instruction=str((state.get("skill") or {}).get("instruction") or ""),` at both call sites (303 and 452).

- [ ] **Step 4: Run `pytest tests/test_skillopt.py tests/test_concurrency_budget.py -q`** — Expected: PASS.

- [ ] **Step 5: Commit** (per the partner's commit choice):

```bash
git add src/polyglot_lc/orchestrator/state.py src/polyglot_lc/orchestrator/graph.py src/polyglot_lc/agents/translator.py src/polyglot_lc/pipeline/coordinator.py tests/
git commit -m "feat(skillopt): apply the promoted skill to every Translator prompt"
```

---

### Task 3: Docs, legacy cleanup, acceptance sweep

**Files:**
- Modify: `README.md:188-190` (SkillOpt bullet), `README.md:302` (`--domain` row)
- Modify: `docs/configuration.md:113` (`SKILLS_DIR` row)
- Delete: `skills/en-vi/general.json`, `skills/en-vi/general.history.jsonl` (git-tracked legacy mocks; explicitly requested by the spec)

- [ ] **Step 1: README.md** — SkillOpt bullet becomes: collects `(prompt, output, score)` per chunk; Planner proposes a textual gradient; a promotion writes `skills/<src>-<tgt>/skill.md`; the next run reads that file back and injects its instruction + few-shot into the Translator prompt; promotion only when the 20-chunk mean **beats** the stored one. `--domain` row: `dùng cho Planner/RB` (drop SkillOpt).
- [ ] **Step 2: docs/configuration.md:113** — `SKILLS_DIR` description → `Nơi lưu skill.md của SkillOpt (skills/<src>-<tgt>/skill.md).`
- [ ] **Step 3: Delete the two legacy files**: `git rm skills/en-vi/general.json skills/en-vi/general.history.jsonl`.
- [ ] **Step 4: Acceptance sweep (criteria 3 + 4)**:

```bash
grep -rn "general\.json\|history\.jsonl\|append_score\|recent_scores" src tests --include=*.py   # Expected: no output
grep -rn "domain" src/polyglot_lc/skillopt/                                                     # Expected: no output
grep -n "domain" src/polyglot_lc/pipeline/coordinator.py | grep -v "domain=domain\|domain: str\|domain=domain, domain\|skill_opt"  # only Planner/RB/doc_key/ui lines remain
python -m pytest tests -q                                                                        # Expected: exit 0
```

- [ ] **Step 5: Commit** (per the partner's commit choice):

```bash
git add README.md docs/configuration.md && git rm -q skills/en-vi/general.json skills/en-vi/general.history.jsonl
git commit -m "docs(skillopt): describe skill.md and drop the legacy domain mocks"
```

## Self-review notes (decisions an implementer would otherwise guess)

- `load()` returns `None` for scores-only files (no `Instruction` section): preserves the old promotion semantics exactly (`current is None` on the first run) while `mean_recent` still reads run-1 scores from the frontmatter.
- `record_score` creates a frontmatter-only `skill.md` when missing (the old `append_score` created the history file; scores must never be lost).
- The few-shot merge does not mutate the plan object: it is applied in `base_state` construction, so a frozen/dataclass `DocumentPlan` cannot break it.
- The skill line goes into `messages[1]` (`user_parts`), never `messages[0]`: `source_context_messages` is shared with the Critic/Refiner for cache reuse.
- Both `build_prompt` call sites get `skill_instruction`, not just the main translator node.
