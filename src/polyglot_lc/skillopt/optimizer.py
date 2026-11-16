"""SkillOpt: textual-gradient optimization of the Translator prompt skill.

After each chunk we collect (prompt, output, critic_score). The Planner LLM
proposes a *textual gradient* — an improved instruction/few-shot text. A new
skill is only promoted when the run has enough evidence (``SKILL_MIN_SAMPLES``
chunks) and its mean critic score beats the score the current skill was
measured at by ``SKILL_MIN_GAIN``; a promotion that merely ties is noise, not an
improvement. Scores and skill live together in the system's one
``skills/skill.md`` (the pipeline translates en→vi by default), and every run
leaves a row in ``history.jsonl`` saying which skill it used, what it scored,
and whether anything was promoted.

The measurement recorded for a version is the score of translations that
version actually produced. A freshly promoted skill therefore starts with no
measurement at all — the run that proposed it was translated by its
predecessor — and the next run fills it in. That is what makes "is v3 better
than v2" a fact rather than a label.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from polyglot_lc.agents import PlannerAgent
from polyglot_lc.config import get_app_config
from polyglot_lc.skillopt.skill_store import SkillStore

WINDOW = 20

_DEFAULT_INSTRUCTION = "Translate faithfully and fluently."


class SkillOptimizer:
    """Collect per-chunk feedback and promote improved Translator skills."""

    def __init__(
        self,
        store: SkillStore,
        planner: PlannerAgent,
    ) -> None:
        self.store = store
        self.planner = planner
        #: Flat buffer: one run is one language pair.
        self._buffer: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def observe(
        self,
        src: str,
        tgt: str,
        prompt: str,
        output: str,
        critic_score: float,
    ) -> None:
        """Record one (prompt, output, score) observation for this run.

        Nothing is written to disk here: a 400-chunk run would rewrite the whole
        skill file 400 times, and the file is only worth writing once, when the
        run's own measurement is known.
        """
        self._buffer.append(
            {
                "src": src,
                "tgt": tgt,
                "prompt": prompt,
                "output": output,
                "score": critic_score,
            }
        )

    # ------------------------------------------------------------------
    def propose_gradient(self, src: str, tgt: str) -> dict[str, Any] | None:
        """Ask the Planner for a textual gradient over the current skill.

        Returns {instruction, few_shot, rationale} or None on failure.
        """
        current = self.store.load() or {}
        recent = self._recent_observations(src, tgt)
        if not recent:
            return None

        samples = "\n\n".join(
            f"--- sample {i + 1} (score {o['score']:.1f}) ---\n"
            f"SOURCE (tail): ...{o['prompt'][-300:]}\nTRANSLATION: {o['output'][:400]}"
            for i, o in enumerate(recent[-5:])
        )
        prompt = (
            "You optimize translation prompts via textual gradient descent.\n"
            f"Language pair: {src}->{tgt}\n"
            f"Current translator instruction:\n{current.get('instruction') or _DEFAULT_INSTRUCTION}\n"
            f"Current few-shot examples:\n"
            + "\n".join(current.get("few_shot") or [])
            + "\n\nRecent chunk samples with critic scores:\n"
            + samples
            + "\n\nPropose an improved instruction (max 4 sentences, general rules "
            "for this language pair rather than facts about one document) and up to "
            "3 one-line few-shot examples that fix the observed failure modes. "
            "Reply ONLY with JSON:\n"
            '{"instruction": "...", "few_shot": ["...", ...], "rationale": "..."}'
        )
        try:
            data = self.planner.chat_json([{"role": "user", "content": prompt}])
        except Exception as exc:  # noqa: BLE001 - optimization is best-effort
            logger.warning("SkillOpt gradient proposal failed: {}", exc)
            return None
        return {
            "instruction": str(data.get("instruction", "")),
            "few_shot": [str(x) for x in data.get("few_shot", [])][:3],
            "rationale": str(data.get("rationale", "")),
        }

    # ------------------------------------------------------------------
    def maybe_promote(
        self,
        src: str,
        tgt: str,
        *,
        run_id: str = "",
        doc: str = "",
        verify_fail_rate: float = 0.0,
        retranslate_rate: float = 0.0,
        applied: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Promote the buffered gradient skill when the run earns it.

        Also records this run's measurement of the skill it actually used (and
        its outcome) in ``history.jsonl``: a rejected candidate is evidence too.
        Returns the promoted skill dict, or None when not promoted.
        """
        recent = self._recent_observations(src, tgt)
        scores = [float(o["score"]) for o in recent]
        applied = applied if applied is not None else self.store.load()
        if not scores:
            return None
        mean = sum(scores) / len(scores)
        config = get_app_config()
        min_samples = max(1, int(config.skill_min_samples))
        min_gain = max(0.0, float(config.skill_min_gain))
        current = self.store.load()
        baseline = 0.0
        if current is not None:
            baseline = float(
                current["mean_score"]
                if current.get("mean_score") is not None
                else current["baseline_mean"]
            )

        reason = ""
        if len(scores) < min_samples:
            reason = f"chưa đủ mẫu ({len(scores)} < {min_samples})"
        elif current is not None and current.get("mean_score") is None:
            # A promotion is not evidence: the run that proposed a version was
            # translated by its predecessor, so this version has no score of
            # its own yet. Measuring it is what this run is for — promoting on
            # top of that measurement would leave the next version unmeasured
            # again, run after run.
            reason = "version hiện tại chưa được đo"
        elif current is not None and mean < baseline + min_gain:
            reason = f"chưa vượt baseline ({mean:.2f} < {baseline:.2f} + {min_gain:g})"
        gradient = None if reason else self.propose_gradient(src, tgt)
        if not reason and (gradient is None or not gradient["instruction"]):
            reason = "không có đề xuất instruction"

        if reason:
            self.store.record_run(
                run_id=run_id,
                doc=doc,
                pair=f"{src}-{tgt}",
                used_skill_id=str((applied or {}).get("skill_id", "")),
                used_version=int((applied or {}).get("version", 0) or 0),
                scores=scores,
                verify_fail_rate=verify_fail_rate,
                retranslate_rate=retranslate_rate,
                promoted=False,
                reason=reason,
            )
            logger.info(
                "SkillOpt: no promotion for {}-{} ({}) mean={:.2f} baseline={:.2f} n={}",
                src,
                tgt,
                reason,
                mean,
                baseline,
                len(scores),
            )
            self._buffer = [
                b for b in self._buffer if not (b["src"] == src and b["tgt"] == tgt)
            ]
            return None

        assert gradient is not None
        number = (current["version"] + 1) if current else 1
        self.store.save(
            instruction=gradient["instruction"],
            few_shot=gradient["few_shot"],
            rationale=gradient["rationale"],
            baseline_mean=baseline,
            mean_score=None,
            promoted_from_run=run_id,
            version=number,
        )
        skill = self.store.load()
        self.store.record_run(
            run_id=run_id,
            doc=doc,
            pair=f"{src}-{tgt}",
            used_skill_id=str((applied or {}).get("skill_id", "")),
            used_version=int((applied or {}).get("version", 0) or 0),
            scores=scores,
            verify_fail_rate=verify_fail_rate,
            retranslate_rate=retranslate_rate,
            promoted=True,
            reason=f"v{number} thay v{int((applied or {}).get('version', 0) or 0)}",
        )
        self._buffer = [b for b in self._buffer if not (b["src"] == src and b["tgt"] == tgt)]
        logger.info(
            "SkillOpt promoted v{} for {}-{} | mean={:.2f} > baseline={:.2f} n={}",
            number,
            src,
            tgt,
            mean,
            baseline,
            len(scores),
        )
        return skill

    # ------------------------------------------------------------------
    def _recent_observations(self, src: str, tgt: str) -> list[dict[str, Any]]:
        return [b for b in self._buffer if b["src"] == src and b["tgt"] == tgt]
