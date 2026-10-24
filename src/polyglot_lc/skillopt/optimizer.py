"""SkillOpt: textual-gradient optimization of the Translator prompt skill.

After each chunk we collect (prompt, output, critic_score). The Planner LLM
proposes a *textual gradient* — an improved instruction/few-shot text. A new
skill is only promoted when the mean critic score of the last 20 chunks beats
the currently promoted skill's recorded mean. Scores and skill live together
in one ``skills/<src>-<tgt>/skill.md`` file per language pair.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from polyglot_lc.agents import PlannerAgent
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
        """Record one (prompt, output, score) observation and append history."""
        self._buffer.append(
            {
                "src": src,
                "tgt": tgt,
                "prompt": prompt,
                "output": output,
                "score": critic_score,
            }
        )
        self.store.record_score(src, tgt, critic_score)

    # ------------------------------------------------------------------
    def propose_gradient(self, src: str, tgt: str) -> dict[str, Any] | None:
        """Ask the Planner for a textual gradient over the current skill.

        Returns {instruction, few_shot, rationale} or None on failure.
        """
        current = self.store.load(src, tgt) or {}
        recent = self._recent_observations(src, tgt)
        if not recent:
            return None

        samples = "\n\n".join(
            f"--- sample {i + 1} (score {o['score']:.1f}) ---\n"
            f"PROMPT (tail): ...{o['prompt'][-300:]}\nOUTPUT: {o['output'][:400]}"
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
            + "\n\nPropose an improved instruction (max 4 sentences) and up to 3 "
            "few-shot examples that fix the observed failure modes. "
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
    def maybe_promote(self, src: str, tgt: str) -> dict[str, Any] | None:
        """Promote the buffered gradient skill if recent mean beat the old one.

        Returns the promoted skill dict, or None when not promoted.
        """
        gradient = self.propose_gradient(src, tgt)
        if gradient is None or not gradient["instruction"]:
            return None

        current = self.store.load(src, tgt)
        new_mean = self.store.mean_recent(src, tgt, WINDOW)
        old_mean = float(current.get("mean_score", 0.0)) if current else 0.0

        # Promotion rule (spec): the mean critic score of the last 20 chunks
        # must strictly beat the currently promoted skill's mean. With no
        # existing skill, the first promotion wins outright.
        if current is not None and new_mean <= old_mean:
            logger.debug(
                "SkillOpt: keep old skill (new {:.2f} <= old {:.2f})",
                new_mean,
                old_mean,
            )
            return None

        skill = {
            "instruction": gradient["instruction"],
            "few_shot": gradient["few_shot"],
            "rationale": gradient["rationale"],
            "mean_score": round(new_mean, 3),
            "promoted_at": time.time(),
            "src_lang": src,
            "tgt_lang": tgt,
        }
        self.store.save(
            src,
            tgt,
            instruction=gradient["instruction"],
            few_shot=gradient["few_shot"],
            rationale=gradient["rationale"],
            mean_score=round(new_mean, 3),
        )
        self._buffer = [
            b for b in self._buffer if not (b["src"] == src and b["tgt"] == tgt)
        ]
        logger.info(
            "SkillOpt promoted new skill for {} ({} > {})",
            f"{src}-{tgt}",
            new_mean,
            old_mean,
        )
        return skill

    # ------------------------------------------------------------------
    def _recent_observations(self, src: str, tgt: str) -> list[dict[str, Any]]:
        return [b for b in self._buffer if b["src"] == src and b["tgt"] == tgt]
