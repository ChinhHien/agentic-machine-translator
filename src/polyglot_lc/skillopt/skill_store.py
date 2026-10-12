"""Skill store: per language-pair/domain JSON files under skills/."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


class SkillStore:
    """Reads/writes ``skills/<src>-<tgt>/<domain>.json`` skill files."""

    def __init__(self, skills_dir: str | Path) -> None:
        self.dir = Path(skills_dir)

    def _path(self, src: str, tgt: str, domain: str) -> Path:
        return self.dir / f"{src}-{tgt}" / f"{domain or 'general'}.json"

    def load(self, src: str, tgt: str, domain: str) -> dict[str, Any] | None:
        """Return the promoted skill, or None when absent."""
        p = self._path(src, tgt, domain)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def save(
        self,
        src: str,
        tgt: str,
        domain: str,
        skill: dict[str, Any],
    ) -> Path:
        """Persist a skill file atomically and return its path."""
        p = self._path(src, tgt, domain)
        p.parent.mkdir(parents=True, exist_ok=True)
        skill.setdefault("updated_at", time.time())
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(skill, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
        return p

    def history_path(self, src: str, tgt: str, domain: str) -> Path:
        """Path of the rolling score-history file (not a promoted skill)."""
        return self._path(src, tgt, domain).with_name(f"{domain or 'general'}.history.jsonl")

    def append_score(self, src: str, tgt: str, domain: str, score: float) -> list[float]:
        """Append a critic score to history, keeping the last 50 entries."""
        hp = self.history_path(src, tgt, domain)
        hp.parent.mkdir(parents=True, exist_ok=True)
        scores: list[float] = []
        if hp.exists():
            for line in hp.read_text(encoding="utf-8").splitlines():
                try:
                    scores.append(float(json.loads(line)["score"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        scores.append(score)
        scores = scores[-50:]
        hp.write_text(
            "\n".join(json.dumps({"score": s, "ts": time.time()}) for s in scores) + "\n",
            encoding="utf-8",
        )
        return scores

    def recent_scores(self, src: str, tgt: str, domain: str, n: int = 20) -> list[float]:
        """The last *n* non-NaN scores for this pair/domain."""
        vals: list[float] = []
        hp = self.history_path(src, tgt, domain)
        if hp.exists():
            for line in hp.read_text(encoding="utf-8").splitlines():
                try:
                    s = float(json.loads(line)["score"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
                if not math.isnan(s):  # skip NaN sentinels
                    vals.append(s)
        return vals[-n:]

    recent_scores_all = recent_scores  # backwards-friendly alias

    def mean_recent(self, src: str, tgt: str, domain: str, n: int = 20) -> float:
        """Mean of the last *n* scores (excludes NaN sentinels)."""
        vals = self.recent_scores(src, tgt, domain, n)
        return sum(vals) / len(vals) if vals else 0.0
