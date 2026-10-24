"""Skill store: one ``skills/<src>-<tgt>/skill.md`` file per language pair.

Frontmatter + markdown body, hand-parsed: PyYAML is not a project dependency
and the format is fixed (see the module's tests for the exact shape). Every
read is best-effort — a hand-edited or damaged file reads as "no skill",
never as an error.
"""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

#: Body section markers, in file order.
_SECTIONS = ("Instruction", "Few-shot", "Rationale")

#: Frontmatter keys, in file order; unknown keys are appended after these.
_META_ORDER = ("src_lang", "tgt_lang", "mean_score", "promoted_at", "updated_at", "scores")


def _parse_value(key: str, raw: str) -> Any:
    """One frontmatter value: ``scores`` lists filter non-numeric points."""
    if key == "scores" and raw.startswith("[") and raw.endswith("]"):
        values: list[float] = []
        for part in raw[1:-1].split(","):
            part = part.strip()
            if not part:
                continue
            try:
                score = float(part)
            except ValueError:
                continue  # a hand-edited non-numeric point is dropped
            if not math.isnan(score):
                values.append(score)
        return values
    try:
        return float(raw)
    except ValueError:
        return raw


class SkillStore:
    """Reads/writes ``skills/<src>-<tgt>/skill.md`` skill files."""

    def __init__(self, skills_dir: str | Path) -> None:
        self.dir = Path(skills_dir)

    def _path(self, src: str, tgt: str) -> Path:
        return self.dir / f"{src}-{tgt}" / "skill.md"

    # ------------------------------------------------------------------ read
    @staticmethod
    def _parse(text: str) -> tuple[dict[str, Any] | None, str]:
        """Split a skill file into (frontmatter, body); ``(None, "")`` when the
        frontmatter fence is missing or structurally broken."""
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return None, ""
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is None:
            return None, ""
        meta: dict[str, Any] = {}
        for line in lines[1:end]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, raw = line.partition(":")
            if not sep or not key.strip():
                return None, ""
            meta[key.strip()] = _parse_value(key.strip(), raw.strip())
        return meta, "\n".join(lines[end + 1 :])

    @staticmethod
    def _parse_body(body: str) -> dict[str, Any]:
        """The three sections of the markdown body, tolerantly."""
        collected: dict[str, list[str]] = {name: [] for name in _SECTIONS}
        current: str | None = None
        for line in body.splitlines():
            marker = line.strip()
            if marker in _SECTIONS:
                current = marker
                continue
            if current is not None:
                collected[current].append(line)
        return {
            "instruction": "\n".join(collected["Instruction"]).strip(),
            "few_shot": [line.strip() for line in collected["Few-shot"] if line.strip()],
            "rationale": "\n".join(collected["Rationale"]).strip(),
        }

    def _read(self, path: Path) -> tuple[dict[str, Any], str]:
        """(frontmatter, raw body); empty frontmatter when absent or broken."""
        if not path.exists():
            return {}, ""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("Skill: cannot read {}: {}", path, exc)
            return {}, ""
        meta, body = self._parse(text)
        if meta is None:
            logger.debug("Skill: unparsable frontmatter in {}", path)
            return {}, ""
        return meta, body

    def load(self, src: str, tgt: str) -> dict[str, Any] | None:
        """The promoted skill for this pair, or None: absent, damaged, or a
        scores-only file (score history without an ``Instruction`` is not a
        skill yet)."""
        path = self._path(src, tgt)
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("Skill: cannot read {}: {}", path, exc)
            return None
        meta, body = self._parse(text)
        if meta is None:
            logger.debug("Skill: unparsable frontmatter in {}", path)
            return None
        fields = self._parse_body(body)
        if not fields["instruction"]:
            return None  # scores-only file: nothing has been promoted yet
        out: dict[str, Any] = {key: meta[key] for key in _META_ORDER if key in meta}
        out.update(fields)
        return out

    # ----------------------------------------------------------------- write
    @staticmethod
    def _dump(meta: dict[str, Any]) -> str:
        lines = ["---"]
        for key in tuple(key for key in _META_ORDER if key in meta) + tuple(
            key for key in meta if key not in _META_ORDER
        ):
            value = meta[key]
            if isinstance(value, list):
                rendered = ", ".join(str(float(item)) for item in value)
                lines.append(f"{key}: [{rendered}]")
            else:
                lines.append(f"{key}: {value}")
        lines.append("---")
        return "\n".join(lines) + "\n"

    def _write(self, path: Path, meta: dict[str, Any], body: str) -> None:
        """Write the skill, atomically where the platform allows it.

        On Windows a replace onto a file another process holds open without
        delete sharing fails with ``PermissionError`` (WinError 5) — and a
        second run, an editor, or an indexer reading the skill while a run
        promotes it is ordinary. The replace is retried briefly, and if the
        handle is still there the text is written into the target directly:
        losing a promoted skill because a reader was looking at it is worse
        than the instant in which that reader may see a shorter file.

        The temporary name is unique per process and thread, so two runs
        promoting at once cannot truncate each other's staging file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self._dump(meta)
        if body.strip():
            text += "\n" + body.strip() + "\n"
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            for attempt in range(6):
                try:
                    tmp.replace(path)
                    return
                except PermissionError:
                    time.sleep(0.05 * (attempt + 1))
            path.write_text(text, encoding="utf-8")
            logger.warning(
                "Skill: {} is held open by another process; wrote it in place",
                path,
            )
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _scores(meta: dict[str, Any]) -> list[float]:
        raw = meta.get("scores")
        if not isinstance(raw, list):
            return []
        values: list[float] = []
        for item in raw:
            try:
                score = float(item)
            except (TypeError, ValueError):
                continue
            if not math.isnan(score):
                values.append(score)
        return values

    def record_score(self, src: str, tgt: str, score: float) -> None:
        """Append a critic score to the frontmatter (last 50), keeping any
        promoted content in the file. Creates the file when missing."""
        path = self._path(src, tgt)
        meta, body = self._read(path)
        values = self._scores(meta)
        new = float(score)
        if not math.isnan(new):
            values.append(new)
        meta.update(
            {
                "src_lang": src,
                "tgt_lang": tgt,
                "scores": values[-50:],
                "updated_at": time.time(),
            }
        )
        self._write(path, meta, body)  # body untouched: no instruction lost

    def mean_recent(self, src: str, tgt: str, n: int = 20) -> float:
        """Mean of the last *n* stored scores; 0.0 when there are none."""
        meta, _ = self._read(self._path(src, tgt))
        values = self._scores(meta)[-n:]
        return sum(values) / len(values) if values else 0.0

    def save(
        self,
        src: str,
        tgt: str,
        *,
        instruction: str,
        few_shot: list[str],
        rationale: str,
        mean_score: float,
    ) -> Path:
        """Promote a skill atomically, preserving the stored score history."""
        path = self._path(src, tgt)
        meta, _old_body = self._read(path)
        now = time.time()
        meta.update(
            {
                "src_lang": src,
                "tgt_lang": tgt,
                "mean_score": round(float(mean_score), 3),
                "promoted_at": now,
                "updated_at": now,
                "scores": self._scores(meta),
            }
        )
        body_parts = [f"Instruction\n{instruction.strip()}"]
        body_parts.append(
            "Few-shot\n" + "\n".join(str(item) for item in few_shot if str(item).strip())
        )
        body_parts.append(f"Rationale\n{rationale.strip()}")
        self._write(path, meta, "\n\n".join(body_parts))
        return path
