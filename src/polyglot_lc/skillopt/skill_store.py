"""Skill store: one ``skills/skill.md`` for the whole system.

The pipeline translates English into Vietnamese by default, and every document
goes through the same Translator prompt, so there is one skill, not one per
language pair: ``skills/skill.md`` is read by every run and promoted by every
run. Its history and its version snapshots sit beside it (``history.jsonl``,
``versions/v<N>.md``); an older per-pair file such as
``skills/en-vi/skill.md`` is still read when no global skill exists, and the
next promotion writes the global one.

Frontmatter + markdown body, hand-parsed: PyYAML is not a project dependency
and the format is fixed (see the module's tests for the exact shape). Every
read is best-effort — a hand-edited or damaged file reads as "no skill", never
as an error.

Schema 2 keeps the skill's identity and its measurements apart: ``skill_id`` /
``version`` say *which* skill this is, ``mean_score`` is the score of
translations produced **by this very version** (empty until a run has used it),
and ``baseline_mean`` is the score of the version it replaced — the number a
candidate has to beat. Schema 1 files (no ``schema`` line) still load: their
``mean_score`` becomes the baseline and they are measured from now on.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

#: Frontmatter flavour this store writes.
SKILL_SCHEMA = 2

#: Body section markers, in file order.
_SECTIONS = ("Instruction", "Few-shot", "Rationale")

#: Accepted spellings of a section heading (lower-cased, ``#``/``:`` removed).
_HEADINGS = {
    "instruction": "Instruction",
    "few-shot": "Few-shot",
    "few shot": "Few-shot",
    "fewshot": "Few-shot",
    "rationale": "Rationale",
}

#: A skill is prompt text, not a document: beyond this it stops being one.
MAX_INSTRUCTION_CHARS = 1200

#: Examples the translator prompt can carry without crowding out the chunk.
MAX_FEW_SHOT = 3

#: Frontmatter keys, in file order; unknown keys are appended after these.
_META_ORDER = (
    "schema",
    "skill_id",
    "version",
    "scope",
    "mean_score",
    "measured_runs",
    "baseline_mean",
    "promoted_at",
    "promoted_from_run",
    "updated_at",
    "scores",
)


def _parse_value(key: str, raw: str) -> Any:
    """One frontmatter value: ``scores`` lists filter non-numeric points.

    Only the keys that *are* numbers are coerced: ``skill_id`` is a short hex
    string that ``float`` happily accepts (``67631000`` -> ``67631000.0``),
    which would silently change an identity into a measurement.
    """
    if key in {"scores"} and raw.startswith("[") and raw.endswith("]"):
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
    if key not in {
        "schema",
        "version",
        "mean_score",
        "measured_runs",
        "baseline_mean",
        "promoted_at",
        "updated_at",
    }:
        return raw
    try:
        return float(raw)
    except ValueError:
        return raw


def skill_id_for(instruction: str, few_shot: Iterable[str]) -> str:
    """Short stable id of the skill's content: what a run can name in its log."""
    payload = instruction.strip() + "\n" + "\n".join(str(item) for item in few_shot)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _flatten(item: Any) -> str:
    """One few-shot example on one line, without a list marker."""
    text = re.sub(r"\s+", " ", str(item or "")).strip()
    return re.sub(r"^[-•*]\s*", "", text).strip()


def _clean_scores(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for item in values:
        try:
            score = float(item)
        except (TypeError, ValueError):
            continue
        if math.isnan(score):
            continue
        out.append(min(10.0, max(0.0, score)))
    return out


class SkillStore:
    """Reads/writes the system's single ``skills/skill.md`` skill file."""

    def __init__(self, skills_dir: str | Path) -> None:
        self.dir = Path(skills_dir)

    @property
    def path(self) -> Path:
        return self.dir / "skill.md"

    @property
    def versions_dir(self) -> Path:
        return self.dir / "versions"

    def _legacy_path(self) -> Path | None:
        """A pre-global ``skills/<pair>/skill.md``, if one is still around."""
        if not self.dir.is_dir():
            return None
        for child in sorted(self.dir.iterdir()):
            candidate = child / "skill.md"
            if child.is_dir() and candidate.exists():
                return candidate
        return None

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
        """The three sections of the markdown body, tolerantly.

        ``Instruction``, ``## Instruction`` and ``Instruction:`` all name the
        same section: the file is meant to be readable and editable by hand,
        and a heading level is not content.
        """
        collected: dict[str, list[str]] = {name: [] for name in _SECTIONS}
        current: str | None = None
        for line in body.splitlines():
            marker = re.sub(r"^#+\s*", "", line.strip()).rstrip(":").strip().lower()
            if marker in _HEADINGS:
                current = _HEADINGS[marker]
                continue
            if current is not None:
                collected[current].append(line)
        few_shot: list[str] = []
        for line in collected["Few-shot"]:
            item = _flatten(line)
            if item:
                few_shot.append(item)
        return {
            "instruction": "\n".join(collected["Instruction"]).strip(),
            "few_shot": few_shot,
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

    def load(self) -> dict[str, Any] | None:
        """The promoted system skill, or None: absent, damaged, or a scores-only
        file (score history without an ``Instruction`` is not a skill yet).

        Schema 1 files are accepted and presented in schema 2 terms: version 1,
        their recorded mean as the baseline, nothing measured under their own
        id yet.
        """
        path = self.path if self.path.exists() else self._legacy_path()
        if path is None or not path.exists():
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
        schema = int(float(meta.get("schema", 1) or 1))
        measured = meta.get("mean_score")
        measured = float(measured) if isinstance(measured, (int, float)) else None
        baseline = meta.get("baseline_mean")
        baseline = float(baseline) if isinstance(baseline, (int, float)) else None
        out: dict[str, Any] = {key: meta[key] for key in _META_ORDER if key in meta}
        out.update(fields)
        out.update(
            {
                "schema": schema,
                "skill_id": str(meta.get("skill_id") or skill_id_for(fields["instruction"], fields["few_shot"])),
                "version": int(float(meta.get("version", 1) or 1)),
                "scope": str(meta.get("scope") or "system"),
                "mean_score": measured,
                # Schema 1 knew only one number; it was the score of the skill
                # in the file, so it is the baseline for whatever replaces it.
                "baseline_mean": baseline if baseline is not None else (measured or 0.0),
                "measured_runs": int(float(meta.get("measured_runs", 0) or 0)),
            }
        )
        return out

    # ----------------------------------------------------------------- write
    @staticmethod
    def _dump(meta: dict[str, Any]) -> str:
        lines = ["---"]
        for key in tuple(key for key in _META_ORDER if key in meta) + tuple(
            key for key in meta if key not in _META_ORDER
        ):
            value = meta[key]
            if value is None:
                continue
            if isinstance(value, list):
                rendered = ", ".join(str(float(item)) for item in value)
                lines.append(f"{key}: [{rendered}]")
            elif isinstance(value, float):
                # Counters are integers even though they parse as floats: a
                # skill.md that says ``version: 2.0`` reads like a measurement.
                if key in {"schema", "version", "measured_runs"} and value.is_integer():
                    lines.append(f"{key}: {int(value)}")
                else:
                    lines.append(f"{key}: {round(value, 3)}")
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
    def _body(instruction: str, few_shot: list[str], rationale: str) -> str:
        parts = [f"Instruction\n{instruction.strip()}"]
        parts.append("Few-shot\n" + "\n".join(few_shot))
        parts.append(f"Rationale\n{rationale.strip()}")
        return "\n\n".join(parts)

    def save(
        self,
        *,
        instruction: str,
        few_shot: list[str],
        rationale: str,
        mean_score: float | None = None,
        baseline_mean: float | None = None,
        promoted_from_run: str = "",
        version: int | None = None,
        scope: str = "system",
    ) -> Path:
        """Promote the system skill atomically, preserving scores and versions.

        *mean_score* is a measurement of **this** content, so a freshly
        promoted skill leaves it unset; the run that uses it records the number.
        *baseline_mean* is the score of the version being replaced and defaults
        to its recorded measurement.
        """
        text = (instruction or "").strip()
        if not text:
            raise ValueError("skill instruction is empty")
        if len(text) > MAX_INSTRUCTION_CHARS:
            logger.warning(
                "Skill: instruction is {} chars; trimmed to {}",
                len(text),
                MAX_INSTRUCTION_CHARS,
            )
            text = text[:MAX_INSTRUCTION_CHARS].rstrip()
        examples: list[str] = []
        for item in few_shot or []:
            flat = _flatten(item)
            if flat and flat not in examples:
                examples.append(flat)
        examples = examples[:MAX_FEW_SHOT]

        old = self.load()
        now = time.time()
        number = version or ((old["version"] + 1) if old else 1)
        new_id = skill_id_for(text, examples)
        measured = None if mean_score is None else round(float(mean_score), 3)
        if baseline_mean is not None:
            baseline = round(float(baseline_mean), 3)
        elif old is not None:
            baseline = round(float(old["mean_score"] if old["mean_score"] is not None else old["baseline_mean"]), 3)
        else:
            baseline = 0.0

        meta: dict[str, Any] = {
            "schema": SKILL_SCHEMA,
            "skill_id": new_id,
            "version": number,
            "scope": scope,
            "baseline_mean": baseline,
            "measured_runs": 1 if measured is not None else 0,
            "promoted_at": now,
            "promoted_from_run": promoted_from_run,
            "updated_at": now,
            "scores": self._scores(self._read(self.path)[0]),
        }
        if measured is not None:
            meta["mean_score"] = measured
        body = self._body(text, examples, rationale)
        self._write(self.path, meta, body)
        # A snapshot per version: promotion overwrites the file, and "what did
        # v2 actually say" must stay answerable (and restorable).
        self._write(self.versions_dir / f"v{number}.md", meta, body)
        return self.path

    @staticmethod
    def _scores(meta: dict[str, Any]) -> list[float]:
        raw = meta.get("scores")
        if not isinstance(raw, list):
            return []
        return _clean_scores(raw)

    def record_score(self, score: float) -> None:
        """Append one critic score to the frontmatter (last 50), keeping any
        promoted content in the file. Creates the file when missing.

        Kept for callers that score chunk by chunk; a whole run should use
        :meth:`record_scores` once instead of rewriting the file per chunk.
        """
        self.record_scores([score])

    def record_scores(self, scores: Iterable[float]) -> None:
        """Append a run's scores in one write, body and identity untouched."""
        values = _clean_scores(scores)
        if not values:
            return
        meta, body = self._read(self.path)
        merged = self._scores(meta) + values
        meta.update({"scores": merged[-50:], "updated_at": time.time()})
        self._write(self.path, meta, body)  # body untouched: no instruction lost

    def mean_recent(self, n: int = 20) -> float:
        """Mean of the last *n* stored scores; 0.0 when there are none."""
        meta, _ = self._read(self.path)
        values = self._scores(meta)[-n:]
        return sum(values) / len(values) if values else 0.0

    # --------------------------------------------------------------- history
    @property
    def history_path(self) -> Path:
        return self.dir / "history.jsonl"

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        """The last *limit* run records, oldest first; broken lines are skipped."""
        if not self.history_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for line in self.history_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        except OSError as exc:
            logger.debug("Skill: cannot read history {}: {}", self.history_path, exc)
            return []
        return rows[-limit:] if limit > 0 else rows

    def record_history(self, row: dict[str, Any]) -> None:
        """Append one run record to ``history.jsonl`` (best-effort)."""
        payload = {"at": time.time(), **row}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:  # history is diagnostics, never a run's failure
            logger.warning("Skill: cannot append history {}: {}", self.history_path, exc)

    def record_run(
        self,
        *,
        run_id: str = "",
        doc: str = "",
        pair: str = "",
        used_skill_id: str = "",
        used_version: int = 0,
        scores: Iterable[float] = (),
        verify_fail_rate: float = 0.0,
        retranslate_rate: float = 0.0,
        promoted: bool = False,
        reason: str = "",
    ) -> dict[str, Any] | None:
        """Record one run: its measurement of the applied skill and its outcome.

        The measurement belongs to the version that actually produced those
        chunks. It is written into ``skill.md`` only while that version is still
        the file's own — a promotion replaces the content, and the new version
        starts unmeasured, with the old measurement as its baseline.
        """
        values = _clean_scores(scores)
        if not values:
            return None
        mean = round(sum(values) / len(values), 3)
        current = self.load()
        measured_now = bool(
            current is not None
            and used_skill_id
            and current["skill_id"] == used_skill_id
            and not promoted
        )
        if measured_now:
            meta, body = self._read(self.path)
            meta.update(
                {
                    "mean_score": mean,
                    "measured_runs": int(float(meta.get("measured_runs", 0) or 0)) + 1,
                    "updated_at": time.time(),
                }
            )
            self._write(self.path, meta, body)
        row = {
            "run_id": run_id,
            "doc": doc,
            "pair": pair,
            "skill_id": used_skill_id,
            "version": used_version,
            "n": len(values),
            "mean": mean,
            "verify_fail_rate": round(float(verify_fail_rate), 3),
            "retranslate_rate": round(float(retranslate_rate), 3),
            "promoted": bool(promoted),
            "reason": reason,
        }
        self.record_history(row)
        return row

    # ------------------------------------------------------------ versions
    def restore(self, version: int) -> Path:
        """Put a stored version back as the current skill, recording that act."""
        source = self.versions_dir / f"v{version}.md"
        if not source.exists():
            raise FileNotFoundError(f"no stored version v{version} in {self.dir}")
        text = source.read_text(encoding="utf-8")
        meta, body = self._parse(text)
        if meta is None or not self._parse_body(body)["instruction"]:
            raise ValueError(f"stored version v{version} is not a readable skill")
        meta["updated_at"] = time.time()
        self._write(self.path, meta, body)
        self.record_history(
            {
                "run_id": "",
                "doc": "",
                "pair": "",
                "skill_id": str(meta.get("skill_id", "")),
                "version": version,
                "n": 0,
                "mean": None,
                "promoted": False,
                "reason": "restored",
            }
        )
        return self.path

    def validate(self) -> list[str]:
        """Everything wrong with the system skill file, as readable sentences."""
        if not self.path.exists() and self._legacy_path() is None:
            return [f"chưa có skill ({self.path})"]
        path = self.path if self.path.exists() else self._legacy_path()
        assert path is not None
        meta, body = self._parse(path.read_text(encoding="utf-8"))
        if meta is None:
            return ["frontmatter không đọc được (thiếu hoặc hỏng dấu ---)"]
        fields = self._parse_body(body)
        issues: list[str] = []
        if not fields["instruction"]:
            issues.append("thiếu mục Instruction")
        if len(fields["instruction"]) > MAX_INSTRUCTION_CHARS:
            issues.append(
                f"Instruction dài {len(fields['instruction'])} ký tự (tối đa {MAX_INSTRUCTION_CHARS})"
            )
        if len(fields["few_shot"]) > MAX_FEW_SHOT:
            issues.append(f"{len(fields['few_shot'])} few-shot (tối đa {MAX_FEW_SHOT})")
        schema = int(float(meta.get("schema", 1) or 1))
        if schema < SKILL_SCHEMA:
            issues.append(f"schema {schema} (hiện tại {SKILL_SCHEMA}); sẽ được nâng khi promote")
        expected = skill_id_for(fields["instruction"], fields["few_shot"])
        if meta.get("skill_id") and str(meta["skill_id"]) != expected:
            issues.append("skill_id không khớp nội dung (file bị sửa tay?)")
        measured = meta.get("mean_score")
        if measured is not None and not isinstance(measured, (int, float)):
            issues.append("mean_score không phải số")
        return issues
