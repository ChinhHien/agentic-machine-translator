"""Stores for the web UI: configuration now, runs/skills/glossaries later.

Everything here is a plain file reader meant to be called from the server
process, so nothing in this module imports :mod:`polyglot_lc` (see the package
docstring in :mod:`webui`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

#: Settings the UI may change; values are written into `.env`.
EDITABLE_KEYS: tuple[str, ...] = (
    "CHUNK_MAX_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "MAX_CONCURRENCY",
    "CRITIC_THRESHOLD",
    "REFINE_ROUNDS",
    "REFINE_MIN_SCORE",
    "RB_TOP_K",
    "UNIFIED_PLANNER",
    "ADAPTIVE_QUALITY",
    "FAST_PATH_MAX_COMPLEXITY",
    "FAST_PATH_MIN_SCORE",
    "FAST_PATH_SKIP_LLM_VERIFY",
    "CONSISTENCY_CHECK",
    "CONSISTENCY_MAX_PATCHES",
    "PARTIAL_OUTPUT",
    "RESUME",
    "PDF_EXTRACTOR",
    "LLM_MOCK",
)

#: Settings the UI only displays. Repointing a path from the UI would send a
#: later run's output somewhere unexpected, for no benefit the UI can explain.
READONLY_KEYS: tuple[str, ...] = (
    "LOGS_DIR",
    "SKILLS_DIR",
    "REASONING_BANK_PATH",
    "CHECKPOINT_DIR",
)

#: Name fragments marking a value the server must never read or return.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")

#: A `NAME = value` (or bare `NAME`) assignment line.
_KEY_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")


def is_secret(name: str) -> bool:
    """True when *name* looks like a credential the UI must not touch.

    Markers are matched as whole underscore-separated segments, or as a suffix
    of the collapsed name, rather than as plain substrings. Plain substring
    matching flags ``CHUNK_MAX_TOKENS`` (it contains ``TOKEN``) and so would
    declare a token *budget* a credential — contradicting the spec, which lists
    that setting as editable. The direction is still conservative: a name is
    secret when it has any doubt, so ``x_token`` and ``OPENAI_APIKEY`` are both
    caught.
    """
    upper = name.upper()
    if any(segment in _SECRET_MARKERS for segment in upper.split("_")):
        return True
    collapsed = upper.replace("_", "")
    return any(collapsed.endswith(marker) for marker in _SECRET_MARKERS)


class ConfigStore:
    """Reads and writes `.env` without ever touching a secret's value.

    ``env_keys`` returns the *names* present in the file (never values), and
    ``save`` rewrites exactly the lines for the keys it was asked to change,
    leaving comments, blank lines and every other setting byte-for-byte alone.
    Writes go through a temporary file in the same directory so a reader can
    never see a half-written `.env`.
    """

    def __init__(self, root: Path, env_path: Path) -> None:
        self.root = root
        self.env_path = env_path

    def _read_text(self) -> str:
        """The file's exact text, with its own line endings preserved.

        ``Path.read_text`` uses universal newlines, which silently rewrites
        CRLF to LF on the way in; the style is then lost and ``save`` would
        write the whole file with the wrong ending. Reading with ``newline=""``
        keeps the bytes as they are.
        """
        try:
            with self.env_path.open("r", encoding="utf-8", newline="") as handle:
                return handle.read()
        except FileNotFoundError:
            return ""
        except OSError:  # pragma: no cover - unreadable or exotic file
            return ""

    def env_keys(self) -> set[str]:
        """Names of the settings assigned in `.env` — never their values."""
        return {
            match.group(1)
            for line in self._read_text().splitlines()
            if (match := _KEY_LINE_RE.match(line))
        }

    @staticmethod
    def _render(value: Any) -> str:
        """Render a value the way `python-dotenv` reads it back."""
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    def save(self, values: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
        """Apply an editable whitelist of *values*; return (saved, rejected)."""
        saved: list[str] = []
        rejected: list[dict[str, str]] = []
        text = self._read_text()
        newline = "\r\n" if "\r\n" in text else "\n"
        lines = text.splitlines()
        for key, value in values.items():
            # Secrets are refused before the whitelist check, so a credential
            # can never be written even if a whitelist entry were added wrongly.
            if is_secret(key):
                rejected.append({"key": key, "reason": "secret"})
                continue
            if key not in EDITABLE_KEYS:
                rejected.append({"key": key, "reason": "not editable"})
                continue
            rendered = f"{key}={self._render(value)}"
            for index, line in enumerate(lines):
                match = _KEY_LINE_RE.match(line)
                if match is not None and match.group(1) == key:
                    lines[index] = rendered
                    break
            else:
                lines.append(rendered)
            saved.append(key)
        if lines:
            self._write_atomic(newline.join(lines) + newline)
        return saved, rejected

    def _write_atomic(self, body: str) -> None:
        """Write *body* through a temp file so readers never see it half-written."""
        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.env_path.with_name(self.env_path.name + ".tmp")
        tmp.write_text(body, encoding="utf-8", newline="")
        os.replace(tmp, self.env_path)
