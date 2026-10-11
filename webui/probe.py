"""Child-process probe: effective configuration and per-file chunk planning.

Runs as a **separate process** so the server never imports :mod:`polyglot_lc`
(see the package docstring in :mod:`webui`). It prints exactly one JSON object to
stdout on success; on any failure it prints a message to stderr, leaves stdout
empty and exits non-zero, so the caller can show a real reason instead of a
traceback.

Usage::

    python webui/probe.py --config
    python webui/probe.py --plan FILE --src en --tgt vi [--max-tokens 700]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

#: Settings the UI may edit: environment-variable name -> AppConfig attribute.
#: Kept here rather than imported from :mod:`webui.store` because a script's
#: ``sys.path[0]`` is its own directory, not the repository root, so ``probe``
#: cannot import the package it lives in. ``tests/test_webui_probe.py`` asserts
#: that these keys and the store's whitelist stay identical.
EDITABLE_FIELDS: dict[str, str] = {
    "CHUNK_MAX_TOKENS": "chunk_max_tokens",
    "CHUNK_OVERLAP_TOKENS": "chunk_overlap_tokens",
    "MAX_CONCURRENCY": "max_concurrency",
    "CRITIC_THRESHOLD": "critic_threshold",
    "REFINE_ROUNDS": "refine_rounds",
    "REFINE_MIN_SCORE": "refine_min_score",
    "RB_TOP_K": "rb_top_k",
    "UNIFIED_PLANNER": "unified_planner",
    "ADAPTIVE_QUALITY": "adaptive_quality",
    "FAST_PATH_MAX_COMPLEXITY": "fast_path_max_complexity",
    "FAST_PATH_MIN_SCORE": "fast_path_min_score",
    "FAST_PATH_SKIP_LLM_VERIFY": "fast_path_skip_llm_verify",
    "CONSISTENCY_CHECK": "consistency_check",
    "CONSISTENCY_MAX_PATCHES": "consistency_max_patches",
    "PARTIAL_OUTPUT": "partial_output",
    "RESUME": "resume",
    "PDF_EXTRACTOR": "pdf_extractor",
}

#: Settings the UI only displays.
READONLY_FIELDS: dict[str, str] = {
    "LOGS_DIR": "logs_dir",
    "SKILLS_DIR": "skills_dir",
    "REASONING_BANK_PATH": "reasoning_bank_path",
    "CHECKPOINT_DIR": "checkpoint_dir",
}

#: Not an ``AppConfig`` field: the demo switch is read straight from the env.
MOCK_KEY = "LLM_MOCK"

#: The tokenizer used when the translator is not configured yet. Matches
#: :func:`polyglot_lc.segment.chunker.chunk_document`'s own default.
FALLBACK_MODEL = "gpt-4o"

#: Every name this probe is allowed to look up, so the `.env` parser can ignore
#: everything else: a credential's value is then never read into memory at all.
KNOWN_KEYS = frozenset(EDITABLE_FIELDS) | frozenset(READONLY_FIELDS) | {MOCK_KEY}

_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _dotenv_path() -> Path:
    """The `.env` beside the repository root (see :mod:`webui.paths`)."""
    return Path(__file__).resolve().parents[1] / ".env"


def _parse_value(raw: str) -> str:
    """A `.env` value, with quotes or a trailing comment removed."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    for marker in (" #", "\t#"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def _dotenv_pairs() -> dict[str, str]:
    """Whitelist values assigned in `.env`.

    Only known setting names are read; every other line — including a
    credential's — is skipped without its value ever being interpreted.
    """
    try:
        with _dotenv_path().open("r", encoding="utf-8", newline="") as handle:
            text = handle.read()
    except OSError:
        return {}
    pairs: dict[str, str] = {}
    for line in text.splitlines():
        match = _ASSIGNMENT_RE.match(line)
        if match is not None and match.group(1) in KNOWN_KEYS:
            pairs[match.group(1)] = _parse_value(match.group(2))
    return pairs


def _canon(value: object) -> str:
    """A canonical string for comparing an effective value with a `.env` text."""
    if isinstance(value, bool):
        return "1" if value else "0"
    text = str(value).strip()
    lowered = text.lower()
    if lowered in _TRUTHY:
        return "1"
    if lowered in {"false", "no", "off"}:
        return "0"
    text = text.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _same(effective: object, raw: str) -> bool:
    """True when an effective value and a `.env` entry mean the same thing."""
    left, right = _canon(effective), _canon(raw)
    if left == right:
        return True
    try:  # `7` and `7.0` are the same setting written two ways
        return float(left) == float(right)
    except ValueError:
        return False


def _classify(key: str, effective: object, dotenv: dict[str, str]) -> str:
    """Where the effective value comes from: ``env``, ``dotenv`` or ``default``.

    Environment wins over `.env` because ``load_dotenv`` never overrides an
    existing variable, and the child inherits the server's environment.
    """
    raw = dotenv.get(key)
    if raw is not None and _same(effective, raw):
        return "dotenv"
    if raw is not None or key in os.environ:
        return "env"
    return "default"


def _jsonable(value: Any) -> Any:
    """A value JSON can carry: paths become strings, nothing is stringified."""
    if isinstance(value, Path):
        return str(value)
    return value


def effective_config() -> dict[str, Any]:
    """Effective settings plus each one's source; no credential is reported."""
    from polyglot_lc.config import get_app_config

    app = get_app_config()
    dotenv = _dotenv_pairs()
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key, attribute in {**EDITABLE_FIELDS, **READONLY_FIELDS}.items():
        effective = getattr(app, attribute)
        values[key] = _jsonable(effective)
        sources[key] = _classify(key, effective, dotenv)
    mock = os.getenv(MOCK_KEY, "").strip().lower() in _TRUTHY
    values[MOCK_KEY] = mock
    sources[MOCK_KEY] = _classify(MOCK_KEY, mock, dotenv)
    return {"values": values, "sources": sources}


def _translator_model() -> str:
    """The configured translator model, or the chunker's own default.

    Planning a file has to work before any credential exists — the UI should be
    able to show how a document will be split even on a fresh checkout — so an
    unconfigured translator degrades to the tokenizer default instead of
    failing. The model actually used is reported, so nothing is hidden.
    """
    from polyglot_lc.config import ConfigError, get_agent_config

    try:
        return get_agent_config("TRANSLATOR").model
    except ConfigError:
        return FALLBACK_MODEL


def plan_file(file_path: Path, *, tgt: str, max_tokens: int | None) -> dict[str, Any]:
    """How a file will be split, without spending a single model request."""
    from polyglot_lc.ingest import load_document
    from polyglot_lc.pipeline.assembler import output_path
    from polyglot_lc.segment.chunker import chunk_document

    if not file_path.is_file():
        raise FileNotFoundError(f"input file not found: {file_path}")
    model = _translator_model()
    doc = load_document(file_path)
    chunks = chunk_document(doc, max_tokens=max_tokens, model=model)
    return {
        "chunks": len(chunks),
        "output_path": str(output_path(file_path, tgt, None)),
        "title": Path(str(doc.metadata.get("filename") or file_path.name)).name,
        "model": model,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="webui-probe",
        description="Report effective settings or a file's chunk plan as JSON.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", action="store_true", help="Print effective settings.")
    group.add_argument("--plan", metavar="FILE", help="Print a file's chunk plan.")
    # --src is accepted for interface symmetry with the CLI; planning depends
    # only on the target language, because the tokenizer does not read the text.
    parser.add_argument("--src", default="auto", help="Source language (informational).")
    parser.add_argument("--tgt", default="vi", help="Target language.")
    parser.add_argument("--max-tokens", type=int, dest="max_tokens", default=None,
                        help="Chunk token budget (default: CHUNK_MAX_TOKENS).")
    args = parser.parse_args(argv)

    try:
        if args.config:
            payload = effective_config()
        else:
            payload = plan_file(Path(args.plan), tgt=args.tgt, max_tokens=args.max_tokens)
    except Exception as exc:  # noqa: BLE001 - the contract is JSON on stdout, a
        # reason on stderr and a non-zero exit; a traceback would break the
        # single-object contract the caller parses.
        print(f"probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    json.dump(payload, sys.stdout, ensure_ascii=False, indent=None)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
