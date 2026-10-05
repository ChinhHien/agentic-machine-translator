"""Shared, stable read-source context for the Translator/Critic/Refiner group.

The three read-source agents all cross-reference the *same* original source
chunk. To avoid re-billing that text on every call and to let the provider's
native prompt/context cache serve it, they all receive an **identical** stable
prefix as the first message of the request:

    role + language pair + strategy + pitfalls + glossary + original chunk

Agent-specific, dynamic content (the draft being criticised, the critic's
feedback, the instruction to translate/refine) goes *after* the prefix, so it
never invalidates the cached block. Planner and Verifier are independent and
do not use this shared prefix.
"""

from __future__ import annotations

import hashlib
from typing import Any

SOURCE_START = "--- SOURCE TEXT START ---"
SOURCE_END = "--- SOURCE TEXT END ---"

#: Bump when the stable prefix layout/wording changes (invalidates old keys).
FINGERPRINT_VERSION = "polyglot-cache-v1"


def build_source_context(
    src_lang: str,
    tgt_lang: str,
    strategy: str = "",
    pitfalls: list[str] | None = None,
    glossary_md: str = "",
    source: str = "",
) -> str:
    """Build the stable read-source prefix shared by Translator/Critic/Refiner.

    Ordering is deliberate and deterministic (strategy → pitfalls → glossary →
    source text) so the byte-identical prefix can be cached across the three
    agent calls for a given chunk.
    """
    parts = [
        f"You are a member of an expert {src_lang}->{tgt_lang} translation team.",
    ]
    if strategy:
        parts.append(f"Global translation strategy:\n{strategy}")
    pitfalls = [p for p in (pitfalls or []) if p]
    if pitfalls:
        parts.append("Pitfalls to avoid:\n" + "\n".join(f"- {p}" for p in pitfalls))
    if glossary_md and glossary_md != "(none)":
        parts.append(f"Glossary (use consistently):\n{glossary_md}")
    parts.append(f"{SOURCE_START}\n{source}\n{SOURCE_END}")
    return "\n\n".join(parts)


def extract_source(messages: list[dict[str, Any]]) -> str:
    """Recover the original source text from a shared-context prompt.

    Used by the translator to seed the translation memory without reparsing
    agent-specific instruction text.
    """
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(b.get("text", "")) if isinstance(b, dict) else str(b)
                for b in content
            )
        text = str(content)
        start = text.find(SOURCE_START)
        end = text.find(SOURCE_END, start + 1)
        if start != -1 and end != -1:
            body = text[start + len(SOURCE_START):end]
            if body.strip():
                return body.strip()
    return ""


def source_context_messages(
    src_lang: str,
    tgt_lang: str,
    strategy: str = "",
    pitfalls: list[str] | None = None,
    glossary_md: str = "",
    source: str = "",
) -> dict[str, str]:
    """Return the shared prefix as its own ``system`` message dict."""
    return {
        "role": "system",
        "content": build_source_context(
            src_lang, tgt_lang, strategy, pitfalls, glossary_md, source
        ),
    }


def stable_prefix_fingerprint(messages: list[dict[str, Any]], model: str = "") -> str:
    """Lightweight debug hash of the byte-identical cacheable prefix.

    Hashes only the components that must be identical for a provider cache hit
    (the shared system message — which already encodes language pair, strategy,
    pitfalls, glossary and the original chunk — plus the model and a version
    tag). The raw source text is never logged: only this digest is.
    """
    prefix = ""
    if messages:
        content = messages[0].get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(b.get("text", "")) if isinstance(b, dict) else str(b)
                for b in content
            )
        prefix = str(content)
    digest = hashlib.sha1()
    for part in (FINGERPRINT_VERSION, model, prefix):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]
