"""Markdown loader: parse with markdown-it-py, preserve structure, shield code."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from markdown_it import MarkdownIt

from polyglot_lc.ingest.model import Block, Document

_CODE_PH = "⟦CODE_{n}⟧"
_PH_RE = re.compile(r"⟦CODE_(\d+)⟧")


def _inline_text(children: list[Any]) -> str:
    """Flatten inline tokens to plain text; drop emphasis markers, keep content."""
    out: list[str] = []
    for tok in children:
        if tok.type == "text":
            out.append(tok.content)
        elif tok.type == "code_inline":
            out.append(tok.content)
        elif tok.type in {"softbreak", "hardbreak"}:
            out.append(" ")
        elif tok.type == "image":
            alt = tok.content or tok.attrs.get("alt", "")
            if alt:
                out.append(str(alt))
    return "".join(out).strip()


def shield_code_blocks(md_text: str) -> tuple[str, dict[str, str]]:
    """Replace fenced and indented code blocks with placeholders.

    Returns (new_text, mapping placeholder -> original code block).
    """
    mapping: dict[str, str] = {}
    counter = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        ph = _CODE_PH.format(n=counter)
        mapping[ph] = match.group(0)
        return ph

    fenced = re.compile(r"```[^\n]*\n.*?\n```", re.DOTALL)
    text = fenced.sub(_sub, md_text)

    # Indented code blocks (4 spaces or tab).
    # Use the same _sub() so the counter remains shared with fenced blocks.
    indented = re.compile(r"(?m)^(?:    |\t).+\n(?:    |\t.+\n)*")
    text = indented.sub(_sub, text)

    return text, mapping


def parse_markdown(
    raw_text: str,
    *,
    source_format: str = "markdown",
    extra_metadata: dict[str, Any] | None = None,
) -> Document:
    """Parse a markdown string into structural blocks with code shielded.

    Shared entry point for both the ``.md`` loader and the PDF loader (which
    converts PDFs to markdown with MarkItDown first). *source_format* is
    recorded in the document metadata so downstream steps know the original
    container.
    """
    raw = unicodedata.normalize("NFC", raw_text)
    text, code_map = shield_code_blocks(raw)

    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(text)

    blocks: list[Block] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            level = int(tok.tag[1]) if len(tok.tag) == 2 else 1
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            heading_text = _inline_text(inline.children or []) if inline else ""
            blocks.append(Block(kind="heading", text=heading_text, level=level))
            i += 3  # heading_open, inline, heading_close
            continue
        if tok.type == "paragraph_open":
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            body = _inline_text(inline.children or []) if inline else ""
            if body:
                blocks.append(Block(kind="paragraph", text=body))
            i += 3
            continue
        if tok.type == "bullet_list_open" or tok.type == "ordered_list_open":
            items: list[str] = []
            depth = 0
            j = i
            while j < len(tokens):
                t = tokens[j]
                if t.type.endswith("_list_open"):
                    depth += 1
                elif t.type.endswith("_list_close"):
                    depth -= 1
                    if depth == 0:
                        break
                elif t.type == "inline" and depth == 1:
                    items.append(_inline_text(t.children or []))
                j += 1
            blocks.append(Block(kind="list", text="\n".join(items),
                                meta={"items": items}))
            i = j + 1
            continue
        if tok.type == "fence" or tok.type == "code_block":
            blocks.append(Block(kind="code", text=tok.content))
            i += 1
            continue
        if tok.type == "blockquote_open":
            inner: list[str] = []
            depth = 0
            j = i
            while j < len(tokens):
                t = tokens[j]
                if t.type == "blockquote_open":
                    depth += 1
                elif t.type == "blockquote_close":
                    depth -= 1
                    if depth == 0:
                        break
                elif t.type == "inline" and depth == 1:
                    inner.append(_inline_text(t.children or []))
                j += 1
            blocks.append(Block(kind="quote", text="\n".join(inner)))
            i = j + 1
            continue
        i += 1

    metadata: dict[str, Any] = {"format": source_format, "code_map": code_map}
    if extra_metadata:
        metadata.update(extra_metadata)
    return Document(blocks=blocks, lang_hint="auto", metadata=metadata)


def load_md(data: bytes) -> Document:
    """Parse markdown bytes into structural blocks with code shielded."""
    raw = data.decode("utf-8", errors="replace")
    return parse_markdown(raw, source_format="markdown")


def restore_code(translated_text: str, code_map: dict[str, str]) -> str:
    """Put shielded code blocks back after translation."""
    restored = translated_text
    for ph, code in code_map.items():
        restored = restored.replace(ph, code)
    return restored


def code_placeholders(text: str) -> list[str]:
    """Return placeholders present in *text* (used by tests and verifier)."""
    return _PH_RE.findall(text)
