"""Long-context segmenter: token-aware chunks at paragraph/heading boundaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import tiktoken

from polyglot_lc.config import get_app_config
from polyglot_lc.ingest.model import Block, Document

_FALLBACK_ENCODING = "cl100k_base"


def get_encoding(model: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding for *model* (falls back to cl100k_base)."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in *text* under *model*'s tokenizer."""
    return len(get_encoding(model).encode(text))


@dataclass
class Chunk:
    """One translatable unit: contiguous blocks within the token budget."""

    id: str
    text: str
    token_count: int
    section_path: str = ""
    index: int = 0
    block_refs: list[int] = field(default_factory=list)

    def hash(self) -> str:
        return hashlib.sha1(self.text.encode()).hexdigest()[:10]


def _block_text(block: Block) -> str:
    if block.kind == "heading":
        return "#" * max(block.level, 1) + " " + block.text
    if block.kind == "list":
        return "\n".join("- " + i for i in block.meta.get("items", [block.text]))
    return block.text


def _section_for(blocks: list[tuple[int, Block]], upto: int) -> str:
    """Current heading path for the block at position *upto*."""
    headings = [b.text for _, b in blocks[:upto] if b.kind == "heading"]
    return " > ".join(headings[-2:]) if headings else ""


def chunk_document(
    doc: Document,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    model: str = "gpt-4o",
) -> list[Chunk]:
    """Split *doc* into chunks <= max_tokens at paragraph/heading boundaries.

    Never splits inside a block; an oversized block becomes its own chunk
    (protected from endless splitting by the hard truncation guard below).
    """
    app = get_app_config()
    max_tokens = max_tokens or app.chunk_max_tokens
    overlap_tokens = app.chunk_overlap_tokens if overlap_tokens is None else overlap_tokens

    numbered = list(enumerate(doc.blocks))
    texts = [(idx, b, _block_text(b)) for idx, b in numbered]
    enc = get_encoding(model)

    chunks: list[Chunk] = []
    current: list[tuple[int, Block, str]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        body = "\n\n".join(t for _, _, t in current).strip()
        if not body:
            current, current_tokens = [], 0
            return
        tc = len(enc.encode(body))
        # Hard guard: single block over budget -> keep as one chunk (do not
        # split mid-sentence; translator still handles it, verifier warns).
        chunks.append(
            Chunk(
                id=f"chunk-{len(chunks):03d}",
                text=body,
                token_count=tc,
                section_path=_section_for(numbered, current[0][0]),
                index=len(chunks),
                block_refs=[i for i, _, _ in current],
            )
        )
        # Overlap: replay the tail of the finished chunk as the head of the next.
        if overlap_tokens > 0:
            tail: list[tuple[int, Block, str]] = []
            tail_tokens = 0
            for item in reversed(current):
                t = len(enc.encode(item[2]))
                if tail_tokens + t > overlap_tokens and tail:
                    break
                tail.insert(0, item)
                tail_tokens += t
            current, current_tokens = tail, tail_tokens
        else:
            current, current_tokens = [], 0

    for idx, block, text in texts:
        t = len(enc.encode(text))
        if current_tokens + t > max_tokens and current:
            flush()
        current.append((idx, block, text))
        current_tokens += t
    flush()

    # Drop overlap-only duplicates (chunks with no fresh content).
    deduped: list[Chunk] = []
    seen_texts: set[str] = set()
    for c in chunks:
        if c.text in seen_texts:
            continue
        seen_texts.add(c.text)
        deduped.append(c)
    for i, c in enumerate(deduped):
        c.index = i
        c.id = f"chunk-{i:03d}"
    return deduped
