"""Long-context segmenter: token-aware chunks at paragraph/heading boundaries.

The chunk budget is ``CHUNK_MAX_TOKENS`` and nothing else, and it is counted with
one pinned tokenizer (:data:`CHUNK_ENCODING`). That is deliberate: the split is
part of the checkpoint identity, so it has to be a property of the document and
the budget, not of whichever model happens to be configured. A budget derived
from a model's context window was removed for the same reason — it needs a
trustworthy per-model window (this project has none), and it would have moved
every boundary, and every checkpoint with it, whenever the model changed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import tiktoken

from polyglot_lc.config import get_app_config
from polyglot_lc.ingest.model import Block, Document

_FALLBACK_ENCODING = "cl100k_base"

#: The tokenizer every chunk boundary is counted with, whatever model the run
#: translates with. tiktoken only approximates a non-OpenAI tokenizer, so this
#: count was never the model's own: taking it from the model name split the same
#: document into 74, 76 or 78 chunks depending on which name was configured, and
#: moving a boundary also moves the chunk hashes that checkpoints are keyed by.
#: Pinned, the split is a property of the document and the budget alone.
CHUNK_ENCODING = _FALLBACK_ENCODING

def get_encoding(model: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding for *model* (falls back to cl100k_base)."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


def chunk_encoding() -> tiktoken.Encoding:
    """The encoding chunk boundaries are counted with (see :data:`CHUNK_ENCODING`).

    Deliberately not the translated model's tokenizer: it is a proxy either way,
    and letting the model name choose it made the split — and the checkpoint
    identity that rides on it — change when only the model was switched.
    """
    return tiktoken.get_encoding(CHUNK_ENCODING)


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
    #: What kind of prose this chunk carries. The verifier treats a reference
    #: list differently from body text: article titles and chemical names may
    #: stay in the source language there, while the same English in a body
    #: paragraph is a defect. The chunker decides once, from the source blocks,
    #: and the profile travels with the chunk.
    profile: str = "body"

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


def _chunk_profile(blocks: list[tuple[int, Block, str]], section_path: str) -> str:
    """Classify a chunk for profile-aware translation and verification."""
    headings = " ".join(block.text for _, block, _ in blocks if block.kind == "heading")
    path = (section_path + " " + headings).casefold()
    if any(
        word in path for word in ("reference", "bibliograph", "tham khảo", "tài liệu tham khảo")
    ):
        return "references"
    if any(word in path for word in ("glossary", "thuật ngữ", "định nghĩa")):
        return "glossary"
    kinds = [block.kind for _, block, _ in blocks]
    if kinds and kinds.count("code") * 2 >= len(kinds):
        return "code"
    if kinds and kinds.count("table") * 2 >= len(kinds):
        return "table"
    return "body"


def chunk_document(
    doc: Document,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    model: str = "gpt-4o",
    *,
    base_url: str = "",
) -> list[Chunk]:
    """Split *doc* into chunks <= max_tokens at paragraph/heading boundaries.

    Never splits inside a block; an oversized block becomes its own chunk
    (protected from endless splitting by the hard truncation guard below).

    The split depends only on the configured budget and the pinned tokenizer
    (:func:`chunk_encoding`), never on *model*: chunk hashes are part of the
    checkpoint identity, so boundaries must not move when only the model does.
    *model* and *base_url* stay in the signature for call-site compatibility.
    """
    app = get_app_config()
    max_tokens = max_tokens or app.chunk_max_tokens
    overlap_tokens = app.chunk_overlap_tokens if overlap_tokens is None else overlap_tokens

    numbered = list(enumerate(doc.blocks))
    texts = [(idx, b, _block_text(b)) for idx, b in numbered]
    enc = chunk_encoding()
    #: Token counts by block index. A block's text never changes during the
    #: split, but the body flush and the overlap replay would otherwise encode
    #: it again — two extra passes over the whole document, and the overlap
    #: loop re-encodes the tail of every chunk.
    block_tokens = {idx: len(enc.encode(text)) for idx, _, text in texts}

    chunks: list[Chunk] = []
    current: list[tuple[int, Block, str]] = []
    current_tokens = 0
    #: True while *current* holds nothing but the overlap replayed from the
    #: previous chunk: such a tail can be dropped instead of flushed as a chunk
    #: of its own.
    tail_only = False

    def flush() -> None:
        nonlocal current, current_tokens, tail_only
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
                profile=_chunk_profile(current, _section_for(numbered, current[0][0])),
            )
        )
        # Overlap: replay the tail of the finished chunk as the head of the next.
        if overlap_tokens > 0:
            tail: list[tuple[int, Block, str]] = []
            tail_tokens = 0
            for item in reversed(current):
                t = block_tokens[item[0]]
                if t > overlap_tokens:
                    # A block bigger than the whole overlap window is not
                    # context: replaying it carries its full size into the next
                    # chunk (one 1 383-token block replayed as written produced
                    # a 1 588-token chunk against a 700-token budget).
                    break
                if tail_tokens + t > overlap_tokens and tail:
                    break
                tail.insert(0, item)
                tail_tokens += t
            current, current_tokens = tail, tail_tokens
        else:
            current, current_tokens = [], 0
        tail_only = bool(current)

    for idx, block, text in texts:
        t = block_tokens[idx]
        if current_tokens + t > max_tokens and current:
            if not tail_only:
                flush()  # the chunk that just filled up keeps its own content
            if current_tokens + t > max_tokens and current:
                # What is left is a replayed tail — or the flush above produced
                # one — that cannot share this block. The overlap is context,
                # not content: flushing it alone would translate it a second
                # time, and keeping it here is an over-budget chunk.
                current, current_tokens = [], 0
        current.append((idx, block, text))
        current_tokens += t
        tail_only = False
    flush()

    # Drop overlap-only chunks — those whose blocks an earlier chunk already
    # carries. Comparing *text* instead, as this used to, also dropped a chunk
    # holding a different block that happens to read the same (a repeated
    # heading, the same exercise twice): its content disappeared with it.
    deduped: list[Chunk] = []
    covered: set[int] = set()
    for c in chunks:
        if c.block_refs and all(ref in covered for ref in c.block_refs):
            continue
        deduped.append(c)
        covered.update(c.block_refs)
    for i, c in enumerate(deduped):
        c.index = i
        c.id = f"chunk-{i:03d}"
    return deduped
