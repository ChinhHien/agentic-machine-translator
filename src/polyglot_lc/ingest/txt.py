"""Plain-text loader."""

from __future__ import annotations

import unicodedata

from polyglot_lc.ingest.model import Block, Document


def load_txt(data: bytes) -> Document:
    """Load UTF-8 (or latin-1 fallback) text, NFC-normalized, split by paragraphs."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    text = unicodedata.normalize("NFC", text)
    blocks: list[Block] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            blocks.append(Block(kind="paragraph", text=para))
    return Document(
        blocks=blocks,
        lang_hint="auto",
        metadata={"format": "txt", "encoding": "utf-8"},
    )
