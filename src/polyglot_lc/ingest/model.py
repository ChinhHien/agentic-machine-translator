"""Document model shared by ingestion and the pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Block(BaseModel):
    """One structural unit of a document (paragraph, heading, code, list...)."""

    kind: str = "paragraph"  # heading | paragraph | list | code | quote
    text: str
    level: int = 0  # heading level (1..6); 0 for others
    meta: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Normalized, NFC text plus structural blocks and metadata."""

    blocks: list[Block] = Field(default_factory=list)
    lang_hint: str = "auto"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def plain_text(self) -> str:
        """Concatenate non-code blocks with blank lines (headings prefixed)."""
        parts: list[str] = []
        for b in self.blocks:
            if b.kind == "heading":
                parts.append("#" * max(b.level, 1) + " " + b.text)
            elif b.kind == "list":
                parts.extend("- " + item for item in b.meta.get("items", [b.text]))
            else:
                parts.append(b.text)
        return "\n\n".join(parts)
