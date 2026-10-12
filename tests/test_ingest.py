"""Tests for ingestion: the Markdown parser, the PDF-only registry and geometry."""

from __future__ import annotations

import pytest

from polyglot_lc.ingest import load_document
from polyglot_lc.ingest.md import parse_markdown, restore_code, shield_code_blocks
from polyglot_lc.ingest.registry import get_loader, register

MD = """# Title

A paragraph with **bold** text.

## Section 2

- item one
- item two

```python
x = 1
y = 2
```

Closing paragraph.
"""


def _minimal_pdf(text: str = "Hello PDF World", y: int = 700) -> bytes:
    """Build a tiny single-page PDF with one text object (no extra deps).

    *y* is the baseline height in PDF coordinates (origin bottom-left). The
    default sits in the top margin band, which the layout extractor treats as a
    running head; pass a smaller value to place text in the page body.
    """
    stream = b"BT /F1 18 Tf 72 " + str(y).encode() + b" Td (" + text.encode() + b") Tj ET"
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        (
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
            b"/Resources<</Font<</F1 5 0 R>>>>>>"
        ),
        b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += str(i).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        out += (f"{off:010d} 00000 n \n").encode()
    out += (
        b"trailer\n<</Size "
        + str(len(objs) + 1).encode()
        + b"/Root 1 0 R>>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


# ------------------------------- Markdown parser ----------------------------


def test_markdown_structure():
    doc = parse_markdown(MD)
    kinds = [b.kind for b in doc.blocks]
    assert kinds.count("heading") == 2
    assert "list" in kinds
    assert doc.blocks[0].kind == "heading"
    assert doc.blocks[0].text == "Title"
    assert doc.blocks[0].level == 1


def test_markdown_code_shielded():
    doc = parse_markdown(MD)
    code_blocks = [b for b in doc.blocks if b.kind == "code"]
    assert len(code_blocks) == 0  # code lives in code_map, not translatable blocks
    code_map = doc.metadata["code_map"]
    assert any("x = 1" in v for v in code_map.values())
    plain = doc.plain_text()
    assert "x = 1" not in plain


def test_markdown_inline_formatting_stripped():
    doc = parse_markdown(MD)
    para = next(b for b in doc.blocks if b.kind == "paragraph")
    assert "bold" in para.text
    assert "**" not in para.text


def test_shield_and_restore_roundtrip():
    text, mapping = shield_code_blocks("before\n```py\nSECRET_PAYLOAD_XYZ\n```\nafter")
    assert "SECRET_PAYLOAD_XYZ" not in text
    restored = restore_code(text, mapping)
    assert "SECRET_PAYLOAD_XYZ" in restored


def test_markdown_table_text_is_kept():
    """Table tokens carry real text: dropping them silently loses documents."""
    doc = parse_markdown(
        "| Term | Meaning |\n| --- | --- |\n| NMT | machine translation |\n",
        source_format="pdf",
    )
    assert doc.blocks, "a markdown table must produce at least one block"
    table = doc.blocks[0]
    assert table.meta.get("table") is True
    # The columns must survive as columns: a flattened row is how a license
    # matrix lost which value belonged to which license.
    assert table.text.startswith("| Term | Meaning |")
    assert "| --- | --- |" in table.text
    text = doc.plain_text()
    assert "NMT" in text
    assert "machine translation" in text


# ---------------------------------- Registry --------------------------------


def test_registry_accepts_pdf_only(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(_minimal_pdf("Registry Body", y=500))
    doc = load_document(p)
    assert doc.metadata["format"] == "pdf"
    assert doc.metadata["filename"] == "doc.pdf"


@pytest.mark.parametrize("name", ["data.md", "data.txt", "data.xyz"])
def test_registry_refuses_everything_but_pdf(tmp_path, name):
    """Input is PDF only: a Markdown or text file is refused, not silently read."""
    f = tmp_path / name
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_document(f)


def test_registry_custom_loader():
    calls = []

    def custom(data: bytes):
        calls.append(data)
        from polyglot_lc.ingest.model import Document

        return Document()

    register(".custom", custom)
    assert get_loader(".custom") is custom


# ------------------------------------ PDF -----------------------------------


def test_pdf_is_extracted_by_typography_and_bound_to_geometry(tmp_path):
    """The one extractor reads the page *and* records where every block was."""
    p = tmp_path / "sample.pdf"
    p.write_bytes(_minimal_pdf("Hello PDF World", y=500))  # body, not top margin
    doc = load_document(p)
    assert doc.metadata["format"] == "pdf"
    assert doc.metadata["converter"] == "layout"
    assert "Hello PDF World" in doc.plain_text()
    # The geometry is what makes the rebuilt PDF possible, so every block must
    # carry a page rectangle for its own text.
    geometry = [b.meta.get("pdf") for b in doc.blocks]
    assert all(geometry), "every block needs its source rectangles"
    assert all(g["rects"] for g in geometry)
    assert all(g["kind"] for g in geometry)


def test_pdf_without_a_text_layer_fails_loudly(tmp_path):
    """No OCR: a page with no text is an error, never a silent empty document."""
    p = tmp_path / "empty.pdf"
    p.write_bytes(_minimal_pdf("", y=500))
    with pytest.raises(ValueError):
        load_document(p)
