"""Tests for ingestion: markdown, txt, registry, code shielding."""

from __future__ import annotations

import pytest

from polyglot_lc.ingest import load_document
from polyglot_lc.ingest.md import load_md, restore_code, shield_code_blocks
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


def test_markdown_structure():
    doc = load_md(MD.encode())
    kinds = [b.kind for b in doc.blocks]
    assert kinds.count("heading") == 2
    assert "list" in kinds
    assert doc.blocks[0].kind == "heading"
    assert doc.blocks[0].text == "Title"
    assert doc.blocks[0].level == 1


def test_markdown_code_shielded():
    doc = load_md(MD.encode())
    code_blocks = [b for b in doc.blocks if b.kind == "code"]
    assert len(code_blocks) == 0  # code lives in code_map, not translatable blocks
    code_map = doc.metadata["code_map"]
    assert any("x = 1" in v for v in code_map.values())
    plain = doc.plain_text()
    assert "x = 1" not in plain


def test_markdown_inline_formatting_stripped():
    doc = load_md(MD.encode())
    para = next(b for b in doc.blocks if b.kind == "paragraph")
    assert "bold" in para.text
    assert "**" not in para.text


def test_shield_and_restore_roundtrip():
    text, mapping = shield_code_blocks("before\n```py\nSECRET_PAYLOAD_XYZ\n```\nafter")
    assert "SECRET_PAYLOAD_XYZ" not in text
    restored = restore_code(text, mapping)
    assert "SECRET_PAYLOAD_XYZ" in restored


def test_txt_loader_splits_paragraphs():
    from polyglot_lc.ingest.txt import load_txt

    doc = load_txt("para one\n\npara two\n\npara three".encode())
    assert len(doc.blocks) == 3
    assert all(b.kind == "paragraph" for b in doc.blocks)


def test_txt_nfc_normalization():
    from polyglot_lc.ingest.txt import load_txt

    decomposed = "cafe\u0301\n\nnho\u0301"  # e/o + combining acute
    doc = load_txt(decomposed.encode())
    assert doc.blocks[0].text == "café"
    assert doc.blocks[1].text == "nhó"


def test_registry_dispatch(sample_md_file):
    doc = load_document(sample_md_file)
    assert doc.metadata["format"] == "markdown"


def test_registry_unsupported_extension(tmp_path):
    f = tmp_path / "data.xyz"
    f.write_text("x")
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


# ------------------------------ PDF (MarkItDown) ----------------------------
def _minimal_pdf(text: str = "Hello PDF World") -> bytes:
    """Build a tiny single-page PDF with one text object (no extra deps)."""
    stream = b"BT /F1 18 Tf 72 700 Td (" + text.encode() + b") Tj ET"
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length " + str(len(stream)).encode() + b">>stream\n"
        + stream + b"\nendstream",
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
        out += ("%010d 00000 n \n" % off).encode()
    out += (
        b"trailer\n<</Size "
        + str(len(objs) + 1).encode()
        + b"/Root 1 0 R>>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


def test_pdf_markdown_is_parsed_into_blocks(monkeypatch):
    """The markdown MarkItDown returns must go through the markdown parser."""
    from polyglot_lc.ingest import pdf as pdf_mod

    markdown = (
        "# Report Title\n\nIntro paragraph.\n\n"
        "- one\n- two\n\n```py\nx = 1\n```\n"
    )
    monkeypatch.setattr(pdf_mod, "convert_pdf_to_markdown", lambda data: markdown)

    doc = pdf_mod.load_pdf(b"%PDF-1.4 fake")
    assert doc.metadata["format"] == "pdf"
    assert doc.metadata["converter"] == "markitdown"
    kinds = [b.kind for b in doc.blocks]
    assert kinds[0] == "heading"
    assert doc.blocks[0].text == "Report Title"
    assert "list" in kinds
    # code shielded, not left as translatable text
    assert "x = 1" not in doc.plain_text()
    assert any("x = 1" in v for v in doc.metadata["code_map"].values())


def test_pdf_real_conversion_via_markitdown(tmp_path):
    from polyglot_lc.ingest import pdf as pdf_mod

    assert pdf_mod._get_markitdown() is not None, "markitdown must be installed"
    p = tmp_path / "sample.pdf"
    p.write_bytes(_minimal_pdf("Hello PDF World"))
    doc = load_document(p)
    assert doc.metadata["format"] == "pdf"
    assert doc.metadata["converter"] == "markitdown"
    assert "Hello PDF World" in doc.plain_text()


def test_pdf_falls_back_when_markitdown_fails(monkeypatch, tmp_path):
    from polyglot_lc.ingest import pdf as pdf_mod

    def boom(data):
        raise RuntimeError("markitdown exploded")

    monkeypatch.setattr(pdf_mod, "convert_pdf_to_markdown", boom)
    p = tmp_path / "sample.pdf"
    p.write_bytes(_minimal_pdf("Fallback Text"))
    doc = pdf_mod.load_pdf(p.read_bytes())
    assert doc.metadata["converter"] == "legacy"
    assert "Fallback Text" in doc.plain_text()


def test_pdf_falls_back_when_markitdown_empty(monkeypatch):
    from polyglot_lc.ingest import pdf as pdf_mod

    monkeypatch.setattr(pdf_mod, "convert_pdf_to_markdown", lambda data: "   ")
    doc = pdf_mod.load_pdf(_minimal_pdf("Empty Conversions"))
    assert doc.metadata["converter"] == "legacy"
    assert "Empty Conversions" in doc.plain_text()
