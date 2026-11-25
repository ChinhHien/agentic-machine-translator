"""Tests for the opt-in block-level AI repair layer.

The extractor is never replaced here: these tests pin the guard rails that let
an AI correction be kept only when it cannot change structure, geometry or
content-bearing tokens.
"""

from __future__ import annotations

from types import SimpleNamespace

from polyglot_lc.ingest.ai_review import (
    clean_reply,
    needs_review,
    review_document,
    review_with_llm,
    validate_correction,
)
from polyglot_lc.ingest.model import Block, Document


def _block(text: str, **meta) -> Block:
    return Block(kind="paragraph", text=text, meta={"pdf": [(1, 0, 0, 100, 100)], **meta})


def _doc(*blocks: Block) -> Document:
    return Document(blocks=list(blocks))


def test_clean_text_never_calls_the_reviewer():
    doc = _doc(_block("A clean paragraph with 10 kg and a name."))
    calls: list[object] = []

    report = review_document(doc, lambda request: calls.append(request) or "changed")

    assert report.candidates == 0
    assert report.reviewed == 0
    assert calls == []


def test_a_suspicious_block_is_repaired_without_touching_geometry():
    block = _block("The value is 10  kg in the table.")
    before_meta = block.meta.copy()
    doc = _doc(block)
    report = review_document(
        doc,
        lambda request: request.text.replace("  ", " "),
    )

    assert report.changed == 1
    assert block.text == "The value is 10 kg in the table."
    assert block.meta == before_meta
    assert len(doc.blocks) == 1


def test_a_repaired_list_keeps_its_structured_items_in_step():
    block = Block(
        kind="list",
        text="one  item\ntwo items",
        meta={"items": ["one  item", "two items"]},
    )
    report = review_document(_doc(block), lambda request: "one item\ntwo items")

    assert report.changed == 1
    assert block.meta["items"] == ["one item", "two items"]


def test_a_cid_artifact_is_detected_and_repaired():
    block = _block("The´(cid:160)(cid:160)(cid:160)quick´(cid:160)(cid:160)(cid:160)fox.")
    assert needs_review(block)
    report = review_document(_doc(block), lambda request: "The quick fox.")

    assert report.changed == 1
    assert block.text == "The quick fox."


def test_a_lost_code_placeholder_is_rejected():
    block = _block("Keep ⟦CODE_1⟧  here.")
    report = review_document(_doc(block), lambda request: "Keep here.")

    assert report.rejected == 1
    assert block.text == "Keep ⟦CODE_1⟧  here."


def test_a_changed_number_is_rejected():
    block = _block("Version 1.2  of the tool.")
    report = review_document(_doc(block), lambda request: "Version 1.3 of the tool.")

    assert report.rejected == 1
    assert "1.2" in block.text


def test_a_rewrite_that_changes_the_meaning_is_rejected():
    block = _block("The quick  brown fox jumps.")
    report = review_document(
        _doc(block),
        lambda request: "An entirely different sentence about a fox.",
    )

    assert report.rejected == 1
    assert block.text == "The quick  brown fox jumps."


def test_a_rewritten_table_cell_is_rejected_even_with_the_same_shape():
    block = _block("| Name  | Value |\n| --- | --- |\n| Alpha | 10 |", table=True)
    report = review_document(
        _doc(block),
        lambda request: "| Name | Value |\n| --- | --- |\n| Omega | 10 |",
    )

    assert report.rejected == 1
    assert "Alpha" in block.text


def test_a_changed_table_shape_is_rejected():
    block = _block(
        "| A  | B |\n| --- | --- |\n| 1 | 2 |",
        table=True,
    )
    report = review_document(_doc(block), lambda request: "| A | B |\n| 1 | 2 |")

    assert report.rejected == 1
    assert block.text.count("\n") == 2


def test_a_review_failure_leaves_the_extracted_text_alone():
    block = _block("BrokenÂ·text  here.")

    def explode(request):
        raise TimeoutError("provider timed out")

    report = review_document(_doc(block), explode)

    assert report.failed == 1
    assert block.text == "BrokenÂ·text  here."


def test_a_block_longer_than_the_limit_is_skipped_not_truncated():
    block = _block("A  " + "very long " * 20)
    calls: list[object] = []
    report = review_document(
        _doc(block),
        lambda request: calls.append(request) or "short",
        max_chars=20,
    )

    assert report.skipped == 1
    assert calls == []
    assert block.text.startswith("A  very long")


def test_reply_cleaning_removes_a_fence_and_honours_no_change():
    assert clean_reply("```text\nFixed  text\n```") == "Fixed  text"
    assert clean_reply("NO_CHANGE") is None
    assert clean_reply("   ") is None


def test_llm_review_uses_block_scope_and_a_bounded_reply():
    block = _block("Broken  text with 10 kg.")
    seen: dict[str, object] = {}

    class FakeClient:
        def chat(self, messages, **kwargs):
            seen["messages"] = messages
            seen["kwargs"] = kwargs
            return SimpleNamespace(text="Broken text with 10 kg.")

    report = review_with_llm(_doc(block), FakeClient(), max_blocks=1, max_chars=100)

    assert report.changed == 1
    assert block.text == "Broken text with 10 kg."
    assert "temperature" not in seen["kwargs"]
    assert seen["kwargs"]["max_tokens"] >= 1024
    prompt = "\n".join(message["content"] for message in seen["messages"])
    assert "Broken  text with 10 kg." in prompt
    assert "source text, not a translation" in prompt


def test_validator_rejects_candidate_that_is_only_whitespace():
    assert validate_correction(_block("Broken  text."), "   ") == "empty correction"
