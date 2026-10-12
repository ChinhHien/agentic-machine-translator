"""Tests for deterministic repair of dropped code placeholders."""

from __future__ import annotations

from polyglot_lc.pipeline.placeholders import repair_code_placeholders

SOURCE = "Alpha one.\n\n⟦CODE_1⟧\n\nBeta two.\n\n⟦CODE_2⟧\n\nGamma three."


def test_no_repair_when_the_model_keeps_every_marker():
    translation = "Đoạn một.\n\n⟦CODE_1⟧\n\nĐoạn hai.\n\n⟦CODE_2⟧\n\nĐoạn ba."
    text, repaired = repair_code_placeholders(SOURCE, translation)
    assert text == translation
    assert repaired == []


def test_dropped_marker_is_reinserted_at_its_source_offset():
    translation = "Đoạn một.\n\nĐoạn hai.\n\nĐoạn ba."
    text, repaired = repair_code_placeholders(SOURCE, translation)
    assert repaired == ["⟦CODE_1⟧", "⟦CODE_2⟧"]
    assert text == ("Đoạn một.\n\n⟦CODE_1⟧\n\nĐoạn hai.\n\n⟦CODE_2⟧\n\nĐoạn ba.")


def test_only_the_missing_marker_is_added():
    translation = "Đoạn một.\n\n⟦CODE_1⟧\n\nĐoạn hai.\n\nĐoạn ba."
    text, repaired = repair_code_placeholders(SOURCE, translation)
    assert repaired == ["⟦CODE_2⟧"]
    assert text == "Đoạn một.\n\n⟦CODE_1⟧\n\nĐoạn hai.\n\n⟦CODE_2⟧\n\nĐoạn ba."


def test_marker_is_kept_even_when_it_points_past_the_end():
    """A short reply must not silently delete code: keep it rather than drop it."""
    text, repaired = repair_code_placeholders(SOURCE, "Một đoạn duy nhất.")
    assert sorted(repaired) == ["⟦CODE_1⟧", "⟦CODE_2⟧"]
    assert "⟦CODE_1⟧" in text and "⟦CODE_2⟧" in text


def test_source_without_markers_is_untouched():
    text, repaired = repair_code_placeholders("Just prose.", "Chỉ văn xuôi.")
    assert text == "Chỉ văn xuôi."
    assert repaired == []


def test_empty_translation_is_returned_unchanged():
    text, repaired = repair_code_placeholders(SOURCE, "")
    assert text == ""
    assert repaired == []
