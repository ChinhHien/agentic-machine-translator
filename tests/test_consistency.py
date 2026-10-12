"""Tests for the cross-chunk consistency pass."""

from __future__ import annotations

from types import SimpleNamespace

from polyglot_lc.pipeline.consistency import (
    ConsistencyChecker,
    contains_term,
    find_inconsistencies,
)


def _chunk(cid: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(id=cid, text=text)


class _StubEditor:
    """Consistency editor stand-in: records calls, rewrites on demand."""

    def __init__(self, rewrite=None) -> None:
        self.calls = 0
        self.rewrite = rewrite

    def patch(
        self,
        *,
        source,
        translation,
        fixes,
        src_lang,
        tgt_lang,
        strategy="",
        glossary_md="",
        section_note="",
    ):
        self.calls += 1
        if self.rewrite is None:
            return translation
        return self.rewrite(translation, fixes)


def test_contains_term_is_case_and_space_insensitive():
    assert contains_term("Dịch máy neural is used", "dịch  máy   neural")
    assert not contains_term("Dịch máy", "transformer")


def test_divergent_term_use_is_reported():
    chunks = [_chunk("chunk-000", "NMT is fast."), _chunk("chunk-001", "NMT is slow.")]
    translations = {"chunk-000": "Dịch máy neural rất nhanh.", "chunk-001": "NMT rất chậm."}
    issues = find_inconsistencies(chunks, translations, {"NMT": "Dịch máy neural"})
    assert len(issues) == 1
    issue = issues[0]
    assert issue.missing_chunks == ("chunk-001",)
    assert issue.compliant_chunks == ("chunk-000",)
    assert issue.expected == "Dịch máy neural"


def test_uniform_miss_is_not_an_inconsistency():
    """Nobody used the planned term: there is nothing to make the chunks agree on."""
    chunks = [_chunk("chunk-000", "NMT is fast."), _chunk("chunk-001", "NMT is slow.")]
    translations = {"chunk-000": "NMT rất nhanh.", "chunk-001": "NMT rất chậm."}
    assert find_inconsistencies(chunks, translations, {"NMT": "Dịch máy neural"}) == []


def test_terms_absent_from_the_source_are_ignored():
    chunks = [_chunk("chunk-000", "Nothing relevant.")]
    assert (
        find_inconsistencies(chunks, {"chunk-000": "translation"}, {"NMT": "Dịch máy neural"}) == []
    )


def test_checker_patches_only_the_drifting_chunk():
    chunks = [_chunk("chunk-000", "NMT is fast."), _chunk("chunk-001", "NMT is slow.")]
    translations = {"chunk-000": "Dịch máy neural rất nhanh.", "chunk-001": "NMT rất chậm."}

    def rewrite(translation, fixes):
        return translation.replace("NMT", "Dịch máy neural")

    editor = _StubEditor(rewrite)
    checker = ConsistencyChecker(editor, max_patches=5)
    patched, report = checker.run(
        chunks,
        translations,
        {"NMT": "Dịch máy neural"},
        context={"src_lang": "en", "tgt_lang": "vi"},
    )
    assert editor.calls == 1
    assert report.inconsistencies == 1
    assert report.patched_chunks == 1
    assert patched["chunk-001"] == "Dịch máy neural rất chậm."
    assert patched["chunk-000"] == translations["chunk-000"]  # untouched


def test_a_patch_that_does_not_fix_anything_is_rejected():
    chunks = [_chunk("chunk-000", "NMT is fast."), _chunk("chunk-001", "NMT is slow.")]
    translations = {"chunk-000": "Dịch máy neural rất nhanh.", "chunk-001": "NMT rất chậm."}
    editor = _StubEditor(lambda translation, fixes: "hoàn toàn không liên quan")

    patched, report = ConsistencyChecker(editor, max_patches=5).run(
        chunks, translations, {"NMT": "Dịch máy neural"}
    )
    assert report.patched_chunks == 0
    assert report.unresolved == ["chunk-001"]
    # The original translation is never lost to a bad edit.
    assert patched["chunk-001"] == "NMT rất chậm."


def test_patch_budget_is_enforced():
    chunks = [
        _chunk("chunk-000", "NMT here."),
        _chunk("chunk-001", "NMT there."),
        _chunk("chunk-002", "NMT elsewhere."),
    ]
    translations = {
        "chunk-000": "Dịch máy neural ở đây.",
        "chunk-001": "NMT ở kia.",
        "chunk-002": "NMT ở chỗ khác.",
    }
    editor = _StubEditor(lambda translation, fixes: translation.replace("NMT", "Dịch máy neural"))
    patched, report = ConsistencyChecker(editor, max_patches=1).run(
        chunks, translations, {"NMT": "Dịch máy neural"}
    )
    assert editor.calls == 1
    assert report.patched_chunks == 1
    assert report.skipped_chunks == 1
    assert patched["chunk-002"] == "NMT ở chỗ khác."  # budget exhausted


def test_disabled_checker_does_not_call_the_model():
    chunks = [_chunk("chunk-000", "NMT is fast."), _chunk("chunk-001", "NMT is slow.")]
    translations = {"chunk-000": "Dịch máy neural nhanh.", "chunk-001": "NMT chậm."}
    editor = _StubEditor(lambda translation, fixes: "Dịch máy neural")
    patched, report = ConsistencyChecker(editor, enabled=False).run(
        chunks, translations, {"NMT": "Dịch máy neural"}
    )
    assert editor.calls == 0
    assert report.inconsistencies == 0
    assert patched == translations
