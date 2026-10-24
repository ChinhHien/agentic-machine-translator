"""Tests for the cross-chunk consistency pass."""

from __future__ import annotations

from types import SimpleNamespace

from polyglot_lc.pipeline.consistency import (
    ConsistencyChecker,
    ConsistencyReport,
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
    # …and the refusal reaches the run's tally, not just the log.
    assert report.as_run_fields()["consistency_rejected"] == 1


def test_a_patch_that_loses_the_bulk_of_the_translation_is_rejected():
    """A terminology edit is never an abridgement.

    The editor's reply still carries the wanted term even when the provider
    cut it short, so the term check alone let a ~2500-character fragment
    replace a complete translation — the way five chunks of the real
    document lost their tails in one run.
    """
    chunks = [
        _chunk("chunk-000", "NMT is fast."),
        _chunk("chunk-001", "NMT is discussed everywhere else here."),
    ]
    translation = "NMT rất nhanh và được nhắc ở khắp mọi nơi trong đoạn này. " * 8
    translations = {
        "chunk-000": "Dịch máy neural rất nhanh.",  # proof the term is achievable
        "chunk-001": translation,
    }
    fragment = "Dịch máy neural rất nhanh."  # term fixed, but almost everything else lost
    editor = _StubEditor(lambda translation, fixes: fragment)

    patched, report = ConsistencyChecker(editor, max_patches=5).run(
        chunks, translations, {"NMT": "Dịch máy neural"}
    )
    assert editor.calls == 1
    assert report.patched_chunks == 0
    assert report.unresolved == ["chunk-001"]
    # The full translation survives; only the fragment is refused.
    assert patched["chunk-001"] == translation


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
    assert report.as_run_fields()["consistency_skipped"] == 1


def test_the_report_counts_every_outcome_of_the_pass():
    """Bốn con số: lệch, đã vá, bỏ sót, bị từ chối — không chỉ hai như trước."""
    report = ConsistencyReport(
        checked_terms=6, inconsistencies=3, patched_chunks=2, skipped_chunks=4
    )
    report.unresolved.extend(["chunk-004", "chunk-009"])

    stats = report.as_run_fields()
    assert stats["consistency_inconsistencies"] == 3
    assert stats["consistency_patches"] == 2
    assert stats["consistency_skipped"] == 4
    assert stats["consistency_rejected"] == 2


def test_patch_budget_grows_with_the_number_of_drifting_chunks():
    """Trần cố định 5 là quá nhỏ với tài liệu dài: 9 chunk lệch thì được vá 3.

    Một con số không hợp cho cả bản ghi nhớ 3 chunk và sách hướng dẫn 300 chunk.
    Mỗi bản vá là một request trên một chunk, nên ngân sách đi theo số chunk thực
    sự lệch — vẫn giữ sàn ``max_patches`` cho tài liệu nhỏ.
    """
    chunks = [_chunk(f"chunk-{i:03d}", f"NMT {i}.") for i in range(10)]
    translations = {c.id: f"NMT {i}." for i, c in enumerate(chunks)}
    translations["chunk-000"] = "Dịch máy neural 0."  # proof the term is achievable
    editor = _StubEditor(lambda translation, fixes: translation.replace("NMT", "Dịch máy neural"))

    patched, report = ConsistencyChecker(editor, max_patches=1).run(
        chunks, translations, {"NMT": "Dịch máy neural"}
    )
    assert report.inconsistencies == 1
    assert editor.calls == 3  # max(1, ceil(0.3 * 9 chunk lệch))
    assert report.patched_chunks == 3
    assert report.skipped_chunks == 6
    assert patched["chunk-003"] == "Dịch máy neural 3."  # hoà số thuật ngữ: theo thứ tự trang
    assert patched["chunk-005"] == "NMT 5."  # hết ngân sách
    # Ngân sách là chi phí của pass (mỗi bản vá = một request), nên nó phải
    # hiện ra trước khi ai đó đi tìm trong log tại sao lượt này tốn 3 request.
    assert report.patch_budget == 3
    assert report.as_run_fields()["consistency_patch_budget"] == 3


def test_the_pass_stops_after_two_patches_in_a_row_are_refused():
    """Hai lần liền bị từ chối là editor đang phớt lờ chỉ dẫn, không phải may rủi.

    Đốt hết ngân sách vào cùng một câu trả lời chỉ tốn thêm request, mỗi request
    lại gửi cả bản dịch của chunk. Số chunk không được vá vẫn phải được báo là
    bỏ sót: pass dừng, chỗ lệch vẫn còn.
    """
    chunks = [_chunk(f"chunk-{i:03d}", f"NMT {i}.") for i in range(6)]
    translations = {c.id: f"NMT {i}." for i, c in enumerate(chunks)}
    translations["chunk-000"] = "Dịch máy neural 0."  # proof the term is achievable
    editor = _StubEditor(lambda translation, fixes: translation)  # ignoring the fix

    _patched, report = ConsistencyChecker(editor, max_patches=5).run(
        chunks, translations, {"NMT": "Dịch máy neural"}
    )

    assert editor.calls == 2  # not 5
    assert report.unresolved == ["chunk-001", "chunk-002"]
    assert report.skipped_chunks == 3
    assert report.patched_chunks == 0
    assert report.stopped_early is True
    assert report.patch_budget == 5


def test_the_chunk_with_the_most_drifting_terms_is_patched_first():
    """Cùng một request, chunk lệch hai thuật ngữ đáng vá trước chunk lệch một."""
    chunks = [
        _chunk("chunk-000", "NMT and LLM."),
        _chunk("chunk-001", "NMT only."),
        _chunk("chunk-002", "LLM and NMT."),
    ]
    translations = {
        "chunk-000": "Dịch máy neural và mô hình ngôn ngữ lớn.",
        "chunk-001": "NMT only.",
        "chunk-002": "LLM and NMT.",
    }
    seen: list[str] = []

    def rewrite(translation, fixes):
        seen.append(translation)
        return translation.replace("NMT", "Dịch máy neural").replace("LLM", "mô hình ngôn ngữ lớn")

    patched, report = ConsistencyChecker(_StubEditor(rewrite), max_patches=1).run(
        chunks,
        translations,
        {"NMT": "Dịch máy neural", "LLM": "mô hình ngôn ngữ lớn"},
    )
    assert seen == ["LLM and NMT."], "chunk lệch nhiều thuật ngữ phải được vá trước"
    assert report.patched_chunks == 1
    assert patched["chunk-002"] == "mô hình ngôn ngữ lớn and Dịch máy neural."
    assert patched["chunk-001"] == "NMT only."  # hết ngân sách: giữ nguyên


def test_zero_patches_still_turns_the_pass_off():
    """``0`` vẫn là \"tắt vá\", không phải \"vá 30% số chunk lệch\"."""
    chunks = [_chunk("chunk-000", "NMT here."), _chunk("chunk-001", "NMT there.")]
    translations = {"chunk-000": "Dịch máy neural ở đây.", "chunk-001": "NMT ở kia."}
    editor = _StubEditor(lambda translation, fixes: translation.replace("NMT", "Dịch máy neural"))

    patched, report = ConsistencyChecker(editor, max_patches=0).run(
        chunks, translations, {"NMT": "Dịch máy neural"}
    )
    assert editor.calls == 0
    assert report.skipped_chunks == 1
    assert patched == translations


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


# --------------------- labels and repeated openings -------------------------
def _label_chunks() -> list[SimpleNamespace]:
    return [
        _chunk("chunk-000", "Preconditions: user is logged in.\n\nBody one."),
        _chunk("chunk-001", "Preconditions: admin rights.\n\nBody two."),
        _chunk("chunk-002", "Preconditions: session open.\n\nBody three."),
    ]


def test_a_label_rendered_two_ways_is_reported_and_the_minority_patched():
    """``Preconditions:`` has no planned target form; the chunks vote on one."""
    chunks = _label_chunks()
    translations = {
        "chunk-000": "Điều kiện tiên quyết: người dùng đã đăng nhập.\n\nThân một.",
        "chunk-001": "Điều kiện tiên quyết: quyền quản trị.\n\nThân hai.",
        "chunk-002": "Tiền điều kiện: phiên đang mở.\n\nThân ba.",
    }
    editor = _StubEditor(
        lambda translation, fixes: translation.replace("Tiền điều kiện", "Điều kiện tiên quyết")
    )

    patched, report = ConsistencyChecker(editor, max_patches=5).run(chunks, translations, {})

    assert editor.calls == 1
    assert report.labels_checked == 1
    assert report.drifted_labels == 1
    assert patched["chunk-002"].startswith("Điều kiện tiên quyết:")
    assert patched["chunk-000"] == translations["chunk-000"]  # the majority is left alone


def test_a_one_all_tie_is_not_an_inconsistency():
    """One chunk each way is a tie, not a standard to enforce."""
    chunks = _label_chunks()[:2]
    translations = {
        "chunk-000": "Điều kiện: a.\n\nb.",
        "chunk-001": "Tiền điều kiện: c.\n\nd.",
    }
    editor = _StubEditor()

    _, report = ConsistencyChecker(editor, max_patches=5).run(chunks, translations, {})

    assert editor.calls == 0
    assert report.inconsistencies == 0
    assert report.labels_checked == 1  # compared, but nothing to enforce


def test_a_chunk_whose_paragraph_count_differs_is_skipped():
    """Position is the only alignment there is: no match, no guess."""
    chunks = _label_chunks()[:2]
    translations = {
        "chunk-000": "Điều kiện tiên quyết: a.\n\nb.",
        "chunk-001": "Tiền điều kiện: c.",  # one paragraph where the source has two
    }
    editor = _StubEditor()

    _, report = ConsistencyChecker(editor, max_patches=5).run(chunks, translations, {})

    assert editor.calls == 0
    assert report.drifted_labels == 0
    assert report.labels_checked == 0  # only one chunk could be read


def test_a_caption_repeated_across_chunks_drifts():
    """A repeated opening needs no colon and no marker word to be compared."""
    chunks = [
        _chunk("chunk-000", "Figure 2.1 The architecture"),
        _chunk("chunk-001", "Figure 2.1 The architecture"),
        _chunk("chunk-002", "Figure 2.1 The architecture"),
    ]
    translations = {
        "chunk-000": "Hình 2.1 Kiến trúc",
        "chunk-001": "Hình 2.1 Kiến trúc",
        "chunk-002": "Hình 2.1: Kiến trúc",
    }
    editor = _StubEditor(lambda translation, fixes: "Hình 2.1 Kiến trúc")

    patched, report = ConsistencyChecker(editor, max_patches=5).run(chunks, translations, {})

    assert editor.calls == 1
    assert report.drifted_labels == 1
    assert patched["chunk-002"] == "Hình 2.1 Kiến trúc"


def test_a_unit_used_in_one_chunk_is_not_compared():
    chunks = [
        _chunk("chunk-000", "Preconditions: only here"),
        _chunk("chunk-001", "Unrelated body."),
    ]
    _, report = ConsistencyChecker(_StubEditor(), max_patches=5).run(
        chunks, {"chunk-000": "Điều kiện: a", "chunk-001": "Nội dung khác"}, {}
    )
    assert report.labels_checked == 0


def test_a_label_and_a_glossary_term_share_one_patch_request():
    """A patch costs one request however many fixes it carries."""
    chunks = [
        _chunk("chunk-000", "Preconditions: NMT is fast.\n\nBody one."),
        _chunk("chunk-001", "Preconditions: NMT is slow.\n\nBody two."),
        _chunk("chunk-002", "Preconditions: NMT is old.\n\nBody three."),
    ]
    translations = {
        "chunk-000": "Điều kiện tiên quyết: Dịch máy neural nhanh.\n\nThân một.",
        "chunk-001": "Điều kiện tiên quyết: Dịch máy neural chậm.\n\nThân hai.",
        "chunk-002": "Tiền điều kiện: NMT khác.\n\nThân ba.",
    }
    seen: list[list] = []

    def rewrite(translation, fixes):
        seen.append(fixes)
        return translation.replace("Tiền điều kiện", "Điều kiện tiên quyết").replace(
            "NMT", "Dịch máy neural"
        )

    patched, report = ConsistencyChecker(_StubEditor(rewrite), max_patches=5).run(
        chunks, translations, {"NMT": "Dịch máy neural"}
    )

    assert len(seen) == 1
    assert len(seen[0]) == 2  # the label and the term, in one request
    assert patched["chunk-002"].startswith("Điều kiện tiên quyết:")
    assert "Dịch máy neural" in patched["chunk-002"]
    assert report.patched_chunks == 1


def test_structural_drift_reaches_the_report_counters():
    report = ConsistencyReport(labels_checked=7, drifted_labels=2)
    stats = report.as_run_fields()
    assert stats["consistency_labels"] == 7
    assert stats["consistency_label_drift"] == 2


def test_a_term_split_across_lines_still_matches_after_the_memo():
    """The pass normalises each chunk once; it must still see across breaks."""
    chunks = [
        _chunk("chunk-000", "Dịch  máy\nneural is fast."),
        _chunk("chunk-001", "Dịch máy neural is slow."),
    ]
    translations = {"chunk-000": "Dịch máy neural nhanh.", "chunk-001": "NMT chậm."}

    issues = find_inconsistencies(chunks, translations, {"dịch máy neural": "Dịch máy neural"})

    assert len(issues) == 1
    assert issues[0].missing_chunks == ("chunk-001",)


def test_a_patch_that_drops_a_code_marker_is_repaired_before_it_is_applied():
    """A terminology rewrite must not be able to delete a code block.

    Restoration at assembly is a plain ``str.replace``: a marker the editor
    dropped is never substituted, so the block silently disappears. The patch is
    now repaired the same way a translator's reply is, and gated after that like
    any other translation.
    """
    chunks = [
        _chunk("chunk-000", "NMT is fast.\n\n⟦CODE_1⟧"),
        _chunk("chunk-001", "NMT is slow.\n\n⟦CODE_1⟧"),
    ]
    translations = {
        "chunk-000": "Dịch máy neural rất nhanh.\n\n⟦CODE_1⟧",
        "chunk-001": "NMT rất chậm.\n\n⟦CODE_1⟧",
    }

    def rewrite(translation, fixes):
        fixed = translation.replace("NMT", "Dịch máy neural")
        return fixed.replace("⟦CODE_1⟧", "").strip()

    patched, report = ConsistencyChecker(_StubEditor(rewrite), max_patches=5).run(
        chunks, translations, {"NMT": "Dịch máy neural"}
    )

    assert report.patched_chunks == 1
    assert "⟦CODE_1⟧" in patched["chunk-001"]
    assert "Dịch máy neural" in patched["chunk-001"]


def test_a_patch_that_changes_the_heading_structure_is_rejected():
    """The same hard checks a translation faces, applied to a rewrite of it."""
    chunks = [
        _chunk("chunk-000", "NMT is fast."),
        _chunk("chunk-001", "# Data Structures\n\nNMT is fast."),
    ]
    translations = {
        "chunk-000": "Dịch máy neural rất nhanh.",
        "chunk-001": "# Cấu trúc dữ liệu\n\nNMT rất nhanh.",
    }
    editor = _StubEditor(
        lambda translation, fixes: "# Cấu trúc dữ liệu\n\n## Dịch máy neural rất nhanh."
    )

    patched, report = ConsistencyChecker(editor, max_patches=5).run(
        chunks, translations, {"NMT": "Dịch máy neural"}
    )

    assert report.patched_chunks == 0
    assert report.unresolved == ["chunk-001"]
    assert patched["chunk-001"] == translations["chunk-001"]


def test_the_memo_finds_exactly_the_issues_the_naive_scan_finds():
    """Normalising once per chunk must not change what matches."""
    chunks = [
        _chunk("chunk-000", "NMT is  fast.\nAnd NMT too."),
        _chunk("chunk-001", "NMT is slow."),
        _chunk("chunk-002", "Nothing here."),
    ]
    translations = {
        "chunk-000": "Dịch máy neural nhanh.",
        "chunk-001": "NMT chậm.",
        "chunk-002": "Không có gì.",
    }
    pairs = {"NMT": "Dịch máy neural", "nothing": "không có gì"}

    def naive() -> list[tuple]:
        found = []
        for term, expected in pairs.items():
            carrying = [c.id for c in chunks if contains_term(c.text, term)]
            if not carrying:
                continue
            compliant = tuple(cid for cid in carrying if contains_term(translations[cid], expected))
            if not compliant:
                continue
            missing = tuple(cid for cid in carrying if cid not in compliant)
            if missing:
                found.append((term, expected, missing, compliant))
        return found

    actual = find_inconsistencies(chunks, translations, pairs)

    assert [
        (issue.term, issue.expected, issue.missing_chunks, issue.compliant_chunks)
        for issue in actual
    ] == naive()
