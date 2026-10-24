"""End-to-end pipeline tests (mock LLM) + router unit tests."""

from __future__ import annotations

import json

import pytest

from polyglot_lc.orchestrator.router import after_critic, after_verifier


# ----------------------------- router rules ---------------------------------
class _C:
    def __init__(self, score: float) -> None:
        self.score = score


def test_router_critic_below_threshold_routes_to_refine():
    state = {"critique": _C(5.0), "refine_round": 0}
    assert after_critic(state, threshold=7.0, max_rounds=2) == "refine"


def test_router_critic_max_rounds_goes_to_verify():
    state = {"critique": _C(5.0), "refine_round": 2}
    assert after_critic(state, threshold=7.0, max_rounds=2) == "verify"


def test_router_critic_above_threshold_goes_to_verify():
    state = {"critique": _C(8.0), "refine_round": 0}
    assert after_critic(state, threshold=7.0, max_rounds=2) == "verify"


def test_router_verifier_fail_retranslates_once():
    state = {"verify_result": False, "retranslate_count": 0}
    assert after_verifier(state) == "retranslate"
    state2 = {"verify_result": False, "retranslate_count": 1}
    assert after_verifier(state2) == "next"


# ------------------------------ pipeline ------------------------------------
def test_pipeline_end_to_end_mock(mock_env, sample_pdf_file):
    from polyglot_lc.pipeline import TranslationPipeline

    p = TranslationPipeline(quiet=True)
    result = p.run(sample_pdf_file, src="en", tgt="vi")
    assert result.chunks >= 1
    assert result.avg_critic > 0
    assert result.output_path.endswith(".vi.md")
    with open(result.output_path, encoding="utf-8") as fh:
        out = fh.read()
    assert out.strip()


def test_pipeline_writes_jsonl_log(mock_env, sample_pdf_file):
    import os

    os.environ["LOG_JSONL"] = "1"
    from polyglot_lc.pipeline import TranslationPipeline

    p = TranslationPipeline(quiet=True)
    result = p.run(sample_pdf_file, src="en", tgt="vi")
    assert result.log_path
    with open(result.log_path, encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh]
    assert lines, "log must contain events"
    required = {"ts", "step", "chunk_id", "status", "latency", "error"}
    assert required.issubset(set(lines[0].keys()))
    steps = {l["step"] for l in lines}
    assert "ingest" in steps and "translate" in steps


def test_the_run_end_record_carries_the_consistency_tally(mock_env, sample_pdf_file):
    """Chunk bỏ sót và bản vá bị từ chối phải đọc được từ bản ghi cuối lượt.

    Trước đây chúng chỉ nằm rải rác trong log dưới dạng dòng `info` của từng
    chunk, nên một lượt để sót phần lớn chỗ lệch trông giống một lượt sạch.
    """
    import os

    os.environ["LOG_JSONL"] = "1"
    from polyglot_lc.pipeline import TranslationPipeline

    result = TranslationPipeline(quiet=True).run(sample_pdf_file, src="en", tgt="vi")
    with open(result.log_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    end = [record for record in records if record.get("type") == "run_end"][-1]
    assert end["consistency_patched"] == result.consistency_patches
    assert end["consistency_skipped"] == result.consistency_skipped == 0
    assert end["consistency_rejected"] == result.consistency_rejected == 0


def test_a_run_without_glossary_pairs_says_what_it_compared(mock_env, sample_pdf_file, monkeypatch):
    """0 lệch trên 0 thuật ngữ không phải tài liệu sạch.

    Planner fallback (không có cặp nào) để lại `consistency_inconsistencies = 0` y
    hệt một lượt đã so 175 thuật ngữ và không thấy lệch. Số term *thực sự* được so và cờ
    "không có cặp nào" phải lên cả `PipelineResult` lẫn `run_end`.

    The pair's built-in terms are cleared here: an en→vi run always has them, so
    without this the branch under test — a run with *no* pair at all — could not
    be reached from any configuration.
    """
    import os

    from polyglot_lc.pipeline import coordinator

    monkeypatch.setattr(coordinator, "default_glossary", lambda src, tgt: {})
    os.environ["LOG_JSONL"] = "1"
    from polyglot_lc.pipeline import TranslationPipeline

    result = TranslationPipeline(quiet=True).run(sample_pdf_file, src="en", tgt="vi")

    assert result.consistency_no_pairs is True
    assert result.consistency_checked_terms == 0
    # The heuristic terms still reach every prompt: "0 pair" is not "0 term".
    assert result.glossary_terms >= 1

    with open(result.log_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    end = [record for record in records if record.get("type") == "run_end"][-1]
    assert end["consistency_no_pairs"] is True
    assert end["consistency_checked_terms"] == 0


def test_a_run_with_a_glossary_reports_the_terms_it_compared(mock_env, sample_pdf_file, tmp_path):
    """With a term to enforce, the same counters read differently."""
    import os

    os.environ["LOG_JSONL"] = "1"
    from polyglot_lc.pipeline import TranslationPipeline

    csv = tmp_path / "terms.csv"
    csv.write_text("NMT,Dịch máy neural\n", encoding="utf-8")
    result = TranslationPipeline(quiet=True).run(
        sample_pdf_file, src="en", tgt="vi", glossary=str(csv)
    )

    assert result.consistency_no_pairs is False
    assert result.consistency_checked_terms >= 1


def test_a_missing_verifier_is_reported_not_hidden(
    mock_env, sample_pdf_file, monkeypatch, tmp_path
):
    """Chunk không được ai kiểm phải đọc là "chưa kiểm", không phải "đã đạt".

    Verifier lỗi vẫn trả `ok=True` để không chặn bản dịch, nhưng trước đây đó là
    thứ duy nhất được ghi: decision ghi "verifier accepted the draft" và báo cáo
    không có con số nào cho biết QA đã không chạy.
    """
    import os

    from polyglot_lc.agents.verifier import VerifierAgent, VerifyResult

    os.environ["LOG_JSONL"] = "1"
    asked: list[str] = []

    def unavailable(
        self, source, translation, src_lang, tgt_lang, glossary_md="", *, hard_issues=None
    ):
        asked.append(source)
        return VerifyResult(ok=True, reasons=["verifier unavailable; skipped"], skipped=True)

    monkeypatch.setattr(VerifierAgent, "verify", unavailable)
    result = _run_pipeline(sample_pdf_file, tmp_path / "a.md")

    assert asked, "the sample must reach the verifier for this test to mean anything"
    assert result.verify_skipped == len(asked)

    with open(result.log_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    end = [record for record in records if record.get("type") == "run_end"][-1]
    assert end["verify_skipped"] == len(asked)
    decisions = [
        record
        for record in records
        if record.get("type") == "decision" and record.get("kind") == "retranslate"
    ]
    assert decisions
    assert all("accepted" not in " ".join(record["reasons"]) for record in decisions)
    problems = [
        record
        for record in records
        if record.get("type") == "error" and record.get("cause") == "verify"
    ]
    assert len(problems) == len(asked)


def test_drift_the_pass_could_not_fix_is_reported(mock_env, sample_pdf_file, monkeypatch, tmp_path):
    """Chunk bỏ sót và bản vá bị từ chối phải lên `PipelineResult` và `run_end`.

    Chúng từng chỉ là những dòng `info` riêng lẻ trong log, nên một lượt để sót
    phần lớn chỗ lệch trông giống một lượt sạch.
    """
    import os

    from polyglot_lc.pipeline import coordinator
    from polyglot_lc.pipeline.consistency import ConsistencyReport

    os.environ["LOG_JSONL"] = "1"

    class _StubChecker:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, chunks, translations, pairs, *, context=None):
            report = ConsistencyReport(
                checked_terms=len(pairs),
                labels_checked=9,
                drifted_labels=3,
                inconsistencies=3,
                patch_budget=3,
                patched_chunks=2,
                skipped_chunks=4,
            )
            report.unresolved.append("chunk-001")
            return translations, report

    monkeypatch.setattr(coordinator, "ConsistencyChecker", _StubChecker)
    from polyglot_lc.pipeline import TranslationPipeline

    # A glossary pair is what makes the pass run at all: it enforces the pairs the
    # user asked for, never terminology the document did not declare.
    user_glossary = tmp_path / "glossary.csv"
    user_glossary.write_text("NMT,Dịch máy neural\n", encoding="utf-8")

    result = TranslationPipeline(quiet=True).run(
        sample_pdf_file, src="en", tgt="vi", glossary=str(user_glossary)
    )
    assert result.consistency_patches == 2
    assert result.consistency_skipped == 4
    assert result.consistency_rejected == 1
    assert result.consistency_labels == 9
    assert result.consistency_label_drift == 3
    assert result.consistency_patch_budget == 3

    with open(result.log_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    end = [record for record in records if record.get("type") == "run_end"][-1]
    assert end["consistency_skipped"] == 4
    assert end["consistency_rejected"] == 1
    assert end["consistency_labels"] == 9
    assert end["consistency_label_drift"] == 3
    assert end["consistency_patched"] == 2
    # The budget the pass was given is part of the record: each patch is a
    # request over a whole chunk plus the glossary, so its cost is not implied.
    assert end["consistency_patch_budget"] == 3


def test_pipeline_missing_config_error_panel(monkeypatch, sample_pdf_file):
    # No env vars set -> ConfigError with a clear message.
    for name in ("PLANNER", "TRANSLATOR", "CRITIC", "REFINER", "VERIFIER", "EMBED"):
        for suffix in ("BASE_URL", "MODEL", "API_KEY", "TEMPERATURE"):
            monkeypatch.delenv(f"{name}_{suffix}", raising=False)
    from polyglot_lc.pipeline import TranslationPipeline

    with pytest.raises(Exception, match="PLANNER_BASE_URL"):
        TranslationPipeline(quiet=True)


def test_pipeline_out_path_override(mock_env, sample_pdf_file, tmp_path):
    from polyglot_lc.pipeline import TranslationPipeline

    out = tmp_path / "sub" / "out.md"
    p = TranslationPipeline(quiet=True)
    result = p.run(sample_pdf_file, src="en", tgt="vi", out=str(out))
    assert result.output_path == str(out)
    assert out.exists()


# ------------------------ output naming (md + pdf) ---------------------------
def test_default_paths_are_markdown_and_the_pdf_beside_it():
    """The PDF never reuses the Markdown name, and neither reuses the source's."""
    from polyglot_lc.pipeline.assembler import output_paths

    md, pdf = output_paths("book.pdf", "vi")
    assert str(md) == "book.vi.md"
    assert str(pdf) == "book.vi.pdf"


def test_explicit_out_names_one_pair_whichever_suffix_is_given():
    """`--out x.md` and `--out x.pdf` must name the same two files."""
    from polyglot_lc.pipeline.assembler import output_paths

    assert tuple(map(str, output_paths("book.pdf", "vi", "custom.md"))) == (
        "custom.md",
        "custom.pdf",
    )
    assert tuple(map(str, output_paths("book.pdf", "vi", "custom.pdf"))) == (
        "custom.md",
        "custom.pdf",
    )


# --------------------- overlap de-dup in assembly ---------------------------
class _FakeChunk:
    def __init__(self, cid, block_refs, text=""):
        self.id = cid
        self.block_refs = block_refs
        self.text = text


def test_assemble_dedupes_overlapping_chunks():
    """Two chunks sharing tail/head source blocks must emit each block once."""
    from polyglot_lc.pipeline.assembler import assemble

    # chunk0 covers blocks 0,1,2 ; chunk1 re-covers block 2 plus fresh 3,4.
    c0 = _FakeChunk("c0", [0, 1, 2])
    c1 = _FakeChunk("c1", [2, 3, 4])
    out = assemble(
        [c0, c1],
        {
            "c0": "Para0\n\nPara1\n\nPara2",
            "c1": "Para2-again\n\nPara3\n\nPara4",
        },
    )
    assert "Para0\n\nPara1\n\nPara2\n\nPara3\n\nPara4" == out
    assert "Para2-again" not in out


# --------------------- truncated-translation coverage ------------------------
def _coverage_chunk(n_paras=8):
    """A chunk whose source is *n_paras* full paragraphs."""
    body = [
        f"Paragraph {i} carries enough english words to give the chunk a real "
        "length for the coverage measurement to compare against."
        for i in range(n_paras)
    ]
    return _FakeChunk("c0", list(range(n_paras)), text="\n\n".join(body))


def test_coverage_error_flags_a_reply_missing_the_chunks_tail():
    """A reply cut short mid-chunk must name the shortfall, not pass silently.

    The provider's finish_reason only catches a reply cut at the token limit;
    an early stop still reports "stop" while the tail of the chunk was never
    translated — that is how section 13.10-13.13 reached the output in
    English (and the .md without them at all).
    """
    from polyglot_lc.pipeline.assembler import CHUNK_SEPARATOR, coverage_error

    chunk = _coverage_chunk()
    paras = chunk.text.split(CHUNK_SEPARATOR)
    truncated = CHUNK_SEPARATOR.join(paras[:4])  # half the chunk's words
    reason = coverage_error(chunk, truncated)
    assert reason is not None
    assert "truncated" in reason


def test_coverage_error_accepts_a_complete_reply():
    from polyglot_lc.pipeline.assembler import coverage_error

    chunk = _coverage_chunk()
    assert coverage_error(chunk, chunk.text) is None


def test_coverage_error_accepts_a_reply_that_merges_paragraphs():
    """Fewer paragraphs than blocks is a merge, not a loss: same words."""
    from polyglot_lc.pipeline.assembler import CHUNK_SEPARATOR, coverage_error

    chunk = _coverage_chunk()
    merged = CHUNK_SEPARATOR.join(chunk.text.split(CHUNK_SEPARATOR))
    one_paragraph = " ".join(merged.split(CHUNK_SEPARATOR))
    assert coverage_error(chunk, one_paragraph) is None


def test_coverage_error_passes_when_there_is_nothing_to_measure():
    """No source text (or no reply) is someone else's call, never a flag."""
    from polyglot_lc.pipeline.assembler import coverage_error

    assert coverage_error(_FakeChunk("c0", [0]), "một câu dịch") is None
    assert coverage_error(_coverage_chunk(), "") is None


# ------------------- the coverage gate inside the run -----------------------
def _run_pipeline(path, out, **kw):
    from polyglot_lc.pipeline import TranslationPipeline

    return TranslationPipeline(quiet=True).run(path, src="en", tgt="vi", out=str(out), **kw)


def _force_coverage_gate(monkeypatch):
    """Measure replies even under the mock harness, whose canned answers are
    deliberately short pseudo-translations."""
    from polyglot_lc.pipeline import coordinator

    monkeypatch.setattr(coordinator, "_coverage_enabled", lambda cfg: True)


def test_a_cut_short_reply_is_retried_and_never_checkpointed(
    mock_env, tmp_path, monkeypatch, sample_pdf_file
):
    """The gate's cap path: a chunk that keeps coming back cut short is asked
    two more times, its partial words still reach this run's output, and the
    checkpoint stays empty so a later --resume can ask again.

    This is the chunk-036 failure: a reply cut mid-word was accepted, stored
    and inherited by every later resume, silently deleting sections
    13.10-13.13 from the Markdown and leaving them in English in the PDF.
    """
    from polyglot_lc.llm.client import TokenUsage, clear_mock_replies
    from polyglot_lc.llm.resilience import RetryStats

    clear_mock_replies()
    _force_coverage_gate(monkeypatch)
    RetryStats.reset()
    first = _run_pipeline(sample_pdf_file, tmp_path / "a.md")
    assert first.chunks >= 1
    # Re-asked at least twice per chunk: the cut replies are retried, and the
    # salvage split adds its own asks for the halves before the reply that is
    # STILL partial is kept.
    assert RetryStats.snapshot()["chunk_retries"] >= 2 * first.chunks

    TokenUsage.reset()
    second = _run_pipeline(sample_pdf_file, tmp_path / "b.md", resume=True)
    assert second.resumed_chunks == 0  # nothing partial was checkpointed
    assert TokenUsage.snapshot()["calls"] > 0  # so the chunk was asked again
    # The retry counters are process-global and later tests read them from
    # their run records: leave the suite the counters it found.
    RetryStats.reset()


def test_a_covering_reply_checkpoints_normally(mock_env, tmp_path, monkeypatch, sample_pdf_file):
    """A reply that covers its chunk passes the gate and resumes for free."""
    from polyglot_lc.llm.client import TokenUsage, clear_mock_replies, register_mock_reply

    clear_mock_replies()
    _force_coverage_gate(monkeypatch)
    # The translator's own instruction line is the one prompt marker unique to
    # its requests; its reply is long enough to cover any chunk of the sample.
    register_mock_reply(
        "Output ONLY the translation, no commentary.",
        "Bản dịch đầy đủ của tài liệu mẫu. " * 40,
    )
    first = _run_pipeline(sample_pdf_file, tmp_path / "a.md")
    assert first.chunks >= 1

    TokenUsage.reset()
    second = _run_pipeline(sample_pdf_file, tmp_path / "b.md", resume=True)
    assert second.resumed_chunks == second.chunks
    assert TokenUsage.snapshot()["calls"] == 0  # nothing to translate again


def test_a_chunk_cut_short_every_time_is_salvaged_by_splitting(
    mock_env, tmp_path, monkeypatch, sample_pdf_file
):
    """The gate's salvage path: when whole-chunk asks keep coming back cut
    short, the chunk is asked again as two halves — a shorter question is a
    different question — the halves are merged, and the merged translation
    is checkpointed so a later --resume costs nothing.

    This is the chunk that fails every which way: two re-asks do not undo a
    deterministic early stop, but half the chunk does not reach it.
    """
    from polyglot_lc.ingest import load_document
    from polyglot_lc.llm.client import TokenUsage, clear_mock_replies
    from polyglot_lc.llm.resilience import RetryStats
    from polyglot_lc.pipeline import coordinator
    from polyglot_lc.segment.chunker import chunk_document

    clear_mock_replies()
    _force_coverage_gate(monkeypatch)
    # Every whole-chunk question stays cut short, however often it is asked.
    # Only a reply built from the halves' own replies — so the halves, once
    # split — covers its chunk; anything else is missing the tail.
    full_texts = {c.text for c in chunk_document(load_document(sample_pdf_file))}
    asked: dict[str, int] = {}
    half_replies: set[str] = set()

    def fake_coverage(chunk, translated):
        if chunk.text not in full_texts:
            half_replies.add(translated)  # a half is asked and answers in full
            return None
        asked[chunk.text] = asked.get(chunk.text, 0) + 1
        if half_replies and all(h in translated for h in half_replies):
            return None  # the merge of the halves is the whole chunk
        return "truncated translation: (fake) missing the tail"

    monkeypatch.setattr(coordinator, "coverage_error", fake_coverage)
    RetryStats.reset()
    first = _run_pipeline(sample_pdf_file, tmp_path / "a.md")
    assert first.chunks >= 1
    # At least one chunk was asked at least three times: twice cut short,
    # then merged from its halves and measured against the whole.
    assert any(n >= 3 for n in asked.values())

    # The salvaged translations reached the checkpoint: a resume re-asks nothing.
    TokenUsage.reset()
    second = _run_pipeline(sample_pdf_file, tmp_path / "b.md", resume=True)
    assert second.resumed_chunks == second.chunks
    assert TokenUsage.snapshot()["calls"] == 0
    # Leave the suite the counters it found.
    RetryStats.reset()


class _SilentGraph:
    """A chunk graph whose translator always answers with nothing."""

    def __init__(self, limit: int) -> None:
        self.calls = 0
        self.limit = limit

    def invoke(self, state, config=None):
        self.calls += 1
        if self.calls > self.limit:
            raise AssertionError("an empty reply was re-asked without a bound")
        return {"translated": "", "chunk": state["chunk"], "trace": {}, "path": "standard"}


def test_a_provider_that_answers_with_nothing_does_not_hold_the_run(
    mock_env, tmp_path, monkeypatch, sample_pdf_file
):
    """An empty reply is re-asked a bounded number of times, then given up on.

    The wait for an empty reply used to have no ceiling: a provider that refused,
    or a model that answered with whitespace, left the run sitting on that one
    chunk until the reader stopped it. The chunk now ends up marked untranslated —
    visible in the output, absent from the checkpoint, so --resume asks again —
    and the run finishes.
    """
    import os

    from polyglot_lc.ingest import load_document
    from polyglot_lc.llm.resilience import RetryStats
    from polyglot_lc.pipeline import coordinator
    from polyglot_lc.segment.chunker import chunk_document

    os.environ["LOG_JSONL"] = "1"
    chunks = chunk_document(load_document(sample_pdf_file))
    limit = 25 * len(chunks)
    graph = _SilentGraph(limit)
    monkeypatch.setattr(coordinator, "make_chunk_graph", lambda ctx: graph)
    # A real back-off would slow the test without testing anything.
    monkeypatch.setattr(coordinator, "jittered_delay", lambda cfg, attempt: 0.0)
    # This run is made of retries: the counters are process-global, so leave the
    # suite the ones it found (other tests assert exact retry totals).
    RetryStats.reset()

    result = _run_pipeline(sample_pdf_file, tmp_path / "a.md")

    assert result.chunks == len(chunks)
    assert result.chunks_failed == result.chunks
    assert graph.calls <= limit
    # A chunk run again is a retry, and the report's total must include it:
    # counting only request-level retries understated what the run cost.
    stats = RetryStats.snapshot()
    assert stats["chunk_retries"] > 0
    assert result.chunk_retries == stats["chunk_retries"]
    assert (
        result.retries
        == stats["chunk_retries"] + stats["rate_limit_retries"] + stats["transient_retries"]
    )
    with open(result.log_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    end = [record for record in records if record.get("type") == "run_end"][-1]
    assert end["retries"] == result.retries
    assert end["chunk_retries"] == result.chunk_retries
    with open(result.output_path, encoding="utf-8") as fh:
        text = fh.read()
    assert "untranslated" in text  # the hole is visible, not silent

    # Nothing empty reached the checkpoint: a later run asks the chunks again.
    graph.calls = 0
    second = _run_pipeline(sample_pdf_file, tmp_path / "b.md", resume=True)
    assert second.resumed_chunks == 0
    RetryStats.reset()


def test_assemble_no_overlap_is_unchanged():
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0])
    c1 = _FakeChunk("c1", [1])
    out = assemble([c0, c1], {"c0": "A", "c1": "B"})
    assert out == "A\n\nB"


def test_assemble_keeps_every_distinct_contents_entry():
    """A contents page is not duplicated prose.

    Consecutive entries share a long title prefix, so a near-duplicate check
    could read 'two entries that look alike' as 'the same paragraph rendered
    twice' and delete a real entry. Two entries whose page numbers differ are
    two entries, even across a chunk boundary.
    """
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0])
    c1 = _FakeChunk("c1", [1])
    first = "Chương 3 Mảng và danh sách liên kết tuyến tính … 42"
    second = "Chương 3 Mảng và danh sách liên kết tuyến tính bổ sung … 45"
    out = assemble([c0, c1], {"c0": first, "c1": second})
    assert first in out
    assert second in out


def test_two_paragraphs_that_differ_by_one_number_are_both_kept():
    """Only the pair across a chunk boundary is de-duplicated, and only when
    the evidence they carry (numbers, identifiers, code markers) is identical.

    The document-wide pass this replaces collapsed *any* two similar paragraphs,
    so a pair of clauses — or two exercises — differing by one figure lost the
    second one without a marker.
    """
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0])
    c1 = _FakeChunk("c1", [1])
    first = "Điều 1.2: Người dùng phải đăng nhập trước khi gửi yêu cầu lên máy chủ."
    second = "Điều 1.3: Người dùng phải đăng nhập trước khi gửi yêu cầu lên máy chủ."
    out = assemble([c0, c1], {"c0": first, "c1": second})
    assert first in out
    assert second in out
    # And nothing inside one chunk is ever dropped, however alike it reads.
    merged = assemble([_FakeChunk("c0", [0, 1])], {"c0": f"{first}\n\n{second}"})
    assert merged == f"{first}\n\n{second}"


def test_a_merged_overlap_does_not_take_new_content_with_it():
    """The replay is cut by block ownership, not by counting paragraphs.

    The translator returned the replayed block and the first new block as one
    paragraph. Cutting the overlap by count (one block = one paragraph) removed
    that whole paragraph, deleting the new block's translation with no marker.
    A paragraph that also carries a new block's words has to stay.
    """
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0, 1], "A\n\nB")
    c1 = _FakeChunk("c1", [1, 2, 3], "B\n\nC\n\nD")
    out = assemble(
        [c0, c1],
        {"c0": "PA\n\nPB", "c1": "PB và PC\n\nPD"},
    )
    assert out == "PA\n\nPB\n\nPB và PC\n\nPD"
    assert "PC" in out


def test_assemble_collapses_consecutive_reworded_duplicates():
    """The same content translated twice in slightly different words is one block."""
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0])
    c1 = _FakeChunk("c1", [1])
    first = (
        "GitHub là một dịch vụ lưu trữ cung cấp không gian lưu trữ cho các kho "
        "lưu trữ Git và một giao diện web tiện lợi."
    )
    reworded = (
        "GitHub là một dịch vụ lưu trữ cung cấp không gian cho các kho lưu trữ "
        "Git và một giao diện web tiện lợi."
    )
    out = assemble([c0, c1], {"c0": first, "c1": reworded})
    assert out == first


def test_assemble_keeps_different_paragraphs():
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0])
    c1 = _FakeChunk("c1", [1])
    out = assemble(
        [c0, c1],
        {
            "c0": "Cấu trúc dữ liệu là cách tổ chức dữ liệu trong bộ nhớ.",
            "c1": "Thuật toán là một chuỗi các bước để giải quyết vấn đề.",
        },
    )
    assert out.count("\n\n") == 1
    assert "Thuật toán" in out


def test_assemble_marks_a_chunk_that_never_got_a_translation():
    """A failed chunk must be visible, never a silently missing section."""
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0])
    c1 = _FakeChunk("c1", [1])
    out = assemble([c0, c1], {"c0": "Đoạn một."})
    # The note names what is gone: c0 covers block 0 only, so block 1 is a
    # total loss and a re-run has to recover exactly that.
    assert "<!-- c1: untranslated (source block 1 lost) -->" in out
    assert "Đoạn một." in out


def test_assemble_keeps_the_next_chunks_head_when_the_previous_emitted_nothing():
    """Overlap may only be stripped when the previous chunk's copy survives.

    c1 never got a translation, so block 2 reaches the document only through
    c2's head. Stripping that head deletes the content twice over — and with no
    marker, because c2 itself did produce a translation.
    """
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0])
    c1 = _FakeChunk("c1", [1, 2])
    c2 = _FakeChunk("c2", [2, 3])  # re-covers block 2 as its head
    out = assemble([c0, c1, c2], {"c0": "P0", "c2": "P2\n\nP3"})
    assert "P2" in out
    assert "P3" in out
    # Block 1 is covered by no neighbour, so it is the real loss.
    assert "<!-- c1: untranslated (source block 1 lost) -->" in out


def test_assemble_still_strips_the_head_when_the_previous_chunk_did_emit():
    """The fix must not cost the de-duplication it protects."""
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0, 1])
    c1 = _FakeChunk("c1", [1, 2])
    out = assemble([c0, c1], {"c0": "P0\n\nP1", "c1": "P1\n\nP2"})
    assert out == "P0\n\nP1\n\nP2"


def test_lost_block_note_reports_the_contiguous_range():
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0])
    c1 = _FakeChunk("c1", [1, 2, 3, 4])
    c2 = _FakeChunk("c2", [5])
    out = assemble([c0, c1, c2], {"c0": "P0", "c2": "P5"})
    assert "<!-- c1: untranslated (source blocks 1-4 lost) -->" in out


def test_lost_block_note_says_when_a_neighbour_carries_the_content():
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0])
    c1 = _FakeChunk("c1", [1])
    c2 = _FakeChunk("c2", [1, 2])  # c2's head re-covers c1's only block
    out = assemble([c0, c1, c2], {"c0": "P0", "c2": "P1\n\nP2"})
    assert "P1" in out
    assert "<!-- c1: untranslated (content carried by the neighbouring chunk) -->" in out


def test_assemble_partial_does_not_strip_a_head_whose_neighbour_is_pending():
    """While a run is going, the overlap of a pending chunk is its own copy."""
    from polyglot_lc.pipeline.assembler import assemble_partial

    c0 = _FakeChunk("c0", [0, 1])
    c1 = _FakeChunk("c1", [1, 2])
    out = assemble_partial([c0, c1], {"c1": "P1\n\nP2"})
    assert "<!-- c0: pending -->" in out
    assert "P1" in out
    assert "P2" in out


def test_assemble_partial_still_dedupes_a_finished_pair():
    from polyglot_lc.pipeline.assembler import assemble_partial

    c0 = _FakeChunk("c0", [0, 1])
    c1 = _FakeChunk("c1", [1, 2])
    out = assemble_partial([c0, c1], {"c0": "P0\n\nP1", "c1": "P1\n\nP2"})
    assert out == "P0\n\nP1\n\nP2"


def test_assemble_never_drops_fresh_content_on_collapsed_paras():
    """If the translator collapsed the overlap paragraphs, we must not over-strip."""
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0, 1])
    # chunk1 has 1 overlap block but its translation has only 1 paragraph.
    c1 = _FakeChunk("c1", [1, 2])
    out = assemble([c0, c1], {"c0": "P0\n\nP1", "c1": "merged-P1P2"})
    # drop=min(1, len(paras)-1)=min(1,0)=0 -> nothing stripped.
    assert "merged-P1P2" in out


# ------------------------- emphasis in the Markdown output ------------------
def test_the_markdown_output_has_markdown_emphasis():
    """Vocab gửi cho model là sentinel, nhưng tài liệu giao cho người đọc là Markdown."""
    from polyglot_lc.pipeline.assembler import assemble

    chunks = [_FakeChunk("c0", [0, 1])]
    out = assemble(chunks, {"c0": "A ⟦BOLD⟧B⟦/BOLD⟧ C"}, {})
    assert "**B**" in out
    assert "⟦" not in out


def test_a_long_document_fills_the_glossary_budget(
    mock_env, sample_pdf_file, monkeypatch, tmp_path
):
    """A book-sized text is no longer pinned at the memo's 40 terms.

    The planner mock returns no terms, so what lands in ``glossary_md`` is the
    heuristic glossary — the half whose fixed cap the budget replaced.
    """
    from polyglot_lc.ingest.model import Block, Document
    from polyglot_lc.orchestrator import graph
    from polyglot_lc.pipeline import coordinator

    blocks: list[Block] = []
    for i in range(90):
        blocks.append(Block(kind="heading", text=f"Chapter {i}", level=2))
        terms = f"Widget{i}Alpha Widget{i}Beta Widget{i}Gamma"
        blocks.append(
            Block(
                kind="paragraph",
                text=f"{terms} " + "the system stores and returns the value. " * 105,
            )
        )
        # Each term must recur: the extractor keeps only what it sees twice.
        blocks.append(Block(kind="paragraph", text=f"{terms} again, as before."))
    doc = Document(blocks=blocks, metadata={"filename": "long.pdf"})
    monkeypatch.setattr(coordinator, "load_document", lambda path: doc)

    seen: dict = {}
    real = graph.analyze_document

    def spy(*args, **kwargs):
        plan = real(*args, **kwargs)
        seen["plan"] = plan
        return plan

    monkeypatch.setattr(graph, "analyze_document", spy)

    from polyglot_lc.pipeline import TranslationPipeline

    result = TranslationPipeline(quiet=True).run(
        sample_pdf_file, src="en", tgt="vi", out=str(tmp_path / "long.md")
    )

    assert result.chunks >= 100
    lines = [line for line in seen["plan"].glossary_md.splitlines() if line.startswith("- ")]
    assert len(lines) > 80
