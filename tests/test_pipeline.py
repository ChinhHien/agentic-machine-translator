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
    result = p.run(sample_pdf_file, src="en", tgt="vi", domain="tech")
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
    def __init__(self, cid, block_refs):
        self.id = cid
        self.block_refs = block_refs


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


def test_assemble_no_overlap_is_unchanged():
    from polyglot_lc.pipeline.assembler import assemble

    c0 = _FakeChunk("c0", [0])
    c1 = _FakeChunk("c1", [1])
    out = assemble([c0, c1], {"c0": "A", "c1": "B"})
    assert out == "A\n\nB"


def test_assemble_keeps_every_distinct_contents_entry():
    """A contents page is not duplicated prose.

    Consecutive entries share a long title prefix, so the near-duplicate
    collapse used to see 'two entries that look alike' as 'the same paragraph
    rendered twice' and delete a real entry. Keeping a distinct entry beats
    folding it.
    """
    from polyglot_lc.pipeline.assembler import CHUNK_SEPARATOR, _deduplicate_paragraphs

    toc = [
        "Chương 3 Mảng … 42",
        "Chương 3 Mảng và danh sách liên kết … 45",
        "Chương 4 Chuỗi … 61",
    ]
    out = _deduplicate_paragraphs(CHUNK_SEPARATOR.join(toc))
    assert out.split(CHUNK_SEPARATOR) == toc


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
