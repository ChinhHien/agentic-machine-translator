"""Tests for the analysis layer: complexity, draft signals, unified planner."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from polyglot_lc.analysis import (
    analyze_document,
    draft_signals,
    heuristic_document_complexity,
    merge_glossary_md,
    score_chunk,
)
from polyglot_lc.analysis.unified_planner import (
    DOC_EXCERPT_CHARS,
    DOC_EXCERPT_WINDOWS,
    OUTLINE_MAX_CHARS,
    SECTION_PREVIEW_CHARS,
    GlossaryPair,
    render_outline,
    sample_body,
)
from polyglot_lc.ingest.model import Block, Document
from polyglot_lc.segment.chunker import chunk_document, count_tokens
from polyglot_lc.segment.glossary import GlossaryTerm, extract_glossary, glossary_budget


def _chunk(cid: str, text: str, tokens: int = 10, section: str = "") -> SimpleNamespace:
    return SimpleNamespace(id=cid, text=text, token_count=tokens, section_path=section)


class _StubPlanner:
    """Planner stand-in that records requests instead of calling a model."""

    def __init__(self, reply: dict | None = None, fail: bool = False) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.reply = reply or {}
        self.fail = fail

    def chat_json(self, messages, cache_prefix: bool = False):
        self.calls += 1
        self.prompts.append(messages[0]["content"])
        if self.fail:
            raise RuntimeError("planner down")
        return dict(self.reply)


# ------------------------------- complexity ---------------------------------
def test_structural_features_raise_complexity():
    plain = score_chunk("A short and simple sentence.", token_count=6)
    hard = score_chunk(
        "### Deep heading\n\n- item one with ⟦CODE_1⟧\n- item two with ⟦CODE_2⟧",
        token_count=40,
    )
    assert plain.value < hard.value
    assert any("placeholder" in r for r in hard.reasons)


def test_longer_chunks_score_higher():
    short = score_chunk("word " * 20, token_count=20)
    long = score_chunk("word " * 20, token_count=400)
    assert long.value > short.value
    assert any("long chunk" in r for r in long.reasons)


def test_document_complexity_is_blended_in():
    text = "A short and simple sentence."
    alone = score_chunk(text, token_count=6, doc_complexity=0.0)
    part_of_hard_doc = score_chunk(text, token_count=6, doc_complexity=1.0)
    assert part_of_hard_doc.value > alone.value
    # A fully-complex document contributes its whole 30% weight.
    assert part_of_hard_doc.value - alone.value == pytest.approx(0.3, abs=0.01)


def test_planner_section_note_raises_complexity():
    base = score_chunk("A short and simple sentence.", token_count=6)
    noted = score_chunk("A short and simple sentence.", token_count=6, section_note="careful here")
    assert noted.value > base.value


def test_heuristic_document_complexity_scales_with_length():
    assert heuristic_document_complexity("") == 0.0
    small = heuristic_document_complexity("NMT is fine. " * 5)
    large = heuristic_document_complexity("NMT is fine. " * 2000)
    assert 0.0 <= small < large <= 1.0


# ------------------------------ draft signals -------------------------------
def test_clean_draft_scores_high():
    signals = draft_signals("Neural machine translation systems.", "[vi] Hệ thống dịch.")
    assert signals.score >= 9.0
    assert signals.clean


def test_empty_and_identical_drafts_score_zero():
    assert draft_signals("some source text here", "   ").score == 0.0
    same = draft_signals("some source text here and more", "some source text here and more")
    assert same.score == 0.0
    assert not same.clean


def test_missing_placeholder_is_reported_as_hard_issue():
    signals = draft_signals("keep ⟦CODE_1⟧ intact", "bỏ mất chỗ trống")
    assert not signals.clean
    assert signals.score < 9.0


def test_truncated_long_draft_is_flagged():
    long_source = "This is a long English paragraph about translation. " * 8
    signals = draft_signals(long_source, "short")
    assert not signals.clean
    assert any("source length" in issue for issue in signals.issues)


def test_short_chunks_are_not_judged_on_length_ratio():
    # A terse but legitimate translation of a short chunk must stay clean.
    signals = draft_signals("Translate me please.", "Dịch tôi.")
    assert signals.clean, signals.issues


def test_untranslated_source_is_detected():
    source = "The pipeline splits documents into translatable chunks for the model."
    signals = draft_signals(source, source)  # byte-identical -> caught explicitly
    assert signals.score == 0.0


# ----------------------------- unified planner ------------------------------
def test_unified_planner_costs_one_request_for_many_sections():
    chunks = [
        _chunk(f"chunk-{i:03d}", f"Body of section {i}.", section=f"Section {i}") for i in range(20)
    ]
    planner = _StubPlanner(
        reply={
            "summary": "twenty sections",
            "strategy": "formal",
            "few_shot": ["a => b"],
            "pitfalls": ["avoid literalism"],
            "glossary_terms": [{"term": "NMT", "translation": "dịch máy", "reason": "acronym"}],
            "complexity": 0.66,
            "section_notes": {"Section 0": "be careful"},
            "notes": "ok",
        }
    )
    plan = analyze_document(
        chunks,
        planner,
        src_lang="en",
        tgt_lang="vi",
        doc_text="Body of section 0.",
        doc_title="Doc",
    )
    assert planner.calls == 1
    assert plan.used_llm is True
    assert plan.complexity == pytest.approx(0.66)
    assert plan.glossary_pairs == [GlossaryPair("NMT", "dịch máy", "acronym")]
    assert plan.note_for("Section 0") == "be careful"
    assert plan.section_count == 20
    # The prompt must carry the structural context the Planner needs.
    assert "Section 7" in planner.prompts[0]


def test_section_note_falls_back_to_parent_section():
    chunks = [_chunk("chunk-000", "text", section="Chapter 1 > Details")]
    planner = _StubPlanner(reply={"section_notes": {"Chapter 1": "keep the tone"}})
    plan = analyze_document(
        chunks,
        planner,
        src_lang="en",
        tgt_lang="vi",
    )
    assert plan.note_for("Chapter 1 > Details") == "keep the tone"


def test_unified_planner_survives_model_failure():
    chunks = [_chunk("chunk-000", "NMT appears twice. NMT again.", section="Intro")]
    planner = _StubPlanner(fail=True)
    plan = analyze_document(
        chunks,
        planner,
        src_lang="en",
        tgt_lang="vi",
        doc_text="NMT appears twice. NMT again.",
    )
    assert planner.calls == 1
    assert plan.used_llm is False
    assert plan.strategy  # a usable default strategy
    assert plan.summary  # local structural summary
    assert plan.complexity >= 0.0


def test_unified_planner_clamps_and_tolerates_bad_values():
    chunks = [_chunk("chunk-000", "text")]
    planner = _StubPlanner(
        reply={
            "strategy": "",
            "complexity": "not-a-number",
            "glossary_terms": [{"term": "no translation"}, "not a dict"],
            "few_shot": 42,
            "pitfalls": None,
            "section_notes": ["not a dict"],
        }
    )
    plan = analyze_document(chunks, planner, src_lang="en", tgt_lang="vi")
    assert plan.strategy  # fell back to the default
    assert 0.0 <= plan.complexity <= 1.0
    assert plan.glossary_pairs == []  # pair without a translation dropped
    assert plan.few_shot == []  # a non-list field is dropped
    assert plan.pitfalls == []
    assert plan.section_notes == {}


def test_bare_string_few_shot_is_wrapped_not_iterated():
    chunks = [_chunk("chunk-000", "text")]
    planner = _StubPlanner(reply={"few_shot": "Hello => Xin chào"})
    plan = analyze_document(chunks, planner, src_lang="en", tgt_lang="vi")
    assert plan.few_shot == ["Hello => Xin chào"]


# ------------------------------ glossary merge ------------------------------
def test_glossary_precedence_user_then_planner_then_heuristics():
    md = merge_glossary_md(
        planner_pairs=[GlossaryPair("NMT", "dịch máy neural")],
        heuristic_terms=[
            GlossaryTerm("NMT", 3, "proper_noun"),
            GlossaryTerm("Berlin", 2, "capitalized"),
        ],
        user_glossary={"NMT": "MT neural (của người dùng)"},
    )
    lines = md.splitlines()
    # The user's entry wins and the planner's conflicting one is dropped.
    assert any("người dùng" in line for line in lines)
    assert not any("dịch máy neural" in line for line in lines)
    # Heuristic terms survive only when nothing authoritative covered them.
    assert any("Berlin" in line for line in lines)
    assert sum(1 for line in lines if "NMT" in line) == 1


def test_glossary_empty_renders_none_marker():
    assert merge_glossary_md(planner_pairs=[], heuristic_terms=[]) == "(none)"


# ------------------------------- prompt contract ----------------------------
def test_planner_prompt_pins_the_complexity_scale():
    """complexity is 0-1 and is clamped, so the prompt must say so.

    Unlabelled, a model answers on its own scale (0-10, or a percentage) and
    every such answer is silently clamped to 1.0 — which then drives the
    fast-path gate that decides whether a chunk pays for Critic/Refiner.
    """
    chunks = [_chunk("chunk-000", "text", section="Intro")]
    planner = _StubPlanner(reply={})
    analyze_document(chunks, planner, src_lang="en", tgt_lang="vi")
    prompt = planner.prompts[0]
    assert "[0, 1]" in prompt
    assert "not 0-10" in prompt
    # section_notes must be keyed by names the prompt actually shows.
    assert "section_notes keys" in prompt
    assert "Intro" in prompt


# ----------------------- long documents: 1000+ pages ------------------------
def _tagged_text(segments: int = 200, sentences: int = 40) -> str:
    """A body whose every 1/200th is labelled, so an excerpt can be traced."""
    return (
        "".join(
            f"[[seg-{i:03d}]] " + "the service returns an error code. " * sentences
            for i in range(segments)
        )
        + "[[end]] the document is over."
    )


def test_the_body_excerpt_is_drawn_from_the_whole_document():
    """A 1000-page book opens with a cover and a contents list, not a book."""
    text = _tagged_text()
    planner = _StubPlanner(reply={})
    analyze_document(
        [_chunk("chunk-000", "text", section="Intro")],
        planner,
        src_lang="en",
        tgt_lang="vi",
        doc_text=text,
    )
    prompt = planner.prompts[0]
    tags = [int(m) for m in re.findall(r"\[\[seg-(\d{3})\]\]", prompt)]
    assert tags
    assert min(tags) == 0, "the opening still counts"
    assert max(tags) >= 100, "the second half of the book is represented"
    assert "[[end]]" in prompt, "and so is the last page of a 1000-page book"


def test_the_body_excerpt_spends_no_more_than_its_budget():
    excerpt = sample_body(_tagged_text())
    assert len(excerpt) <= DOC_EXCERPT_CHARS
    assert excerpt.count("[...]") == DOC_EXCERPT_WINDOWS - 1


def test_a_document_that_fits_the_budget_is_sent_whole():
    text = "Short document sentence. " * 60
    assert sample_body(text) == text


def test_the_outline_never_exceeds_its_budget():
    outline = {
        f"Chapter {i}: a heading of realistic length": ("preview sentence " * 10)[
            :SECTION_PREVIEW_CHARS
        ]
        for i in range(150)
    }
    rendered = render_outline(outline)
    assert len(rendered) <= OUTLINE_MAX_CHARS
    assert all(name in rendered for name in outline), "previews shrink before names are cut"
    assert rendered.count("…") == len(outline), "every preview was clipped to fit"


def test_an_outline_too_large_for_its_names_says_what_it_dropped():
    outline = {f"Chapter {i}: a heading long enough to matter": "p" * 160 for i in range(5000)}
    rendered = render_outline(outline)
    assert len(rendered) <= OUTLINE_MAX_CHARS
    shown = [line for line in rendered.splitlines() if not line.startswith("- (+")]
    omitted = int(re.search(r"- \(\+(\d+) more sections not shown\)", rendered).group(1))
    assert omitted > 0
    assert omitted == len(outline) - len(shown)


def test_a_normal_book_keeps_full_section_previews():
    outline = {f"Chapter {i}": f"Preview of chapter {i}." for i in range(75)}
    expected = "\n".join(f"- {name}: {preview}" for name, preview in outline.items())
    assert render_outline(outline) == expected


def test_a_thousand_page_plan_prompt_stays_small():
    blocks: list[Block] = []
    for page in range(1000):
        blocks.append(Block(kind="heading", text=f"Chapter {page + 1}", level=2))
        blocks.append(Block(kind="paragraph", text="The service returns an error code. " * 40))
    doc = Document(blocks=blocks)
    chunks = chunk_document(doc)
    planner = _StubPlanner(reply={})
    analyze_document(chunks, planner, src_lang="en", tgt_lang="vi", doc_text=doc.plain_text())
    assert len(chunks) >= 300
    assert count_tokens(planner.prompts[0]) < 10_000


# ----------------------------- glossary budget ------------------------------
def test_the_glossary_budget_grows_with_the_document():
    assert glossary_budget(0) == 40
    assert glossary_budget(200_000) == 65
    assert glossary_budget(1_020_000) == 167
    assert glossary_budget(10_000_000) == 200  # never past the user glossary's cap


def test_the_planner_accepts_more_terms_from_a_long_document():
    pairs = [{"term": f"Term {i}", "translation": f"Thuật ngữ {i}"} for i in range(120)]
    planner = _StubPlanner(reply={"glossary_terms": pairs, "complexity": 0.2})
    doc_text = "x" * 1_020_000

    plan = analyze_document(
        [_chunk("chunk-000", "text", section="Intro")],
        planner,
        src_lang="en",
        tgt_lang="vi",
        doc_text=doc_text,
    )

    assert len(plan.glossary_pairs) == 120  # 40 was the old fixed ceiling
    assert f"up to {glossary_budget(len(doc_text))} terms" in planner.prompts[0]


def test_a_short_document_keeps_the_planners_old_ceiling():
    pairs = [{"term": f"Term {i}", "translation": f"Thuật ngữ {i}"} for i in range(60)]
    planner = _StubPlanner(reply={"glossary_terms": pairs, "complexity": 0.2})

    plan = analyze_document(
        [_chunk("chunk-000", "text", section="Intro")],
        planner,
        src_lang="en",
        tgt_lang="vi",
        doc_text="x" * 2_000,
    )

    assert len(plan.glossary_pairs) == 40


def test_the_user_glossary_is_never_truncated_by_the_budget():
    """The budget bounds what the machine adds, never what the user supplied."""
    user = {f"term {i}": f"thuật ngữ người dùng {i}" for i in range(5)}

    md = merge_glossary_md(
        planner_pairs=[GlossaryPair(f"p{i}", f"dạng máy {i}") for i in range(3)],
        heuristic_terms=[GlossaryTerm(f"h{i}", 3, "repeated") for i in range(3)],
        user_glossary=user,
        limit=2,
    )

    lines = md.splitlines()
    assert sum(1 for line in lines if "thuật ngữ người dùng" in line) == 5
    assert not any("dạng máy" in line for line in lines)  # no room left for machine terms


def test_a_long_document_keeps_more_heuristic_terms_than_a_memo():
    text = " ".join(f"Widget{i} appears twice. Widget{i} again." for i in range(120)) * 60
    budget = glossary_budget(len(text))

    memo = extract_glossary(text, max_terms=40)
    book = extract_glossary(text, max_terms=budget)

    assert len(memo) == 40
    assert len(book) == budget  # the memo's ceiling no longer binds a book-sized text
