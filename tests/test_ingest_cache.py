"""Tests for the cached extraction: a resume must not read the document again."""

from __future__ import annotations

import json

from polyglot_lc.ingest.ai_review import ReviewReport
from polyglot_lc.ingest.model import Block, Document
from polyglot_lc.llm.client import clear_mock_replies
from polyglot_lc.pipeline.ingest_cache import (
    INGEST_CACHE_VERSION,
    IngestCache,
    cache_key,
)
from polyglot_lc.segment.glossary import GlossaryTerm

DOC_TEXT = (
    "Data structures and algorithms are important tools. "
    "A linked list stores its elements in nodes.\n"
)


def _write(tmp_path, text: str = DOC_TEXT):
    """Input is PDF-only, so a document has to be rendered into a real one."""
    from pdfdoc import write_pdf

    return write_pdf(tmp_path / "doc.pdf", text)


def _document() -> Document:
    return Document(
        blocks=[
            Block(kind="heading", text="Chương 1", level=1),
            Block(kind="paragraph", text="Một đoạn văn."),
        ],
        metadata={"format": "pdf", "code_map": {"⟦CODE_1⟧": "```\nx = 1\n```"}},
    )


def _run(path, out, **kw):
    from polyglot_lc.pipeline import TranslationPipeline

    return TranslationPipeline(quiet=True).run(path, src="en", tgt="vi", out=str(out), **kw)


# ------------------------------- the cache ----------------------------------
def test_a_document_round_trips_through_the_cache(tmp_path):
    cache = IngestCache(tmp_path / "cp")
    doc = _document()
    terms = [GlossaryTerm(term="linked list", count=3, reason="repeated")]
    cache.save("abc123", doc, terms)

    loaded = cache.load("abc123")

    assert loaded is not None
    restored, restored_terms = loaded
    assert [block.text for block in restored.blocks] == [block.text for block in doc.blocks]
    assert restored.metadata["code_map"] == {"⟦CODE_1⟧": "```\nx = 1\n```"}
    assert [(term.term, term.count, term.reason) for term in restored_terms] == [
        ("linked list", 3, "repeated")
    ]


def test_another_content_digest_is_a_miss(tmp_path):
    cache = IngestCache(tmp_path / "cp")
    cache.save("abc123", _document(), [])
    assert cache.load("def456") is None


def test_an_entry_from_another_format_version_is_a_miss(tmp_path):
    cache = IngestCache(tmp_path / "cp")
    sha = "abc123"
    path = cache.root / f"{cache_key(sha)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": INGEST_CACHE_VERSION + 1,
                "sha256": sha,
                "doc": _document().model_dump(),
                "glossary_terms": [],
            }
        ),
        encoding="utf-8",
    )
    assert cache.load(sha) is None


def test_a_corrupt_entry_is_a_miss(tmp_path):
    cache = IngestCache(tmp_path / "cp")
    path = cache.root / f"{cache_key('abc123')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert cache.load("abc123") is None


def test_off_disables_the_cache(tmp_path):
    """``CHECKPOINT_DIR=off`` turns the cache off with it, not into ./."""
    cache = IngestCache("off")
    assert cache.usable is False
    cache.save("abc123", _document(), [])
    assert cache.load("abc123") is None
    assert not (tmp_path / ".ingest").exists()


# --------------------------- pipeline behaviour ------------------------------
def test_a_resume_reuses_the_cached_extraction(mock_env, tmp_path, monkeypatch):
    """The whole point: "Tiếp tục" must not extract the PDF again."""
    from polyglot_lc.pipeline import coordinator

    clear_mock_replies()
    path = _write(tmp_path)
    first = _run(path, tmp_path / "a.md")
    assert first.chunks >= 1

    def boom(*args, **kwargs):
        raise AssertionError("the document was read again on resume")

    monkeypatch.setattr(coordinator, "load_document", boom)
    again = _run(path, tmp_path / "b.md", resume=True)

    assert again.resumed_chunks == again.chunks


def test_a_resume_does_not_ask_the_ingest_review_again(mock_env, tmp_path, monkeypatch):
    """A review that ran again rewrote blocks — and with them every chunk hash."""
    from polyglot_lc.pipeline import coordinator

    monkeypatch.setenv("INGEST_AI_REVIEW", "1")
    calls: list[int] = []

    def fake_review(document, client, **kwargs):
        calls.append(1)
        return ReviewReport(candidates=1, reviewed=1)

    monkeypatch.setattr(coordinator, "review_with_llm", fake_review)
    clear_mock_replies()
    path = _write(tmp_path)

    _run(path, tmp_path / "a.md")
    assert len(calls) == 1

    _run(path, tmp_path / "b.md", resume=True)
    assert len(calls) == 1, "the review is frozen in the cache, not paid for twice"


def test_a_fresh_run_writes_the_cache(mock_env, tmp_path):
    from polyglot_lc.pipeline.ingest_cache import CACHE_DIRNAME

    clear_mock_replies()
    path = _write(tmp_path)
    _run(path, tmp_path / "a.md")

    entries = list((tmp_path / "checkpoints" / CACHE_DIRNAME).glob("*.json"))
    assert len(entries) == 1
