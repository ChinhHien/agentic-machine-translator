"""The chunk split must not move unless the document or the budget does.

``CheckpointStore.doc_key`` contains ``chunk_hashes``: changing a boundary changes
the identity of the run, i.e. it throws away the checkpoints of a document that is
half translated. So everything that is not the text or the budget — the model, its
endpoint, a context window — must leave the chunks byte-for-byte identical.

The context-window budget that used to live in the chunker is gone for this
reason: no trustworthy per-model window exists here, and shrinking chunks per
model moved every boundary. The budget is ``CHUNK_MAX_TOKENS``, counted with the
pinned tokenizer.
"""

from __future__ import annotations

from polyglot_lc.ingest.model import Block, Document
from polyglot_lc.segment.chunker import chunk_document

BASE = "http://mock.local/v1"
MODEL = "mock-model"


def _doc(paragraphs: int = 12) -> Document:
    return Document(
        blocks=[Block(text=f"Paragraph {i}. " + "content " * 40) for i in range(paragraphs)]
    )


def _shape(chunks) -> list[tuple[str, str]]:
    return [(c.text, c.hash()) for c in chunks]


# ---------------------------------------------------------------------------
# invariance: the split must not move unless it has to
# ---------------------------------------------------------------------------
def test_no_profile_leaves_the_split_untouched(mock_env):
    doc = _doc()
    baseline = chunk_document(doc, max_tokens=60, overlap_tokens=0, model=MODEL)
    assert _shape(
        chunk_document(doc, max_tokens=60, overlap_tokens=0, model=MODEL, base_url=BASE)
    ) == _shape(baseline)


# ---------------------------------------------------------------------------
# the UI preview must agree with the run
# ---------------------------------------------------------------------------
def test_the_preview_and_the_run_split_alike(mock_env, sample_pdf_file, monkeypatch):
    """``probe.py --plan`` is what the reader sees before spending anything."""
    from webui.probe import plan_file

    preview = plan_file(sample_pdf_file)
    from polyglot_lc.ingest import load_document

    run = chunk_document(
        load_document(sample_pdf_file),
        max_tokens=700,
        model=MODEL,
        base_url=BASE,
    )
    assert preview["chunks"] == len(run)
