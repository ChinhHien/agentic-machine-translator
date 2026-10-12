"""Tests for per-chunk checkpoints and resume."""

from __future__ import annotations

import pytest

from polyglot_lc.llm.client import TokenUsage, clear_mock_replies
from polyglot_lc.pipeline.checkpoint import CheckpointStore, ChunkCheckpoint

DOC = (
    "Neural machine translation splits documents into segments.\n\n"
    "Each segment is translated with a consistent glossary.\n"
)


def _write(tmp_path, text: str = DOC, name: str = "doc.pdf"):
    """Input is PDF-only, so a document has to be rendered into a real one."""
    from pdfdoc import write_pdf

    return write_pdf(tmp_path / name, text)


def _key(path, *, src="en", tgt="vi", domain="general", model="mock-model", hashes=("h0", "h1")):
    return CheckpointStore.doc_key(
        path, src=src, tgt=tgt, domain=domain, model=model, chunk_hashes=list(hashes)
    )


# ------------------------------ store semantics -----------------------------
def test_put_get_roundtrip_preserves_fields(tmp_path):
    store = CheckpointStore(tmp_path / "cp")
    store.put(
        "key1",
        ChunkCheckpoint(
            "chunk-000", "h0", "bản dịch", score=9.5, path="fast", stored=True, calls=2
        ),
    )
    got = store.get("key1", "chunk-000", "h0")
    assert got is not None
    assert got.translated == "bản dịch"
    assert (got.score, got.path, got.stored, got.calls) == (9.5, "fast", True, 2)
    assert set(store.completed("key1")) == {"chunk-000"}


def test_get_ignores_a_stale_chunk_hash(tmp_path):
    store = CheckpointStore(tmp_path / "cp")
    store.put("key1", ChunkCheckpoint("chunk-000", "old-hash", "cũ"))
    assert store.get("key1", "chunk-000", "new-hash") is None
    assert store.get("key1", "chunk-000", "old-hash") is not None


def test_missing_checkpoint_and_clear(tmp_path):
    store = CheckpointStore(tmp_path / "cp")
    assert store.get("nope", "chunk-000") is None
    assert store.completed("nope") == {}
    store.put("key1", ChunkCheckpoint("chunk-000", "h0", "x"))
    store.clear("key1")
    assert store.get("key1", "chunk-000", "h0") is None


def test_disabled_store_is_inert(tmp_path):
    store = CheckpointStore(tmp_path / "cp", enabled=False)
    store.put("key1", ChunkCheckpoint("chunk-000", "h0", "x"))
    assert store.usable is False
    assert not (tmp_path / "cp").exists()


def test_off_root_disables_the_store():
    # An empty/off root must not silently write into the working directory.
    assert CheckpointStore("off").enabled is False
    assert CheckpointStore("").enabled is False


def test_unwritable_root_degrades_instead_of_raising(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    store = CheckpointStore(blocker / "cp")
    store.put("key1", ChunkCheckpoint("chunk-000", "h0", "x"))  # no exception
    assert store.usable is False


def test_doc_key_changes_when_the_job_changes(tmp_path):
    path = _write(tmp_path)
    base = _key(path)
    assert base == _key(path)  # stable
    assert base != _key(path, model="other-model")
    assert base != _key(path, tgt="fr")
    assert base != _key(path, hashes=("h0", "different"))  # re-chunked
    _write(tmp_path, DOC + "\nOne more paragraph.\n")
    assert base != _key(path)  # edited document


# -------------------------------- end to end --------------------------------
def _run(path, **kw):
    from polyglot_lc.pipeline import TranslationPipeline

    return TranslationPipeline(quiet=True).run(path, src="en", tgt="vi", **kw)


def test_full_resume_translates_nothing_again(mock_env, tmp_path):
    clear_mock_replies()
    path = _write(tmp_path)
    first = _run(path, out=str(tmp_path / "a.md"))
    assert first.chunks >= 1
    assert first.resumed_chunks == 0

    TokenUsage.reset()
    second = _run(path, out=str(tmp_path / "b.md"), resume=True)

    assert second.resumed_chunks == second.chunks
    assert TokenUsage.snapshot()["calls"] == 0  # zero new requests
    assert (tmp_path / "a.md").read_text(encoding="utf-8") == (tmp_path / "b.md").read_text(
        encoding="utf-8"
    )


def test_resume_without_checkpoints_re_translates(mock_env, tmp_path):
    clear_mock_replies()
    path = _write(tmp_path)
    TokenUsage.reset()
    result = _run(path, out=str(tmp_path / "a.md"), resume=True)
    assert result.resumed_chunks == 0
    assert TokenUsage.snapshot()["calls"] > 0


def test_editing_the_document_invalidates_checkpoints(mock_env, tmp_path):
    clear_mock_replies()
    path = _write(tmp_path)
    _run(path, out=str(tmp_path / "a.md"))
    _write(tmp_path, DOC + "\nA brand new paragraph appears here.\n")

    TokenUsage.reset()
    again = _run(path, out=str(tmp_path / "b.md"), resume=True)
    assert again.resumed_chunks == 0
    assert TokenUsage.snapshot()["calls"] > 0


def _chunks_of(path, max_tokens=15):
    from polyglot_lc.ingest import load_document
    from polyglot_lc.segment.chunker import chunk_document

    return chunk_document(load_document(path), max_tokens=max_tokens, model="mock-model")


def test_a_crashed_run_resumes_the_chunks_it_finished(mock_env, tmp_path, monkeypatch):
    """Crash durability: a later chunk failing must not lose the earlier ones."""
    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.translator import TranslatorAgent
    from polyglot_lc.pipeline import coordinator

    monkeypatch.setenv("MAX_CONCURRENCY", "1")  # deterministic chunk order
    monkeypatch.setenv("PARTIAL_OUTPUT", "1")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")  # isolate resume semantics
    clear_mock_replies()
    path = _write(tmp_path, DOC + "\nA third paragraph to translate as well.\n")
    chunks = _chunks_of(path)
    assert len(chunks) >= 3

    def fake_translate(self, prompt, chunk_id=""):
        return "T::" + extract_source(prompt)

    def flaky_translate(self, prompt, chunk_id=""):
        if chunk_id == chunks[-1].id:
            raise RuntimeError("provider outage")
        return fake_translate(self, prompt, chunk_id)

    snapshots: list[str] = []
    original_write = coordinator.write_atomic

    def record(target, text):
        snapshots.append(text)
        original_write(target, text)

    monkeypatch.setattr(coordinator, "write_atomic", record)
    monkeypatch.setattr(TranslatorAgent, "translate", flaky_translate)
    with pytest.raises(RuntimeError, match="provider outage"):
        _run(path, out=str(tmp_path / "crashed.md"), max_tokens=15)
    # PARTIAL_OUTPUT leaves an explicitly partial file (clearly not a finished
    # translation) so progress is visible; without it, a failed run writes none
    # at all (see test_chunk_failure_propagates_under_concurrency).
    assert "pending" in (tmp_path / "crashed.md").read_text(encoding="utf-8")

    # Every partial write must attribute a chunk's text to that chunk's own slot.
    for snapshot in snapshots:
        first = snapshot.split("\n\n")[0]
        assert first in {f"<!-- {chunks[0].id}: pending -->", "T::" + chunks[0].text}

    monkeypatch.setattr(TranslatorAgent, "translate", fake_translate)
    TokenUsage.reset()
    second = _run(path, out=str(tmp_path / "resumed.md"), max_tokens=15, resume=True)

    assert second.resumed_chunks == len(chunks) - 1
    assert second.planner_calls == 1
    # Exactly one chunk re-ran its graph. Its draft literally repeats the source
    # text, so the leakage signal must refuse the fast path for it.
    by_agent = TokenUsage.snapshot_by_agent()
    assert by_agent.get("VERIFIER", {}).get("calls", 0) == 1
    assert by_agent.get("CRITIC", {}).get("calls", 0) == 1
    # The resumed output is the complete document, in the original order.
    assert (tmp_path / "resumed.md").read_text(encoding="utf-8") == "\n\n".join(
        "T::" + c.text for c in chunks
    )


def test_an_exhausted_rate_limit_costs_one_chunk_not_the_run(mock_env, tmp_path, monkeypatch):
    """A 429 that survives every retry must not abort the whole run.

    Provider failures are contained per chunk: the run finishes, the chunk's
    place is marked in the output, and the failure is reported. A lost section
    must not become a lost document.
    """
    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.translator import TranslatorAgent
    from polyglot_lc.llm.resilience import RateLimitError

    monkeypatch.setenv("MAX_CONCURRENCY", "1")  # deterministic chunk order
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")
    clear_mock_replies()
    path = _write(tmp_path, DOC + "\nA third paragraph to translate as well.\n")
    chunks = _chunks_of(path)
    assert len(chunks) >= 3
    doomed = chunks[1].id

    def flaky_translate(self, prompt, chunk_id=""):
        if chunk_id == doomed:
            raise RateLimitError(
                "Error code: 429 - concurrency_limit_exceeded: Too many "
                "concurrent requests (limit: 1)",
                status=429,
            )
        return "T::" + extract_source(prompt)

    monkeypatch.setattr(TranslatorAgent, "translate", flaky_translate)
    out = tmp_path / "one-chunk-lost.md"
    result = _run(path, out=str(out), max_tokens=15)  # must not raise

    text = out.read_text(encoding="utf-8")
    assert f"<!-- {doomed}: untranslated" in text
    for chunk in chunks:
        if chunk.id != doomed:
            assert "T::" + chunk.text in text
    assert any(doomed in error for error in result.errors), result.errors


def test_partial_output_is_written_while_chunks_are_pending(mock_env, tmp_path, monkeypatch):
    monkeypatch.setenv("PARTIAL_OUTPUT", "1")
    clear_mock_replies()
    # Several small chunks, so some are still in flight when others finish.
    path = _write(tmp_path, DOC + "\nA third paragraph to translate as well.\n")
    out = tmp_path / "partial.md"

    snapshots: list[str] = []

    from polyglot_lc.pipeline import coordinator

    original_write = coordinator.write_atomic

    def record(target, text):
        snapshots.append(text)
        original_write(target, text)

    monkeypatch.setattr(coordinator, "write_atomic", record)
    result = _run(path, out=str(out), max_tokens=15)

    assert result.chunks > 1
    assert result.partial_output is True
    assert snapshots, "partial writes must happen as chunks finish"
    # At least one intermediate write shows the chunks still in flight...
    assert any("pending" in snapshot for snapshot in snapshots)
    # ...and the last write is the complete document.
    assert "pending" not in snapshots[-1]
    assert snapshots[-1] == out.read_text(encoding="utf-8")
