"""Tests for per-chunk checkpoints and resume."""

from __future__ import annotations

from pathlib import Path

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


def _key(path, *, src="en", tgt="vi", model="mock-model", hashes=("h0", "h1"), glossary=""):
    return CheckpointStore.doc_key(
        path,
        src=src,
        tgt=tgt,
        model=model,
        chunk_hashes=list(hashes),
        glossary=glossary,
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


def test_verification_failure_preserves_only_completed_checkpoints(mock_env, tmp_path):
    from types import SimpleNamespace

    from polyglot_lc.agents.verifier import VerificationError
    from polyglot_lc.pipeline import TranslationPipeline
    from polyglot_lc.segment.chunker import Chunk

    chunks = [
        Chunk(id=f"c{i}", text=f"Source {i}.", token_count=3, section_path=[], index=i)
        for i in range(3)
    ]
    asked = []

    def invoke(state, **kwargs):
        chunk = state["chunk"]
        asked.append(chunk.id)
        if chunk.id == "c1":
            raise VerificationError("verification unavailable")
        return {"translated": "Bản dịch đã kiểm định.", "chunk": chunk}

    store = CheckpointStore(tmp_path / "cp")
    pipeline = TranslationPipeline(quiet=True)
    with pytest.raises(VerificationError):
        pipeline._run_chunks(
            SimpleNamespace(invoke=invoke),
            chunks,
            {},
            SimpleNamespace(advance=lambda: None),
            store=store,
            doc_key="document",
        )
    assert asked == ["c0", "c1"]
    assert set(store.completed("document")) == {"c0"}


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


def test_doc_key_changes_when_the_glossary_changes(tmp_path):
    """Terminology is part of the job: a chunk translated without a term the
    reader has since added is not the same deliverable."""
    path = _write(tmp_path)
    base = _key(path)
    assert _key(path, glossary="NMT\tDịch máy neural") != base
    assert _key(path, glossary="NMT\tDịch máy neural") != _key(
        path, glossary="NMT\tDịch máy nơ-ron"
    )
    assert base == _key(path, glossary="")


def test_doc_key_ignores_a_rewrite_that_changes_no_bytes(tmp_path):
    """Re-uploading a document must not orphan chunks that are still valid.

    The web UI writes a file the reader picked up again — same name, same
    bytes — and that write lands a fresh ``mtime`` on it. Keyed on
    ``size:mtime``, that alone minted a new key, so "Tiếp tục" matched nothing
    and translated the whole document again from its first chunk.
    """
    import os

    path = _write(tmp_path)
    base = _key(path)

    os.utime(path, (9_000_000_000, 9_000_000_000))  # same bytes, other mtime
    assert _key(path) == base

    path.write_bytes(path.read_bytes() + b"\n")  # a real edit still counts
    assert _key(path) != base


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


def test_resume_does_not_trust_legacy_checkpoints(mock_env, tmp_path, monkeypatch):
    import hashlib

    from polyglot_lc.pipeline import coordinator

    monkeypatch.setattr(coordinator, "default_glossary", lambda src, tgt: {})
    clear_mock_replies()
    path = _write(tmp_path)
    chunks = _chunks_of(path, max_tokens=1000)
    digest = hashlib.sha1()
    for part in (
        str(path.resolve()),
        "en",
        "vi",
        "mock-model",
        *(chunk.hash() for chunk in chunks),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    legacy_key = digest.hexdigest()[:16]
    store = CheckpointStore(tmp_path / "checkpoints")
    for chunk in chunks:
        store.put(legacy_key, ChunkCheckpoint(chunk.id, chunk.hash(), "Legacy unchecked text"))
    snapshots = {file: file.read_bytes() for file in store.dir_for(legacy_key).glob("*.json")}

    result = _run(path, out=str(tmp_path / "checked.md"), max_tokens=1000, resume=True)

    assert result.resumed_chunks == 0
    assert "Legacy unchecked text" not in (tmp_path / "checked.md").read_text(encoding="utf-8")
    assert all(file.read_bytes() == data for file, data in snapshots.items())
    resumed = _run(path, out=str(tmp_path / "resumed.md"), max_tokens=1000, resume=True)
    assert resumed.resumed_chunks == resumed.chunks


def test_enabling_full_verification_invalidates_fast_skip_checkpoints(
    mock_env, tmp_path, monkeypatch
):
    clear_mock_replies()
    path = _write(tmp_path)
    monkeypatch.setenv("FAST_PATH_SKIP_LLM_VERIFY", "1")
    first = _run(path, out=str(tmp_path / "fast.md"))
    assert first.chunks > 0

    monkeypatch.setenv("FAST_PATH_SKIP_LLM_VERIFY", "0")
    checked = _run(path, out=str(tmp_path / "checked.md"), resume=True)
    assert checked.resumed_chunks == 0


def test_verification_failure_records_finished_chunks_and_can_resume(
    mock_env, tmp_path, monkeypatch
):
    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.translator import TranslatorAgent
    from polyglot_lc.agents.verifier import VerificationError, VerifierAgent, VerifyResult
    from polyglot_lc.pipeline import TranslationPipeline

    monkeypatch.setenv("MAX_CONCURRENCY", "1")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")
    monkeypatch.setenv("LOG_JSONL", "1")
    clear_mock_replies()
    path = _write(tmp_path, DOC + "\nA third paragraph to translate as well.\n")
    chunks = _chunks_of(path)
    assert len(chunks) >= 3
    monkeypatch.setattr(
        TranslatorAgent, "translate", lambda self, prompt, **kw: extract_source(prompt)
    )

    def verify(self, source, *args, **kwargs):
        if source == chunks[1].text:
            raise VerificationError("verification unavailable")
        return VerifyResult(ok=True, reasons=[])

    monkeypatch.setattr(VerifierAgent, "verify", verify)
    pipeline = TranslationPipeline(quiet=True)
    with pytest.raises(VerificationError):
        pipeline.run(path, src="en", tgt="vi", out=str(tmp_path / "stopped.md"), max_tokens=15)

    end = _records_of(pipeline.sink.path, "run_end")[-1]
    assert end["status"] == "failed"
    assert end["chunks_done"] == 1
    assert end["chunks_failed"] == len(chunks) - 1
    assert end["verify_skipped"] == 1
    named = _records_of(pipeline.sink.path, "checkpoint")[0]
    stored = CheckpointStore(tmp_path / "checkpoints").completed(named["doc_key"])
    assert set(stored) == {chunks[0].id}

    monkeypatch.setattr(VerifierAgent, "verify", lambda *a, **kw: VerifyResult(ok=True, reasons=[]))
    resumed = _run(path, out=str(tmp_path / "resumed.md"), max_tokens=15, resume=True)
    assert resumed.resumed_chunks == 1
    assert resumed.chunks_failed == 0


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("rejected_indices", [(1,), (0, 2), None])
def test_quality_rejections_are_isolated_and_resume_only_missing_chunks(
    mock_env, tmp_path, monkeypatch, workers, rejected_indices
):
    from collections import Counter

    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.translator import TranslatorAgent
    from polyglot_lc.agents.verifier import VerifierAgent, VerifyResult
    from polyglot_lc.pipeline.consistency import ConsistencyChecker

    monkeypatch.setenv("MAX_CONCURRENCY", str(workers))
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")
    monkeypatch.setenv("LOG_JSONL", "1")
    clear_mock_replies()
    path = _write(tmp_path, DOC + "\nA third paragraph to translate as well.\n")
    chunks = _chunks_of(path)
    assert len(chunks) >= 3
    rejected = {
        chunks[i].id for i in (range(len(chunks)) if rejected_indices is None else rejected_indices)
    }
    rejected_sources = {chunk.text for chunk in chunks if chunk.id in rejected}
    asked = []

    def translate(self, prompt, chunk_id=""):
        asked.append(chunk_id)
        return "DRAFT " + extract_source(prompt)

    def verify(self, source, *args, **kwargs):
        return VerifyResult(ok=source not in rejected_sources, reasons=["content missing"])

    original_check = ConsistencyChecker.run

    def check(self, checked_chunks, translations, *args, **kwargs):
        assert not rejected.intersection(chunk.id for chunk in checked_chunks)
        return original_check(self, checked_chunks, translations, *args, **kwargs)

    monkeypatch.setattr(TranslatorAgent, "translate", translate)
    monkeypatch.setattr(VerifierAgent, "verify", verify)
    monkeypatch.setattr(ConsistencyChecker, "run", check)
    target = tmp_path / "incomplete.md"
    result = _run(path, out=str(target), max_tokens=15)

    assert Counter(asked) == Counter({c.id: 2 if c.id in rejected else 1 for c in chunks})
    # Best-effort output: every chunk has text; rejected chunks carry a review
    # note instead of a hole and are not checkpointed for resume.
    assert result.chunks_failed == 0
    assert result.chunks_rejected == len(rejected)
    assert result.fatal_error == ""
    assert result.verify_skipped == 0
    assert all(any(cid in error for error in result.errors) for cid in rejected)
    text = target.read_text(encoding="utf-8")
    for chunk in chunks:
        if chunk.id in rejected:
            assert "DRAFT " + chunk.text in text
        else:
            assert "DRAFT " + chunk.text in text
    # Rejected chunks still ship their best draft; the verdict is reported in
    # the run summary, never written into the document itself.
    assert "needs review" not in text
    end = _records_of(Path(result.log_path), "run_end")[-1]
    assert end["status"] == "done_with_warnings"
    assert end["exit_code"] == 0
    assert end["chunks_done"] == len(chunks)
    assert end["chunks_failed"] == 0
    assert end["chunks_rejected"] == len(rejected)
    assert end["fatal_error"] == ""
    named = _records_of(Path(result.log_path), "checkpoint")[0]
    store = CheckpointStore(tmp_path / "checkpoints")
    assert set(store.completed(named["doc_key"])) == {c.id for c in chunks} - rejected
    snapshots = {
        file: file.read_bytes() for file in store.dir_for(named["doc_key"]).glob("chunk-*.json")
    }

    asked.clear()
    monkeypatch.setattr(VerifierAgent, "verify", lambda *a, **kw: VerifyResult(True, []))
    monkeypatch.setattr(ConsistencyChecker, "run", original_check)
    resumed = _run(path, out=str(target), max_tokens=15, resume=True)

    assert Counter(asked) == Counter({cid: 1 for cid in rejected})
    assert resumed.resumed_chunks == len(chunks) - len(rejected)
    assert resumed.chunks_failed == resumed.chunks_rejected == 0
    assert all(file.read_bytes() == data for file, data in snapshots.items())
    assert "needs review" not in target.read_text(encoding="utf-8")
    assert _records_of(Path(resumed.log_path), "run_end")[-1]["status"] == "done"


def test_a_passing_verdict_with_praise_reasons_accepts_every_chunk(mock_env, tmp_path, monkeypatch):
    """A pass that names no defect must ship: the reasons fallback once turned
    the verifier's own praise into a 'major' defect and rejected every chunk."""
    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.translator import TranslatorAgent
    from polyglot_lc.agents.verifier import VerifierAgent

    monkeypatch.setenv("MAX_CONCURRENCY", "1")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")
    monkeypatch.setenv("CONSISTENCY_CHECK", "0")
    monkeypatch.setenv("FAST_PATH_SKIP_LLM_VERIFY", "0")
    monkeypatch.setenv("LOG_JSONL", "1")
    clear_mock_replies()
    path = _write(tmp_path, DOC + "\nA third paragraph to translate as well.\n")
    chunks = _chunks_of(path)
    assert len(chunks) >= 3

    monkeypatch.setattr(
        TranslatorAgent,
        "translate",
        lambda self, prompt, chunk_id="": "DRAFT " + extract_source(prompt),
    )
    monkeypatch.setattr(
        VerifierAgent,
        "chat_json",
        lambda *a, **kw: {
            "pass": True,
            "defects": [],
            "reasons": ["All checks passed: complete, natural Vietnamese, structure intact."],
        },
    )
    target = tmp_path / "praised.md"
    result = _run(path, out=str(target), max_tokens=15)

    assert result.chunks_rejected == 0
    assert result.chunks_failed == 0
    assert result.fatal_error == ""
    assert not any("verification failed" in error for error in result.errors)
    end = _records_of(Path(result.log_path), "run_end")[-1]
    assert end["chunks_rejected"] == 0
    assert end["status"] == "done"
    text = target.read_text(encoding="utf-8")
    for chunk in chunks:
        assert "DRAFT " + chunk.text in text


def test_a_glossary_term_hidden_inside_longer_words_costs_no_retranslation(
    mock_env, tmp_path, monkeypatch
):
    """A real 609-chunk run rejected 27 chunks demanding 'thấp' for 'low'.

    The source only ever said 'below'/'allow'/'flow'. The verdict on record was
    ``pass=true`` with no defects; the deterministic glossary check invented the
    failure, every chunk burned its single retranslate on a term it did not
    contain, and the run logged "verification failed" quoting the verifier's
    praise instead of a fault.
    """
    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.translator import TranslatorAgent
    from polyglot_lc.agents.verifier import VerifierAgent

    monkeypatch.setenv("MAX_CONCURRENCY", "1")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")
    monkeypatch.setenv("CONSISTENCY_CHECK", "0")
    monkeypatch.setenv("FAST_PATH_SKIP_LLM_VERIFY", "0")
    monkeypatch.setenv("LOG_JSONL", "1")
    clear_mock_replies()
    path = _write(
        tmp_path,
        "The values below are listed in the table. " * 6
        + "\n\nAllow the flow to continue and update the candidate list. " * 6
        + "\n\nA higher value follows the same rule. " * 6,
    )
    csv = tmp_path / "glossary.csv"
    csv.write_text("low,thấp\ndate,ngày\nhigh,cao\n", encoding="utf-8")

    monkeypatch.setattr(
        TranslatorAgent,
        "translate",
        lambda self, prompt, chunk_id="": "DRAFT " + extract_source(prompt),
    )
    monkeypatch.setattr(
        VerifierAgent,
        "chat_json",
        lambda *a, **kw: {"pass": True, "defects": [], "reasons": ["All checks passed."]},
    )
    target = tmp_path / "glossary.md"
    result = _run(path, out=str(target), max_tokens=15, glossary=str(csv))

    assert result.chunks >= 1
    assert result.chunks_rejected == 0
    assert result.chunks_failed == 0
    assert not any("verification failed" in error for error in result.errors)
    decisions = _records_of(Path(result.log_path), "decision")
    assert not [
        d for d in decisions if d.get("kind") == "retranslate" and d.get("chosen")
    ], "no chunk may be sent back for a term its source does not contain"
    assert "DRAFT The values below" in target.read_text(encoding="utf-8")


def test_a_real_glossary_miss_is_logged_as_the_miss_not_as_praise(
    mock_env, tmp_path, monkeypatch
):
    """When a chunk genuinely is rejected, the log must say what is wrong.

    The run's error record used to read "verification failed after
    retranslation: All source segments are present..." — the verifier's own
    summary of the checks it cleared, with the actual fault buried past the
    three reasons the graph quotes.
    """
    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.translator import TranslatorAgent
    from polyglot_lc.agents.verifier import VerifierAgent

    monkeypatch.setenv("MAX_CONCURRENCY", "1")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")
    monkeypatch.setenv("CONSISTENCY_CHECK", "0")
    monkeypatch.setenv("FAST_PATH_SKIP_LLM_VERIFY", "0")
    monkeypatch.setenv("LOG_JSONL", "1")
    clear_mock_replies()
    path = _write(tmp_path, "The impact level is low. " * 12)
    csv = tmp_path / "glossary.csv"
    csv.write_text("low,thấp\n", encoding="utf-8")

    monkeypatch.setattr(
        TranslatorAgent,
        "translate",
        lambda self, prompt, chunk_id="": "DRAFT " + extract_source(prompt),
    )
    monkeypatch.setattr(
        VerifierAgent,
        "chat_json",
        lambda *a, **kw: {
            "pass": True,
            "defects": [],
            "reasons": [
                "Completeness: all source sentences are represented.",
                "No additions: no invented content detected.",
                "Language: reads naturally.",
            ],
        },
    )
    result = _run(path, out=str(tmp_path / "miss.md"), max_tokens=15, glossary=str(csv))

    assert result.chunks_rejected >= 1
    records = _records_of(Path(result.log_path), "error")
    failures = [
        str(r.get("message")) for r in records if "verification failed" in str(r.get("message"))
    ]
    assert failures, records
    assert "thấp" in failures[0]
    head = failures[0].split("retranslation:", 1)[-1].lstrip()
    assert head.startswith("glossary term"), head


def test_resume_without_checkpoints_re_translates(mock_env, tmp_path):
    clear_mock_replies()
    path = _write(tmp_path)
    TokenUsage.reset()
    result = _run(path, out=str(tmp_path / "a.md"), resume=True)
    assert result.resumed_chunks == 0
    assert TokenUsage.snapshot()["calls"] > 0


def test_resuming_with_a_new_glossary_asks_the_chunks_again(mock_env, tmp_path):
    """A changed CSV must not be papered over by a checkpoint hit.

    The terminology is part of the checkpoint's identity, so the chunks
    translated without the new term are asked again instead of being replayed
    and leaving that term out of the file.
    """
    clear_mock_replies()
    path = _write(tmp_path)
    csv_a = tmp_path / "a.csv"
    csv_a.write_text("NMT,Dịch máy neural\n", encoding="utf-8")
    csv_b = tmp_path / "b.csv"
    csv_b.write_text("NMT,Dịch máy nơ-ron\n", encoding="utf-8")

    first = _run(path, out=str(tmp_path / "a.md"), glossary=str(csv_a))
    assert first.chunks >= 1

    same = _run(path, out=str(tmp_path / "b.md"), resume=True, glossary=str(csv_a))
    assert same.resumed_chunks == same.chunks

    changed = _run(path, out=str(tmp_path / "c.md"), resume=True, glossary=str(csv_b))
    assert changed.resumed_chunks == 0


def test_the_resume_record_makes_a_replay_visible_early(mock_env, tmp_path):
    """A resumed run's replayed chunks leave no per-chunk events in its own
    log, so the web UI needs one early record carrying how many chunks are
    being replayed and whether the store had anything to match — without it the
    resumed run's progress bar reads 0/N and a checkpoint miss looks like a
    silent start-over."""
    import os

    from polyglot_lc.pipeline import TranslationPipeline

    os.environ["LOG_JSONL"] = "1"
    clear_mock_replies()
    path = _write(tmp_path)

    os.environ["LOGS_DIR"] = str(tmp_path / "first")
    first = TranslationPipeline(quiet=True).run(
        path, src="en", tgt="vi", out=str(tmp_path / "a.md")
    )
    assert first.resumed_chunks == 0
    log = _first_line_types(Path(first.log_path))
    assert log and log["resume_requested"] is False

    os.environ["LOGS_DIR"] = str(tmp_path / "second")
    second = TranslationPipeline(quiet=True).run(
        path, src="en", tgt="vi", out=str(tmp_path / "b.md"), resume=True
    )
    assert second.resumed_chunks == second.chunks
    log = _first_line_types(Path(second.log_path))
    assert log["resume_requested"] is True and log["resumed_chunks"] == second.chunks


def _records_of(path, kind: str) -> list[dict]:
    """Every record of one type, in the run's own log."""
    import json

    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == kind:
            out.append(record)
    return out


def _first_line_types(path) -> dict:
    """The run's own early `resume` record, read from its log."""
    import json

    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "resume":
            return record
    raise AssertionError("no resume record in the run's log")


def test_the_run_names_the_directory_its_chunks_are_written_to(mock_env, tmp_path, monkeypatch):
    """The page shows the translation as it lands, out of the chunks' own files
    on disk — and it cannot derive where those live, because the checkpoint
    directory is the pipeline's configuration and the web UI is not allowed to
    read that. So the run names its own directory, once, before the first chunk
    is written; a run that keeps no checkpoints says nothing at all, which reads
    as "there is nothing to watch".
    """
    monkeypatch.setenv("LOG_JSONL", "1")
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "cp"))
    clear_mock_replies()
    path = _write(tmp_path)
    out = tmp_path / "a.md"

    result = _run(path, out=str(out))

    named = _records_of(Path(result.log_path), "checkpoint")
    assert len(named) == 1, "one attempt names its directory once"
    directory = Path(named[0]["path"])
    assert directory.is_dir() and named[0]["doc_key"] == directory.name
    written = {item.stem for item in directory.glob("*.json") if item.name != "meta.json"}
    assert written == {f"chunk-{index:03d}" for index in range(result.chunks)}


def test_a_run_without_checkpoints_names_no_directory(mock_env, tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DIR", "off")
    monkeypatch.setenv("LOG_JSONL", "1")
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    clear_mock_replies()

    result = _run(_write(tmp_path), out=str(tmp_path / "a.md"))

    assert _records_of(Path(result.log_path), "checkpoint") == []


def test_editing_the_document_invalidates_checkpoints(mock_env, tmp_path):
    clear_mock_replies()
    path = _write(tmp_path)
    _run(path, out=str(tmp_path / "a.md"))
    _write(tmp_path, DOC + "\nA brand new paragraph appears here.\n")

    TokenUsage.reset()
    again = _run(path, out=str(tmp_path / "b.md"), resume=True)
    assert again.resumed_chunks == 0
    assert TokenUsage.snapshot()["calls"] > 0


def test_re_uploading_the_same_document_still_resumes_from_checkpoint(mock_env, tmp_path):
    """The reader uploads a file they already uploaded, then presses "Tiếp tục".

    Nothing about the document changed, so every finished chunk is still valid
    and the resume must replay them instead of paying for the whole document
    again. Only the file's ``mtime`` moved, and that is not the document.
    """
    import os

    clear_mock_replies()
    path = _write(tmp_path)
    first = _run(path, out=str(tmp_path / "a.md"))
    assert first.chunks >= 1 and first.resumed_chunks == 0

    os.utime(path, (9_000_000_000, 9_000_000_000))  # the re-upload's only trace

    TokenUsage.reset()
    second = _run(path, out=str(tmp_path / "b.md"), resume=True)

    assert second.resumed_chunks == second.chunks
    assert TokenUsage.snapshot()["calls"] == 0  # nothing re-translated
    assert (tmp_path / "a.md").read_text(encoding="utf-8") == (tmp_path / "b.md").read_text(
        encoding="utf-8"
    )


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


def test_a_rate_limited_chunk_is_waited_out_not_left_behind(mock_env, tmp_path, monkeypatch):
    """A 429 that survives every retry must not cost the chunk, or the run.

    The resilience layer has already spent its retries on the request by the
    time a chunk fails, so the chunk waits for the provider to come back and is
    run again — as often as it takes. The file a finished run writes holds the
    chunk's translation and no placeholder, and the run reports done: what it
    waited out is a problem in its log, not a lost section.
    """
    import json

    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.translator import TranslatorAgent
    from polyglot_lc.llm.resilience import RateLimitError

    monkeypatch.setenv("MAX_CONCURRENCY", "1")  # deterministic chunk order
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")
    monkeypatch.setenv("CHUNK_RETRY_BASE_DELAY", "0.01")  # the wait is not what is under test
    monkeypatch.setenv("CHUNK_RETRY_MAX_DELAY", "0.02")
    monkeypatch.setenv("LOG_JSONL", "1")
    clear_mock_replies()
    path = _write(tmp_path, DOC + "\nA third paragraph to translate as well.\n")
    chunks = _chunks_of(path)
    assert len(chunks) >= 3
    doomed = chunks[1].id
    attempts = {"n": 0}

    def flaky_translate(self, prompt, chunk_id=""):
        if chunk_id == doomed:
            attempts["n"] += 1
            if attempts["n"] <= 5:  # more attempts than any request-level budget
                raise RateLimitError(
                    "Error code: 429 - concurrency_limit_exceeded: Too many "
                    "concurrent requests (limit: 1)",
                    status=429,
                )
        return "T::" + extract_source(prompt)

    monkeypatch.setattr(TranslatorAgent, "translate", flaky_translate)
    out = tmp_path / "one-chunk-saved.md"
    result = _run(path, out=str(out), max_tokens=15)  # must not raise

    text = out.read_text(encoding="utf-8")
    assert "untranslated" not in text
    for chunk in chunks:
        assert "T::" + chunk.text in text
    # The chunk really was run again five times over: no ceiling at three.
    assert attempts["n"] == 6
    assert result.chunks_failed == 0
    assert any(doomed in error for error in result.errors), result.errors

    records = [
        json.loads(line) for line in Path(result.log_path).read_text(encoding="utf-8").splitlines()
    ]
    waits = [record for record in records if record.get("type") == "retry_wait"]
    assert len(waits) == 5
    assert {wait["chunk_id"] for wait in waits} == {doomed}
    # The wait says where the chunk died and why it is waiting.
    assert waits[0]["step"] == "translate"
    assert "429" in waits[0]["error"]
    # And the run's own verdict is about the file it wrote, not about the road.
    end = [record for record in records if record.get("type") == "run_end"][-1]
    assert (end["status"], end["exit_code"]) == ("done", 0)
    assert end["chunks_failed"] == 0
    assert end["chunk_retries"] == 5


def test_a_chunk_with_no_backup_left_stops_the_run_instead_of_waiting_forever(
    mock_env, tmp_path, monkeypatch
):
    """The attempts are spent and there is no backup: the run ends and says so.

    Waiting on an endpoint that has already said no is how a run used to sit at
    "0/38" for an hour. The chunk is asked once, the run stops, the chunks
    after it are not asked at all, and the record says what happened.
    """
    import json

    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.translator import TranslatorAgent
    from polyglot_lc.llm.resilience import NoBackupError

    monkeypatch.setenv("MAX_CONCURRENCY", "1")  # deterministic chunk order
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")
    monkeypatch.setenv("CHUNK_RETRY_BASE_DELAY", "0.01")
    monkeypatch.setenv("CHUNK_RETRY_MAX_DELAY", "0.02")
    monkeypatch.setenv("LOG_JSONL", "1")
    clear_mock_replies()
    path = _write(tmp_path, DOC + "\nA third paragraph to translate as well.\n")
    chunks = _chunks_of(path)
    assert len(chunks) >= 3
    doomed = chunks[1].id
    attempts = {"n": 0}

    def dying_translate(self, prompt, chunk_id=""):
        if chunk_id == doomed:
            attempts["n"] += 1
            raise NoBackupError("no backup endpoint is configured: 429")
        return "T::" + extract_source(prompt)

    monkeypatch.setattr(TranslatorAgent, "translate", dying_translate)
    out = tmp_path / "stopped.md"
    result = _run(path, out=str(out), max_tokens=15)  # must not raise

    # One attempt, not a wait: the chunk was asked once and the run gave up.
    assert attempts["n"] == 1
    assert result.chunks_failed >= 1
    assert any("no backup endpoint" in error for error in result.errors), result.errors
    records = [
        json.loads(line) for line in Path(result.log_path).read_text(encoding="utf-8").splitlines()
    ]
    # No chunk was waited out: the run did not sit on the dead endpoint.
    assert not [record for record in records if record.get("type") == "retry_wait"]
    end = [record for record in records if record.get("type") == "run_end"][-1]
    assert (end["status"], end["exit_code"]) == ("failed", 1)


def test_an_interrupted_run_is_not_recorded_as_a_finished_one(mock_env, tmp_path, monkeypatch):
    """Ctrl-C leaves no deliverable, so the run's own record says `failed`.

    The verdict is about the file, and an interrupted run has none — the one
    path that reaches the final record without any chunk having failed.
    """
    import json

    from polyglot_lc.pipeline import TranslationPipeline

    monkeypatch.setenv("LOG_JSONL", "1")
    logs = tmp_path / "logs"
    monkeypatch.setenv("LOGS_DIR", str(logs))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(TranslationPipeline, "_run_inner", interrupted)
    with pytest.raises(KeyboardInterrupt):
        TranslationPipeline(quiet=True).run(_write(tmp_path), src="en", tgt="vi")

    log = max(logs.glob("run_*.jsonl"))
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    end = [record for record in records if record.get("type") == "run_end"][-1]
    assert (end["status"], end["exit_code"]) == ("failed", 1)


def test_a_chunk_whose_critic_failed_keeps_the_draft_it_paid_for(mock_env, tmp_path, monkeypatch):
    """The words already bought are kept: a provider failure *after* the
    translator must not send the chunk back to the translator.

    The draft is the newest thing the run has, so a second attempt resumes from
    it and re-runs only the step that failed. Translating the same chunk again
    would bill one sentence twice — which is exactly what a resumed run of the
    same document used to do.
    """
    from polyglot_lc.agents.context import extract_source
    from polyglot_lc.agents.critic import CriticAgent
    from polyglot_lc.agents.translator import TranslatorAgent
    from polyglot_lc.llm.resilience import RateLimitError

    monkeypatch.setenv("MAX_CONCURRENCY", "1")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")
    monkeypatch.setenv("FAST_PATH_MAX_COMPLEXITY", "0")  # every chunk sees the Critic
    monkeypatch.setenv("CHUNK_RETRY_BASE_DELAY", "0.01")
    monkeypatch.setenv("CHUNK_RETRY_MAX_DELAY", "0.02")
    clear_mock_replies()
    path = _write(tmp_path, DOC + "\nA third paragraph to translate as well.\n")
    chunks = _chunks_of(path)
    assert len(chunks) >= 3

    translations: list[str] = []

    def counting_translate(self, prompt, chunk_id=""):
        translations.append(chunk_id)
        return "T::" + extract_source(prompt)

    original_review = CriticAgent.review
    calls = {"n": 0}

    def flaky_review(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("429", status=429)
        return original_review(self, *args, **kwargs)

    monkeypatch.setattr(TranslatorAgent, "translate", counting_translate)
    monkeypatch.setattr(CriticAgent, "review", flaky_review)

    out = tmp_path / "kept-draft.md"
    result = _run(path, out=str(out), max_tokens=15)

    text = out.read_text(encoding="utf-8")
    assert "untranslated" not in text
    # Every chunk was translated exactly once, the one whose Critic failed
    # included: that chunk resumed from the draft it already had.
    assert sorted(translations) == sorted(chunk.id for chunk in chunks)
    assert calls["n"] > len(chunks)  # the Critic ran again; the Translator did not
    assert result.chunks_failed == 0


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
