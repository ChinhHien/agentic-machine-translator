"""Durable per-chunk checkpoints: a crashed run resumes where it stopped.

Each finished chunk is written to ``<root>/<doc_key>/<chunk_id>.json`` as it
completes, so an interrupted run (Ctrl-C, provider outage, crash) can be
resumed with ``--resume`` instead of paying for every chunk again.

A checkpoint belongs to one *document identity* — its path, the digest of its
bytes, the language pair, the Translator model, the user's glossary, the
chunk hashes and the verification policy. Any change invalidates the directory,
so a resumed run can never reuse results produced from different input, a different
chunking, or without the terminology the reader has since asked for.

The document's identity is its **content**, never its ``mtime``. The web UI
uploads a file it has already uploaded again (same name, same bytes), the write
lands a fresh ``mtime`` on the same document, and keyed on ``size:mtime`` that
harmless rewrite orphaned every chunk a paused run had translated — "Tiếp tục"
then quietly re-translated the whole document from the first chunk. The byte
digest changes exactly when the document does.

Checkpointing is best-effort by design: a read-only or missing directory logs a
warning and disables the store instead of failing the translation.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ChunkCheckpoint:
    """Result of one finished chunk, plus enough context to score a run."""

    chunk_id: str
    chunk_hash: str
    translated: str
    score: float = 0.0  # 0-10
    path: str = "standard"  # fast | standard | deep
    stored: bool = False  # trace accepted by the reasoning bank
    calls: int = 0  # LLM requests spent on this chunk
    ts: float = field(default_factory=time.time)
    #: The Translator that produced this text. Recorded, never part of the
    #: job's identity: it says whose words a reused chunk carries (see
    #: :meth:`CheckpointStore.bank_get`).
    model: str = ""
    #: The job this translation was first written for, when it came from the
    #: content bank rather than from this run. Empty for a chunk this job
    #: produced itself.
    source_doc_key: str = ""
    #: The job's identity *without* its model and endpoint (see
    #: :meth:`CheckpointStore.content_key`). The bank hands a chunk from one
    #: model to another only when this matches: a changed glossary, chunking,
    #: skill or verification policy is a different deliverable, and replaying
    #: it would silently keep the old terms.
    content_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> ChunkCheckpoint | None:
        """Rebuild a checkpoint, ignoring anything malformed."""
        if not isinstance(data, dict):
            return None
        chunk_id = str(data.get("chunk_id", ""))
        translated = data.get("translated")
        if not chunk_id or not isinstance(translated, str):
            return None
        return cls(
            chunk_id=chunk_id,
            chunk_hash=str(data.get("chunk_hash", "")),
            translated=translated,
            score=float(data.get("score", 0.0) or 0.0),
            path=str(data.get("path", "standard") or "standard"),
            stored=bool(data.get("stored", False)),
            calls=int(data.get("calls", 0) or 0),
            ts=float(data.get("ts", 0.0) or 0.0),
            model=str(data.get("model", "") or ""),
            source_doc_key=str(data.get("source_doc_key", "") or ""),
            content_key=str(data.get("content_key", "") or ""),
        )


def _content_digest(path: str | Path) -> str:
    """SHA-256 of the file's bytes; ``""`` when it cannot be read.

    Unreadable input yields an empty component rather than an exception: the
    pipeline reports the unreadable document itself, and a key built without
    this part is still stable for the run that is about to fail on it.
    """
    try:
        with Path(path).open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:  # pragma: no cover - the pipeline reports unreadable input
        return ""


class CheckpointStore:
    """Reads and writes per-chunk checkpoints under one root directory."""

    def __init__(self, root: str | Path, *, enabled: bool = True) -> None:
        raw = str(root or "").strip()
        # An empty/"off" root disables the store instead of silently writing
        # checkpoints into the current working directory.
        self.enabled = bool(enabled) and raw.lower() not in {"", ".", "off", "none", "0"}
        self.root = Path(raw) if raw else Path(".")
        self._broken = False
        self._lock = threading.Lock()
        #: Stamped into every checkpoint this store writes, so a chunk reused by
        #: another job keeps its own provenance instead of being relabelled.
        self.model: str = ""
        self.base_url: str = ""
        #: The job's model-independent identity; the bank only matches on it.
        self.content_key: str = ""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @staticmethod
    def doc_key(
        file_path: str | Path,
        *,
        src: str,
        tgt: str,
        model: str,
        chunk_hashes: list[str],
        glossary: str = "",
        skip_llm_verify: bool = False,
        base_url: str = "",
        skill_id: str = "",
        skill_applied: bool = False,
    ) -> str:
        """Stable 16-char key identifying this exact translation job.

        *glossary* is the fingerprint of the terminology the run was given (the
        glossary's pairs, as ``term\ttarget`` lines). It is part of the identity:
        a chunk translated from the same bytes but against other terms is not
        the same deliverable, and replaying it silently left the new term out of
        the file. Its *content* is hashed, so reordering the pairs or rebuilding
        them from another source keeps a resume valid.
        """
        return CheckpointStore._digest(
            (
                src,
                tgt,
                model,
                base_url,
                skill_id,
                "skill-on" if skill_applied else "skill-off",
            ),
            file_path=file_path,
            chunk_hashes=chunk_hashes,
            glossary=glossary,
            skip_llm_verify=skip_llm_verify,
        )

    @staticmethod
    def content_key(
        file_path: str | Path,
        *,
        src: str,
        tgt: str,
        chunk_hashes: list[str],
        glossary: str = "",
        skip_llm_verify: bool = False,
        skill_id: str = "",
        skill_applied: bool = False,
    ) -> str:
        """This job's identity with its model and endpoint left out.

        The content bank hands a translation from one model to another only
        within one such identity. A changed glossary, a different chunking, a
        new skill or another verification policy is a different deliverable —
        replaying it would keep the old terms or the fail-open verdicts — and
        must re-translate even though the chunk text is the same.
        """
        return CheckpointStore._digest(
            (
                src,
                tgt,
                skill_id,
                "skill-on" if skill_applied else "skill-off",
                "content-only",
            ),
            file_path=file_path,
            chunk_hashes=chunk_hashes,
            glossary=glossary,
            skip_llm_verify=skip_llm_verify,
        )

    @staticmethod
    def _digest(
        parts: tuple[str, ...],
        *,
        file_path: str | Path,
        chunk_hashes: list[str],
        glossary: str,
        skip_llm_verify: bool,
    ) -> str:
        """The shared body of :meth:`doc_key` and :meth:`content_key`.

        The file's resolved path and byte digest, the chunks' hashes, the
        glossary content and the verification policy — every part of what a
        translation *is*, so neither key can drift from the other.
        """
        digest = hashlib.sha1()
        try:
            resolved = str(Path(file_path).resolve())
        except OSError:  # pragma: no cover - exotic paths only
            resolved = str(file_path)
        for part in (resolved, *parts):
            digest.update(str(part).encode("utf-8"))
            digest.update(b"\x00")
        for chunk_hash in chunk_hashes:
            digest.update(chunk_hash.encode("utf-8"))
            digest.update(b"\x00")
        digest.update(_content_digest(file_path).encode("utf-8"))
        digest.update(b"\x00")
        if glossary:
            digest.update(glossary.encode("utf-8"))
            digest.update(b"\x00")
        # Old checkpoints may contain fail-open verdicts; keep them out of strict runs.
        digest.update(b"strict-verification-v1\x00")
        digest.update(b"fast-skip\x00" if skip_llm_verify else b"full-verify\x00")
        return digest.hexdigest()[:16]

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def dir_for(self, doc_key: str) -> Path:
        return self.root / doc_key

    def _chunk_path(self, doc_key: str, chunk_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in chunk_id)
        return self.dir_for(doc_key) / f"{safe}.json"

    def _meta_path(self, doc_key: str) -> Path:
        return self.dir_for(doc_key) / "meta.json"

    def _stats_path(self, doc_key: str) -> Path:
        return self.dir_for(doc_key) / "stats.json"

    def _bank_path(self, chunk_hash: str) -> Path:
        digest = hashlib.sha1(chunk_hash.encode("utf-8")).hexdigest()[:16]
        return self.root / "_bank" / f"{digest}.json"

    # ------------------------------------------------------------------
    # IO (all best-effort)
    # ------------------------------------------------------------------
    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            if not self._broken:
                logger.warning("Checkpointing disabled: {} ({})", exc, path)
            self._broken = True

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Ignoring unreadable checkpoint {}: {}", path, exc)
            return None

    @property
    def usable(self) -> bool:
        """True when this store is enabled and has not hit an IO error."""
        return self.enabled and not self._broken

    def save_meta(self, doc_key: str, meta: dict[str, Any]) -> None:
        """Record the job description (used for reporting and invalidation)."""
        if not self.usable:
            return
        with self._lock:
            self._write_json(self._meta_path(doc_key), dict(meta))

    def load_meta(self, doc_key: str) -> dict[str, Any]:
        if not self.usable:
            return {}
        raw = self._read_json(self._meta_path(doc_key))
        return raw if isinstance(raw, dict) else {}

    def save_stats(self, doc_key: str, stats: dict[str, Any]) -> None:
        """Persist the cumulative per-translation counters beside the chunks."""
        if not self.usable:
            return
        with self._lock:
            self._write_json(self._stats_path(doc_key), dict(stats))

    def load_stats(self, doc_key: str) -> dict[str, Any]:
        """Cumulative counters from earlier attempts, or ``{}``."""
        if not self.usable:
            return {}
        raw = self._read_json(self._stats_path(doc_key))
        return raw if isinstance(raw, dict) else {}

    def put(
        self,
        doc_key: str,
        checkpoint: ChunkCheckpoint,
        *,
        bank: bool = True,
    ) -> None:
        """Persist one finished chunk under its job, and in the content bank.

        The bank is keyed by the chunk's own hash — its text — rather than by
        the job, so a chunk another model already translated can still be found
        when the job's identity has changed (a model or endpoint switch). A
        checkpoint that already carries its provenance keeps it: adopting a
        bank entry into a new job must not relabel the old model's words.
        """
        if not self.usable:
            return
        if not checkpoint.model:
            checkpoint.model = self.model
        if not checkpoint.source_doc_key:
            checkpoint.source_doc_key = doc_key
        if not checkpoint.content_key:
            checkpoint.content_key = self.content_key
        with self._lock:
            self._write_json(self._chunk_path(doc_key, checkpoint.chunk_id), checkpoint.to_dict())
            if bank and checkpoint.translated.strip():
                self._write_json(self._bank_path(checkpoint.chunk_hash), checkpoint.to_dict())

    def bank_get(self, chunk_hash: str, content_key: str = "") -> ChunkCheckpoint | None:
        """A verified translation of exactly this chunk text, from any job.

        Returns ``None`` when nothing was banked, the entry is malformed, or the
        stored hash is not the one asked for — a bank file is content addressed,
        and a mismatch means the file is not what it claims to be.
        """
        if not self.usable or not chunk_hash:
            return None
        with self._lock:
            raw = self._read_json(self._bank_path(chunk_hash))
        checkpoint = ChunkCheckpoint.from_dict(raw)
        if checkpoint is None or checkpoint.chunk_hash != chunk_hash:
            return None
        if not checkpoint.translated.strip():
            return None
        wanted = content_key or self.content_key
        if wanted and checkpoint.content_key != wanted:
            # Same chunk text, different job: another glossary, another
            # chunking, another verification policy. Not this deliverable.
            return None
        return checkpoint

    def get(self, doc_key: str, chunk_id: str, chunk_hash: str = "") -> ChunkCheckpoint | None:
        """Load one chunk's checkpoint; None when missing or stale."""
        if not self.usable:
            return None
        with self._lock:
            raw = self._read_json(self._chunk_path(doc_key, chunk_id))
        if raw is None:
            return None
        checkpoint = ChunkCheckpoint.from_dict(raw)
        if checkpoint is None:
            return None
        if checkpoint.chunk_hash != chunk_hash:
            logger.debug(
                "Checkpoint for {} is stale (chunk changed); re-translating",
                chunk_id,
            )
            return None
        return checkpoint

    def completed(self, doc_key: str) -> dict[str, ChunkCheckpoint]:
        """Every usable checkpoint for this job, keyed by chunk id."""
        if not self.usable or not self.dir_for(doc_key).exists():
            return {}
        out: dict[str, ChunkCheckpoint] = {}
        for path in self.dir_for(doc_key).glob("*.json"):
            if path.name in {"meta.json", "stats.json"}:
                continue
            checkpoint = ChunkCheckpoint.from_dict(self._read_json(path))
            if checkpoint is not None:
                out[checkpoint.chunk_id] = checkpoint
        return out

    def missing(self, doc_key: str, chunks: list[Any]) -> list[str]:
        """Ids of chunks with no usable checkpoint for this job.

        A checkpoint exists only for a chunk the pipeline accepted, so this is
        also the list of chunks that are not verified. Chunk ids are the
        plan's own parent ids; the halves produced while salvaging a cut reply
        never get one and must not be counted here.
        """
        return [
            chunk.id
            for chunk in chunks
            if not (checkpoint := self.get(doc_key, chunk.id, chunk.hash()))
            or not checkpoint.translated.strip()
        ]

    def clear(self, doc_key: str) -> None:
        """Delete every checkpoint for this job."""
        if not self.enabled:
            return
        directory = self.dir_for(doc_key)
        if not directory.exists():
            return
        with self._lock:
            for path in directory.glob("*"):
                try:
                    path.unlink()
                except OSError as exc:  # pragma: no cover - best effort
                    logger.debug("Could not remove {}: {}", path, exc)
