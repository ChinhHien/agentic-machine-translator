"""The extracted document, cached by content so a resume does not read it again.

A checkpoint stores translations, not the document: to find the checkpoint
directory the pipeline needs the document's chunks, and to get those it has to
extract the PDF again. On a 500-page book that extraction is two minutes of work
repeated on every "Tiếp tục", and — worse — the optional AI ingest review runs
again too, rewrites a few blocks, changes the chunk hashes and with them the
checkpoint identity, so the resumed run replays nothing and pays for the whole
document a second time.

This cache closes both holes. It is keyed by the file's own bytes (plus the
package version and a format version), so the same document always maps to the
same extraction; the reviewed text is frozen in the cache, so the review changes
a block at most once; and the glossary candidates found by the heuristics come
back with it, so the terminology fingerprint the checkpoint identity hashes is
the one the first attempt used.

Everything here is best-effort, like checkpointing itself: an unwritable cache
directory logs a warning and the run extracts the document as before.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from loguru import logger

from polyglot_lc import __version__
from polyglot_lc.ingest.model import Document
from polyglot_lc.segment.glossary import GlossaryTerm

#: Bumped when the payload's layout changes; an older entry is then ignored
#: rather than half-read.
INGEST_CACHE_VERSION = 1

#: Directory name under the checkpoint root, so a job's cache shares its life.
CACHE_DIRNAME = ".ingest"


def cache_key(sha256: str) -> str:
    """The file name this extraction is stored under."""
    digest = hashlib.sha1()
    for part in (sha256, str(INGEST_CACHE_VERSION), __version__):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


class IngestCache:
    """Reads and writes one document's extraction under a stable key."""

    def __init__(self, root: str | Path, *, enabled: bool = True) -> None:
        raw = str(root or "").strip()
        # The same rule as the checkpoint store: an empty/"off" root means the
        # cache is off, never "write it into the current working directory".
        self.enabled = bool(enabled) and raw.lower() not in {"", ".", "off", "none", "0"}
        self.root = (Path(raw) if raw else Path(".")) / CACHE_DIRNAME
        self._broken = False
        self._lock = threading.Lock()

    @property
    def usable(self) -> bool:
        """True when this cache is enabled and has not hit an IO error."""
        return self.enabled and not self._broken

    def _path(self, sha256: str) -> Path:
        return self.root / f"{cache_key(sha256)}.json"

    def load(self, sha256: str) -> tuple[Document, list[GlossaryTerm]] | None:
        """The cached extraction of the file with this digest, or ``None``."""
        if not self.usable or not sha256:
            return None
        path = self._path(sha256)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Ignoring unreadable ingest cache {}: {}", path, exc)
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != INGEST_CACHE_VERSION or payload.get("sha256") != sha256:
            return None
        try:
            doc = Document.model_validate(payload.get("doc"))
        except Exception as exc:  # noqa: BLE001 - a cache we cannot read is a miss
            logger.debug("Ignoring malformed ingest cache {}: {}", path, exc)
            return None
        terms = [
            GlossaryTerm(
                term=str(item.get("term", "")),
                count=int(item.get("count", 0) or 0),
                reason=str(item.get("reason", "")),
            )
            for item in payload.get("glossary_terms") or []
            if isinstance(item, dict) and str(item.get("term", "")).strip()
        ]
        return doc, terms

    def save(
        self,
        sha256: str,
        doc: Document,
        glossary_terms: list[GlossaryTerm],
    ) -> None:
        """Store this extraction for the next attempt; never fails the run."""
        if not self.usable or not sha256:
            return
        payload = {
            "version": INGEST_CACHE_VERSION,
            "sha256": sha256,
            "package": __version__,
            "doc": doc.model_dump(),
            "glossary_terms": [term.to_dict() for term in glossary_terms],
        }
        path = self._path(sha256)
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline=""
                )
                tmp.replace(path)
            except (OSError, TypeError, ValueError) as exc:
                if not self._broken:
                    logger.warning("Ingest cache disabled: {} ({})", exc, path)
                self._broken = True


__all__ = ["INGEST_CACHE_VERSION", "IngestCache", "cache_key"]
