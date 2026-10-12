"""Output assembly: ordered chunks -> one document, with code restored."""

from __future__ import annotations

import difflib
import os
import re
import threading
from pathlib import Path
from typing import Any

from polyglot_lc.ingest.md import restore_code

#: Separator between two assembled chunks (matches the previous behaviour).
CHUNK_SEPARATOR = "\n\n"


def _block_refs(chunk: Any) -> list[int] | None:
    """The source-block indices a chunk covers, or ``None`` if unknown."""
    refs = getattr(chunk, "block_refs", None)
    if isinstance(refs, (list, tuple)):
        return list(refs)
    return None


def _overlap_prefix_count(prev_refs: list[int], refs: list[int]) -> int:
    """How many leading source blocks of *refs* were already covered by
    *prev_refs* (the chunker replays the previous chunk's tail as context).

    Only counts a *contiguous* leading run that is fully present in the previous
    chunk, so it never mistakes unrelated shared blocks for an overlap.
    """
    prev_set = set(prev_refs)
    count = 0
    for idx in refs:
        if idx in prev_set:
            count += 1
        else:
            break
    return count


#: Whitespace / punctuation normaliser for fuzzy paragraph comparison.
_WS_RE = re.compile(r"\s+")

#: Similarity at which two consecutive paragraphs count as the same content
#: translated twice (0..1).
_NEAR_DUPLICATE_RATIO = 0.85

#: A table-of-contents entry: a dot leader (or the collapsed ellipsis form) and
#: an optional page number. Two *distinct* entries in a row are two entries,
#: however alike their titles look — a contents page is not duplicated prose.
_TOC_ENTRY_RE = re.compile(
    r"(?:\.(?:\s*\.){3,}|…)\s*(?:\d{1,4}|[IVXLCDM]{1,7})?\s*$", re.IGNORECASE
)


def _normalize_para(text: str) -> str:
    """Normalise a paragraph for fuzzy comparison (casefold + collapse WS)."""
    return _WS_RE.sub(" ", text.strip()).casefold()


def _textual_overlap_count(paragraphs: list[str]) -> int:
    """Count how many leading paragraphs in *paragraphs* are near-duplicates.

    A leading paragraph is considered a duplicate when its first 120
    normalised characters match a subsequent paragraph in the same list.
    This detects chunks whose overlap was translated slightly differently.
    """
    if len(paragraphs) < 2:
        return 0
    # Build a set of normalised paragraph prefixes from the second half.
    second_half = {
        _normalize_para(p)[:120]
        for p in paragraphs[1:]
        if len(p.strip()) > 30  # ignore short fragments
    }
    drop = 0
    for p in paragraphs:
        norm = _normalize_para(p)[:120]
        if len(p.strip()) < 30:
            break
        if norm in second_half:
            drop += 1
        else:
            break
    # Never drop everything.
    return min(drop, len(paragraphs) - 1)


def _dedup_predecessor(prev: Any | None, translations: dict[str, str]) -> Any | None:
    """The previous chunk to de-duplicate *against*, or ``None``.

    Overlap may only be stripped when the previous chunk actually contributed
    text: its copy of the overlap is the one that stays in the document. When it
    did not — its retries were exhausted, so it has no translation — stripping
    the next chunk's head deletes that content twice over, and, unlike a missing
    chunk, leaves no marker behind, because the next chunk did produce a
    translation.
    """
    if prev is None:
        return None
    return prev if translations.get(prev.id) else None


def _lost_block_note(
    chunk: Any, neighbours: tuple[Any | None, ...], translations: dict[str, str]
) -> str:
    """Name the source blocks that a missing chunk takes with it.

    A chunk with no translation is not necessarily a total loss: the chunker
    replays the tail of one chunk as the head of the next, so shared blocks
    still reach the output through a neighbour. Naming only the remainder makes
    the marker actionable — it says what a re-run has to recover.
    """
    refs = _block_refs(chunk)
    if not refs:
        return ""
    covered: set[int] = set()
    for other in neighbours:
        if other is not None and translations.get(other.id):
            covered |= set(_block_refs(other) or [])
    lost = sorted(set(refs) - covered)
    if not lost:
        return " (content carried by the neighbouring chunk)"
    if len(lost) == 1:
        return f" (source block {lost[0]} lost)"
    if lost[-1] - lost[0] + 1 == len(lost):
        return f" (source blocks {lost[0]}-{lost[-1]} lost)"
    return f" ({len(lost)} source blocks lost)"


def _chunk_emission(
    chunk: Any, prev_chunk: Any | None, text: str, default: str | None = None
) -> str:
    """The part of *text* to keep for *chunk*'s place in the document.

    With chunk overlap, a later chunk re-contains the previous chunk's tail
    source blocks; the translation of that region would be emitted twice. We
    drop the *translation* of those leading blocks (split on the paragraph
    separator the translator is told to preserve). It conservatively never
    strips more paragraphs than the overlap count, so new content is never lost.

    ``prev_chunk`` must be a chunk that actually contributed text (see
    :func:`_dedup_predecessor`): there is nothing to de-duplicate against when
    the previous chunk emitted nothing, and the overlap is then the only copy.

    Additionally, when block_refs are unavailable, we perform paragraph-level
    textual deduplication against the previous chunk's trailing paragraphs to
    catch rephrased overlap.
    """
    if text is None:
        return default if default is not None else text
    refs = _block_refs(chunk)
    prev_refs = _block_refs(prev_chunk) if prev_chunk is not None else None

    # ---- Strategy 1: block-ref-based overlap removal ----
    if refs and prev_refs:
        drop = _overlap_prefix_count(prev_refs, refs)
        if drop > 0:
            paragraphs = text.split(CHUNK_SEPARATOR)
            drop = min(drop, max(0, len(paragraphs) - 1))
            return CHUNK_SEPARATOR.join(paragraphs[drop:])

    # ---- Strategy 2: textual similarity dedup (fallback) ----
    # When the previous chunk is available, check if any leading paragraphs
    # of this chunk are near-duplicates of trailing paragraphs of the previous.
    if prev_chunk is not None and text and hasattr(prev_chunk, "id"):
        paragraphs = text.split(CHUNK_SEPARATOR)
        if len(paragraphs) > 1:
            drop = _textual_overlap_count(paragraphs)
            if drop > 0:
                return CHUNK_SEPARATOR.join(paragraphs[drop:])

    return text


def assemble(
    chunks: list[Any],
    translations: dict[str, str],
    code_map: dict[str, str] | None = None,
) -> str:
    """Join the translated chunks in the **original document order**.

    Ordering comes from the chunk list, not from completion order or dict
    ordering, so bounded-concurrency runs always emit the same document. Chunks
    that share overlap are de-duplicated so each source block appears once.
    Shielded code blocks are restored from *code_map* here, once, after
    translation.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        if not translations.get(chunk.id):
            # A chunk that never got a translation (its retries were exhausted)
            # must be visible in the output — silently emitting nothing is how
            # a whole chapter disappears. The note names the source blocks
            # nothing else carries, so it is clear what --resume must recover.
            neighbours = (
                chunks[i - 1] if i > 0 else None,
                chunks[i + 1] if i + 1 < len(chunks) else None,
            )
            note = _lost_block_note(chunk, neighbours, translations)
            parts.append(f"<!-- {chunk.id}: untranslated{note} -->")
            continue
        prev = _dedup_predecessor(chunks[i - 1] if i > 0 else None, translations)
        text = _chunk_emission(chunk, prev, translations.get(chunk.id, ""), default="")
        parts.append(restore_code(text, code_map or {}))
    assembled = CHUNK_SEPARATOR.join(parts)
    return _deduplicate_paragraphs(assembled)


def block_translations(
    chunks: list[Any],
    translations: dict[str, str],
) -> dict[int, str]:
    """Per-source-block translated text, with the same overlap rules.

    Parallel to :func:`assemble`: the same chunks, the same overlap removal and
    de-duplication, but keyed by source-block index instead of concatenated into
    one document. The PDF rebuild needs the per-block text, and re-implementing
    the de-duplication rules in a second place is exactly how the Markdown and
    PDF outputs would start to disagree about which translation belongs where.

    A block whose translation cannot be pinned down is simply absent: the caller
    keeps its source text rather than printing the wrong paragraph over it.
    """
    out: dict[int, str] = {}
    for i, chunk in enumerate(chunks):
        text = translations.get(chunk.id)
        refs = _block_refs(chunk)
        if not text or not refs:
            continue
        prev = _dedup_predecessor(chunks[i - 1] if i > 0 else None, translations)
        paragraphs = text.split(CHUNK_SEPARATOR)
        drop = 0
        prev_refs = _block_refs(prev) if prev is not None else None
        if prev_refs:
            drop = _overlap_prefix_count(prev_refs, refs)
        drop = min(drop, max(0, len(paragraphs) - 1))
        kept, targets = paragraphs[drop:], refs[drop:]
        if not targets:
            continue
        if len(kept) == len(targets):
            out.update(zip(targets, kept))
        elif len(kept) > len(targets):
            # The translator produced more paragraphs than the chunk had blocks
            # (it split one): keep them all on the last block instead of
            # dropping the tail.
            for ref, paragraph in zip(targets, kept):
                out[ref] = paragraph
            out[targets[-1]] = CHUNK_SEPARATOR.join(kept[len(targets) - 1 :])
        else:
            # Fewer paragraphs than blocks (it merged two): the blocks it ran
            # out of text for keep their source lines.
            out.update(zip(targets, kept))
    return out


def assemble_partial(
    chunks: list[Any],
    translations: dict[str, str],
    code_map: dict[str, str] | None = None,
) -> str:
    """Assemble only the chunks that are already translated.

    Used for the incremental ``PARTIAL_OUTPUT`` file: placeholders stand in for
    chunks still in flight so the document shape stays readable while a long run
    is still going.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        # De-duplicate only against a neighbour that already has text; while a
        # run is still going, the overlap of a pending chunk is its own only
        # copy, and the incremental file must not show it as missing.
        prev = _dedup_predecessor(chunks[i - 1] if i > 0 else None, translations)
        text = _chunk_emission(chunk, prev, translations.get(chunk.id), default=None)
        if text is None:
            parts.append(f"<!-- {chunk.id}: pending -->")
        else:
            parts.append(restore_code(text, code_map or {}))
    return CHUNK_SEPARATOR.join(parts)


def _deduplicate_paragraphs(text: str) -> str:
    """Remove consecutive near-duplicate paragraphs from the assembled document.

    After per-chunk overlap removal, some duplicates remain because the
    translator rendered the same source content twice at a chunk boundary, in
    slightly different words. This pass collapses *consecutive* paragraphs whose
    normalised text is at least :data:`_NEAR_DUPLICATE_RATIO` similar, keeping
    the longer one (the prefix-equality test this replaces missed reworded
    duplicates, which is what survived into the output).
    """
    paragraphs = text.split(CHUNK_SEPARATOR)
    if len(paragraphs) < 2:
        return text
    kept: list[str] = [paragraphs[0]]
    for para in paragraphs[1:]:
        prev_norm = _normalize_para(kept[-1])
        curr_norm = _normalize_para(para)
        both_toc = bool(_TOC_ENTRY_RE.search(prev_norm) and _TOC_ENTRY_RE.search(curr_norm))
        if len(prev_norm) > 40 and len(curr_norm) > 40 and curr_norm != prev_norm and not both_toc:
            ratio = difflib.SequenceMatcher(None, prev_norm, curr_norm).ratio()
            if ratio >= _NEAR_DUPLICATE_RATIO:
                if len(curr_norm) > len(prev_norm):
                    kept[-1] = para
                continue
        kept.append(para)
    return CHUNK_SEPARATOR.join(kept)


def output_paths(
    file_path: str | Path, tgt: str, out: str | Path | None = None
) -> tuple[Path, Path]:
    """``(markdown_path, pdf_path)`` for a run, derived from the same base.

    Both names always come from one base, so ``--out report.vi.md`` and
    ``--out report.vi.pdf`` name the same pair of files and a run cannot write
    its Markdown and its PDF next to unrelated names.
    """
    base = Path(str(out)) if out else Path(f"{Path(file_path).stem}.{tgt}.md")
    if base.suffix.lower() == ".pdf":
        base = base.with_suffix(".md")
    return base, base.with_suffix(".pdf")


def output_path(file_path: str | Path, tgt: str, out: str | Path | None = None) -> Path:
    """The default Markdown path for a run (``<stem>.<tgt>.md``).

    Markdown keeps its own extension even when the source was a PDF: ingestion
    converts the document, and handing a text file the source's ``.pdf`` name
    would only mislead a PDF reader.
    """
    return output_paths(file_path, tgt, out)[0]


def write_atomic(path: Path, text: str) -> None:
    """Write *text* to *path* via a temp file so readers never see a partial doc.

    The temp name is per process/thread, so concurrent incremental (partial)
    writes of the same document cannot clobber each other's temporary file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
