"""Output assembly: ordered chunks -> one document, with code restored."""

from __future__ import annotations

import difflib
import math
import os
import re
import statistics
import threading
from pathlib import Path
from typing import Any

from polyglot_lc import emphasis
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

#: Score of an alignment state that cannot be reached, so every option below is
#: preferred to it (see :func:`_align_paragraphs`).
_UNREACHED = float("-inf")

#: A table-of-contents entry: a dot leader (or the collapsed ellipsis form) and
#: an optional page number. Two *distinct* entries in a row are two entries,
#: however alike their titles look — a contents page is not duplicated prose.
_TOC_ENTRY_RE = re.compile(
    r"(?:\.(?:\s*\.){3,}|…)\s*(?:\d{1,4}|[IVXLCDM]{1,7})?\s*$", re.IGNORECASE
)


def _normalize_para(text: str) -> str:
    """Normalise a paragraph for fuzzy comparison (casefold + collapse WS)."""
    return _WS_RE.sub(" ", text.strip()).casefold()


def _overlap_paragraph_cut(
    chunk: Any, prev_refs: list[int] | None, refs: list[int], paragraphs: list[str]
) -> int:
    """How many leading paragraphs of a chunk are *only* the replayed overlap.

    The chunker replays the previous chunk's tail source blocks as the head of
    this chunk, so their translation arrives twice. Counting paragraphs assumed
    the translator answers with one paragraph per block: a heading it merged
    with the paragraph below, or a line break it dropped, put new content in
    paragraph 0 — and a count then cut it away. So the blocks are aligned to the
    paragraphs (the same evidence :func:`block_translations` uses) and only a
    leading paragraph whose blocks are *all* replayed is dropped. A paragraph
    that also carries a new block's words stays, which at worst leaves a visible
    duplicate and never a hole.
    """
    if not prev_refs:
        return 0
    drop = _overlap_prefix_count(prev_refs, refs)
    if drop <= 0 or not paragraphs:
        return 0
    sources = str(getattr(chunk, "text", "") or "").split(CHUNK_SEPARATOR)
    if len(sources) != len(refs) or not all(source.strip() for source in sources):
        return min(drop, max(0, len(paragraphs) - 1))
    # Only a leading paragraph can be cut, so only a short prefix needs aligning.
    # The grid is quadratic in the chunk, and the incremental (``PARTIAL_OUTPUT``)
    # file runs this once per completed chunk; the window grows only while the
    # answer is still "the whole window is replay".
    longest = max(len(sources), len(paragraphs))
    limit = min(max(drop + 2, 4), longest)
    while True:
        window = paragraphs[:limit]
        prefix = sources[:limit]
        owners = _align_paragraphs(prefix, window, _ratio_norm(window, prefix))
        if owners is None:
            return min(drop, max(0, len(paragraphs) - 1))
        # Which source blocks each paragraph carries. A block the translator
        # folded into the paragraph above its own (``[]``) is attributed to that
        # paragraph too: its words are there, and dropping the paragraph would
        # drop them.
        holders: dict[int, set[int]] = {}
        last = -1
        for block, owned in enumerate(owners):
            if owned is None:
                continue  # left out entirely: it holds no paragraph
            if not owned:
                if last >= 0:
                    holders.setdefault(last, set()).add(block)
                continue
            last = max(owned)
            for index in owned:
                holders.setdefault(index, set()).add(block)
        replayed = set(range(drop))
        cut = 0
        # A paragraph no block was pinned to (``holders`` has no entry for it) is
        # never cut: its owner is unknown, and keeping it costs a visible
        # duplicate at worst.
        while cut < len(window) and holders.get(cut, {-1}) <= replayed:
            cut += 1
        if cut < len(window) or limit >= longest:
            # Never cut the chunk away entirely: the last paragraph stays even
            # if the alignment says every word of it was already printed.
            return min(cut, max(0, len(paragraphs) - 1))
        limit = min(limit * 2, longest)


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
    separator the translator is told to preserve), and only the paragraphs that
    belong to nothing but the replayed blocks — see
    :func:`_overlap_paragraph_cut` — so new content is never lost.

    ``prev_chunk`` must be a chunk that actually contributed text (see
    :func:`_dedup_predecessor`): there is nothing to de-duplicate against when
    the previous chunk emitted nothing, and the overlap is then the only copy.

    A reworded duplicate that the cut cannot place is caught at the boundary by
    :func:`assemble` (:func:`_repeat_of`), not here: this function only ever sees
    one chunk's reply.
    """
    if text is None:
        return default if default is not None else text
    refs = _block_refs(chunk)
    prev_refs = _block_refs(prev_chunk) if prev_chunk is not None else None
    if refs and prev_refs and text:
        paragraphs = text.split(CHUNK_SEPARATOR)
        cut = _overlap_paragraph_cut(chunk, prev_refs, refs, paragraphs)
        if cut > 0:
            return CHUNK_SEPARATOR.join(paragraphs[cut:])
    return text


def assemble(
    chunks: list[Any],
    translations: dict[str, str],
    code_map: dict[str, str] | None = None,
) -> str:
    """Join the translated chunks in the **original document order**.

    Ordering comes from the chunk list, not from completion order or dict
    ordering, so bounded-concurrency runs always emit the same document. Chunks
    that share overlap are de-duplicated so each source block appears once: the
    replay is cut by block ownership (:func:`_chunk_emission`), and a reworded
    replay that survived the cut is folded only across the chunk boundary where
    it can occur (:func:`_repeat_of`). Shielded code blocks are restored from
    *code_map* while each chunk is placed.
    """
    paragraphs: list[str] = []
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
            paragraphs.append(f"<!-- {chunk.id}: untranslated{note} -->")
            continue
        prev = _dedup_predecessor(chunks[i - 1] if i > 0 else None, translations)
        text = _chunk_emission(chunk, prev, translations.get(chunk.id, ""), default="")
        incoming = restore_code(text, code_map or {}).split(CHUNK_SEPARATOR)
        if paragraphs and incoming and _repeat_of(paragraphs[-1], incoming[0]):
            # The same paragraph translated twice, once on each side of a chunk
            # boundary: keep the longer rendering. Only that pair is compared —
            # see :func:`_repeat_of` for why the document is not swept as a whole.
            if len(_normalize_para(incoming[0])) > len(_normalize_para(paragraphs[-1])):
                paragraphs[-1] = incoming[0]
            incoming = incoming[1:]
        paragraphs.extend(incoming)
    # The sentinels are the vocabulary the *model* is given; a reader gets real
    # Markdown, so they are rendered here, last, after code has been restored.
    # ``to_markdown`` only touches its own four tokens, so restored code — which
    # may contain anything, brackets included — cannot be damaged by it.
    return emphasis.to_markdown(CHUNK_SEPARATOR.join(paragraphs))


#: A token a translation is expected to keep as it stands: a number, a
#: shielded code placeholder, or an identifier / proper noun (it carries a
#: digit, an underscore, a colon or an internal capital). Translations change
#: words; these are the things a reader would notice as wrong if they changed.
_ANCHOR_RE = re.compile(
    r"⟦CODE_\d+⟧|\d[\d.,:%]*|[A-Za-z][A-Za-z0-9_]*[A-Z_][A-Za-z0-9_]*"
    r"|[A-Za-z]+:[A-Za-z]+|[A-Za-z]+\d+[A-Za-z\d]*"
)

#: What makes a source block and a translated paragraph look like each other's:
#: the evidence they share (see :data:`_ANCHOR_RE`), the shape they were laid out
#: in (a heading stays a heading, a five-item list stays a five-item list), and
#: how their lengths compare with the rest of the run. One block per paragraph is
#: the norm, so a plausible pair starts from a positive score: reading a block
#: and a paragraph in order must beat rearranging them into a merge and a split,
#: or a flat run of prose would be free to drift.
_PAIR_BASE = 1.0
_ANCHOR_WEIGHT = 1.5
_SHAPE_WEIGHT = 1.0
_RATIO_WEIGHT = 1.0

#: What an alignment step costs when it is *not* one source block per paragraph.
#: A merge — a title the source wrapped over two blocks comes back as one
#: paragraph — is worth something when the two blocks read as one sentence and
#: the paragraph accounts for both their lengths, and loses more than that when
#: they do not. A block the translator left out costs less than that, and it is
#: kept in its own language: an entry that survived as English is a loss the
#: reader can see, and a paragraph stretched over it is a loss they cannot.
_FOLD_BONUS = 0.5
_FOLD_PENALTY = 1.5
_DROP_BLOCK_COST = 0.6
_SPLIT_COST = 0.2
_STRAY_PARAGRAPH_COST = 2.0

#: A block that ends a sentence ends a paragraph: the next block that does not
#: start a heading or a list item is a continuation of it ("...Rapidly Changing
#: Data" / "Model for Dealing with..."). A contents entry is a whole item too,
#: however its title reads: it ends at its page number, not at a full stop.
_SENTENCE_END = ".!?:;…"
_LINE_STARTS = ("#", "-", "*", "|")
_CONTENTS_TAIL_RE = re.compile(r"(?:…|\.{3,})\s*\d{1,4}\s*$")


def _anchor_tokens(text: str) -> set[str]:
    """The tokens of *text* that a translation is expected to keep."""
    return set(_ANCHOR_RE.findall(text))


def _shape_score(source: str, paragraph: str) -> float:
    """How much the block's *place in the document* agrees with the paragraph's.

    A heading marker survives translation, and so does the number of items of a
    list; both are structure the translator was asked to keep, so their absence
    is evidence about which paragraph this block owns.
    """
    score = 0.0
    stripped = paragraph.lstrip()
    if source.lstrip().startswith("#"):
        score += 1.0 if stripped.startswith("#") else -0.5
    items = sum(1 for line in source.splitlines() if line.lstrip().startswith(("- ", "* ")))
    if items:
        got = sum(1 for line in paragraph.splitlines() if line.lstrip().startswith(("- ", "* ")))
        score += 1.0 if got == items else -0.5
    return score


def _pair_score(
    source: str,
    paragraph: str,
    norm: float,
    *,
    anchors: set[str] | None = None,
    paragraph_anchors: set[str] | None = None,
) -> float:
    """How much *paragraph* looks like the translation of *source*.

    *anchors* and *paragraph_anchors* are the tokens already extracted for the
    two texts. Every cell of the alignment grid asks the same question of the
    same strings, so extracting them per cell made the score quadratic in the
    document by itself: 174 000 regex scans for one 187-page book.
    """
    anchors = _anchor_tokens(source) if anchors is None else anchors
    others = _anchor_tokens(paragraph) if paragraph_anchors is None else paragraph_anchors
    shared = len(anchors & others) / len(anchors) if anchors else 0.0
    ratio = math.exp(-abs(math.log(max(len(paragraph), 1) / max(len(source), 1) / norm)))
    return (
        _PAIR_BASE
        + _ANCHOR_WEIGHT * shared
        + _SHAPE_WEIGHT * _shape_score(source, paragraph)
        + _RATIO_WEIGHT * ratio
    )


def _fold_evidence(source: str, following: str, paragraph: str, norm: float) -> float:
    """What it is worth to read *following* as the rest of *source*'s sentence.

    Two things have to hold at once: the sentence reads on into the next block,
    and the paragraph accounts for both blocks' words — which is what a source
    title wrapped over two blocks looks like once it is translated. Evidence of
    only one of the two is not enough to move a block's translation off it.
    """
    combined = max(len(source) + len(following), 1)
    fit = math.exp(-abs(math.log(max(len(paragraph), 1) / combined / norm)))
    read_on = _FOLD_BONUS if _continues(source, following) else -_FOLD_PENALTY
    return 0.5 * fit + read_on


def _continues(source: str, following: str) -> bool:
    """True when *following* reads as the rest of *source*'s sentence."""
    body = source.rstrip()
    if not body:
        return True
    if body[-1] in _SENTENCE_END or _CONTENTS_TAIL_RE.search(body):
        return False
    if body.lstrip().startswith(_LINE_STARTS):
        return False  # a heading or a list item is a whole block, not a clause
    head = following.lstrip()[:1]
    return bool(head) and head not in _LINE_STARTS


def _align_paragraphs(
    sources: list[str], paragraphs: list[str], norm: float
) -> list[list[int] | None] | None:
    """Which paragraph(s) each source block owns, in document order.

    A dynamic program over the two lists: the translator may return one
    paragraph for two short blocks (a wrapped title) or two paragraphs for one
    long block, and *which* it did is decided by the evidence — the numbers and
    identifiers the two share, the shape they are laid out in, and how their
    lengths compare with the run's — never by the paragraph's position in the
    list.

    One entry per source block: the paragraph indices it owns, ``[]`` when its
    words are in the paragraph printed for the block above it (the merge), and
    ``None`` when the translation has no text for it at all — the block then
    keeps its own language rather than having a neighbour's paragraph stretched
    over it. ``None`` for the whole alignment means it could not be worked out.
    """
    n, m = len(sources), len(paragraphs)
    if n == 0 or m == 0:
        return None
    source_anchors = [_anchor_tokens(source) for source in sources]
    paragraph_anchors = [_anchor_tokens(paragraph) for paragraph in paragraphs]
    scores = [
        [
            _pair_score(
                sources[i],
                paragraphs[j],
                norm,
                anchors=source_anchors[i],
                paragraph_anchors=paragraph_anchors[j],
            )
            for j in range(m)
        ]
        for i in range(n)
    ]
    best: list[list[float]] = [[_UNREACHED] * (m + 1) for _ in range(n + 1)]
    move: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0

    def offer(i: int, j: int, value: float, name: str) -> None:
        if value > best[i][j]:
            best[i][j], move[i][j] = value, name

    for i in range(n + 1):
        for j in range(m + 1):
            here = best[i][j]
            if here == _UNREACHED:
                continue
            if i < n and j < m:
                offer(i + 1, j + 1, here + scores[i][j], "pair")
            if i < n and i > 0 and j > 0:
                # The block above took this block's words with it: the paragraph
                # already printed for it is this block's translation too, so no
                # paragraph is consumed here — only this block's own lines are.
                gain = _fold_evidence(sources[i - 1], sources[i], paragraphs[j - 1], norm)
                offer(i + 1, j, here + gain, "fold")
            if i < n:
                # No paragraph for this block: the translator left it out.
                offer(i + 1, j, here - _DROP_BLOCK_COST, "drop")
            if i < n and j + 1 < m:
                # One block came back as two paragraphs: they are one block's
                # text, so the move is worth what they average out to.
                both = (scores[i][j] + scores[i][j + 1]) / 2 - _SPLIT_COST
                offer(i + 1, j + 2, here + both, "split")
            if j < m:
                offer(i, j + 1, here - _STRAY_PARAGRAPH_COST, "stray")

    if best[n][m] == _UNREACHED:
        return None
    owners: list[list[int] | None] = [[] for _ in range(n)]
    i, j = n, m
    while (i, j) != (0, 0):
        kind = move[i][j]
        if kind == "pair":
            owners[i - 1].append(j - 1)
            i, j = i - 1, j - 1
        elif kind == "fold":
            i -= 1
        elif kind == "drop":
            owners[i - 1] = None
            i -= 1
        elif kind == "split":
            owners[i - 1].append(j - 1)
            owners[i - 1].append(j - 2)
            i, j = i - 1, j - 2
        elif kind == "stray":
            j -= 1
        else:  # pragma: no cover - every reachable state was entered by a move
            return None
    return [sorted(owned) if owned else owned for owned in owners]


def block_translations(
    chunks: list[Any],
    translations: dict[str, str],
) -> dict[int, str | None]:
    """Per-source-block translated text, with the same overlap rules.

    Parallel to :func:`assemble`: the same chunks, the same overlap removal and
    de-duplication, but keyed by source-block index instead of concatenated into
    one document. The PDF rebuild needs the per-block text, and re-implementing
    the de-duplication rules in a second place is exactly how the Markdown and
    PDF outputs would start to disagree about which translation belongs where.

    The block each paragraph belongs to is decided by :func:`_align_paragraphs`,
    not by counting: a translator that merges two of the chunk's blocks into one
    paragraph, or splits one into two, used to move every later paragraph onto
    the wrong block — printed in the wrong place on the page, at the wrong size.

    A block whose translation cannot be pinned down is simply absent: the caller
    keeps its source text rather than printing the wrong paragraph over it. A
    block whose words went into the paragraph above it maps to ``None``: its
    source lines are part of a sentence that was translated there, so they are
    cleared and nothing is printed in their place.

    Alignment runs over the whole chunk, replayed blocks included, and the
    replayed blocks are then skipped: cutting the paragraph list by the overlap
    *count* first, as this used to, moved a paragraph that also carried a new
    block's words off that block whenever the translator merged two of them.
    """
    out: dict[int, str | None] = {}
    for i, chunk in enumerate(chunks):
        text = translations.get(chunk.id)
        refs = _block_refs(chunk)
        if not text or not refs:
            continue
        prev = _dedup_predecessor(chunks[i - 1] if i > 0 else None, translations)
        paragraphs = text.split(CHUNK_SEPARATOR)
        prev_refs = _block_refs(prev) if prev is not None else None
        drop = _overlap_prefix_count(prev_refs, refs) if prev_refs else 0
        drop = min(drop, max(0, len(paragraphs) - 1))
        # The chunk carries its own source text — one piece per block, which is
        # what the chunker joined it from — so the evidence is available without
        # a second argument. A caller that cannot supply it keeps the old
        # position-by-position pairing rather than guessing at an alignment.
        sources = str(getattr(chunk, "text", "") or "").split(CHUNK_SEPARATOR)
        alignable = len(sources) == len(refs) and all(source.strip() for source in sources)
        owners = (
            _align_paragraphs(sources, paragraphs, _ratio_norm(paragraphs, sources))
            if alignable
            else None
        )
        if owners is not None:
            # Which blocks each paragraph carries, and which paragraph a block
            # with no paragraph of its own (``[]``) was folded into.
            owns: dict[int, set[int]] = {}
            folded: dict[int, int] = {}
            last = -1
            for block, owned in enumerate(owners):
                if owned is None:
                    continue  # the translator left it out: its source text stays
                if not owned:
                    if last >= 0:
                        folded[block] = last
                    continue
                last = max(owned)
                for index in owned:
                    owns.setdefault(index, set()).add(block)
            for block, ref in enumerate(refs):
                owned = owners[block]
                if owned is None:
                    continue
                if not owned:
                    if block < drop:
                        continue  # words of a replayed block: already printed
                    holder = folded.get(block)
                    if holder is None:
                        continue
                    if any(other >= drop for other in owns.get(holder, ())):
                        # A block of this chunk owns that paragraph and prints it:
                        # this block's lines are part of it, and would repeat it.
                        out[ref] = None
                    else:
                        # The only paragraph carrying these words was the replayed
                        # block's, which the previous chunk has already printed —
                        # from its own, narrower translation. Printing it here is
                        # a repeat; printing nothing is a hole.
                        out[ref] = paragraphs[holder]
                    continue
                if block < drop:
                    continue  # the previous chunk already emitted this block
                out[ref] = CHUNK_SEPARATOR.join(paragraphs[index] for index in owned)
            continue
        kept, targets = paragraphs[drop:], refs[drop:]
        if not targets:
            continue
        for ref, paragraph in zip(targets, kept):
            out[ref] = paragraph
        if len(kept) > len(targets):
            # The translator produced more paragraphs than the chunk had blocks
            # (it split one): keep them all on the last block instead of
            # dropping the tail.
            out[targets[-1]] = CHUNK_SEPARATOR.join(kept[len(targets) - 1 :])
    return out


#: How short a reply may be relative to its chunk's source before it counts
#: as cut off. Vietnamese runs longer than the English it replaces: every
#: complete chunk-translation measured across two real runs came out at
#: 0.97-1.16 of its source length, while every reply known to have been cut
#: mid-chunk (two different translators, two runs) measured 0.67 or less.
#: 0.85 sits in that gap. A block-level merge or split changes the paragraph
#: count without changing the word count, so it never approaches the floor.
_MIN_COVERAGE_RATIO = 0.85


def coverage_error(chunk: Any, translated: str) -> str | None:
    """Why *translated* cannot be the whole of *chunk*, or ``None``.

    ``finish_reason == "length"`` catches a reply cut at the provider's token
    limit (:class:`TruncatedError`), but an early model stop can end the reply
    mid-word while still reporting ``stop``: the text then simply stops and
    the tail of the chunk is never translated — which shipped whole sections
    in English and deleted them from the Markdown, with no marker and no
    retry. Length is the signal that survives both mechanisms: a translation
    of the whole chunk measures at least as long as its source (in practice
    longer), so a reply under :data:`_MIN_COVERAGE_RATIO` of the source is
    missing words and must be asked again rather than checkpointed.

    The measure is only meaningful for a real provider asked to translate the
    whole chunk; the mock harness answers with a canned pseudo-reply by
    design, so callers decide whether to measure at all.
    """
    source = str(getattr(chunk, "text", "") or "")
    if not source or not translated.strip():
        return None  # nothing to measure against; the empty reply has its own path
    ratio = len(translated) / len(source)
    if ratio >= _MIN_COVERAGE_RATIO:
        return None
    return (
        f"truncated translation: {len(translated)} characters is "
        f"{ratio:.0%} of the {len(source)}-character source "
        f"(a complete reply measures at least {int(_MIN_COVERAGE_RATIO * 100)}%)"
    )


def _ratio_norm(paragraphs: list[str], sources: list[str]) -> float:
    """The typical out/in length ratio of a run: Vietnamese is not English, and
    a book's contents entries are not its prose, so each chunk is its own
    yardstick."""
    ratios = [
        len(paragraph) / len(source)
        for paragraph, source in zip(paragraphs, sources)
        if source.strip()
    ]
    ratios = [r for r in ratios if r > 0]
    return statistics.median(ratios) if ratios else 1.0


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
    # The incremental file is read by the same eyes as the final one, so it
    # carries real Markdown too (see :func:`assemble`).
    return emphasis.to_markdown(CHUNK_SEPARATOR.join(parts))


def _repeat_of(previous: str, first: str) -> bool:
    """True when *first* is *previous* rendered again at a chunk boundary.

    Chunk overlap is translated twice, and the second rendering can be worded
    differently, so a near-duplicate check still has to run. It runs on the one
    pair that can be that duplicate: the last paragraph emitted for the chunk
    before and the first paragraph of the next. Two similar paragraphs *inside* a
    chunk are two paragraphs the author wrote — a pair of exercises differing by
    one number, two clauses differing by one figure — and the document-wide pass
    that used to collapse them deleted real content without leaving a marker.

    The anchors have to agree as well (see :data:`_ANCHOR_RE`): the same sentence
    with a different number, identifier or code marker is a different sentence.
    """
    prev_norm, first_norm = _normalize_para(previous), _normalize_para(first)
    if len(prev_norm) <= 40 or len(first_norm) <= 40:
        return False
    both_toc = bool(_TOC_ENTRY_RE.search(prev_norm) and _TOC_ENTRY_RE.search(first_norm))
    if both_toc:
        return False
    if _anchor_tokens(previous) != _anchor_tokens(first):
        return False
    if prev_norm == first_norm:
        return True
    return difflib.SequenceMatcher(None, prev_norm, first_norm).ratio() >= _NEAR_DUPLICATE_RATIO


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
