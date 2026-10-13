"""The sentinel grammar for the bold and italic spans of a source line.

A document's emphasis is lost before the pipeline ever sees it: the ingest
records one face per *block*, taken from its first line, and nothing downstream
could carry a bold phrase inside a sentence. Markdown's own ``**`` cannot fix
that either — the text is handed to a model, and a model that drops, moves or
re-writes an asterisk gives no signal that it did. These four tokens are the
same idea as ``⟦CODE_n⟧``: a small, counted, greppable marker that a
deterministic check can hold the model to.

    ⟦BOLD⟧…⟦/BOLD⟧        ⟦ITALIC⟧…⟦/ITALIC⟧

They live in a module with **no project imports** on purpose: the ingest layer
writes them and the rebuild layer reads them, and ``ingest`` must not depend on
``pipeline``.

What a token does *not* claim: the text between a pair is the emphasised text
in the **language it is written in**, so the same pair survives translation and
still brackets the words a reader would emphasise. Position is the model's to
choose — when its token counts match the source's, its placement is kept.
"""

from __future__ import annotations

import re

BOLD_OPEN = "⟦BOLD⟧"
BOLD_CLOSE = "⟦/BOLD⟧"
ITALIC_OPEN = "⟦ITALIC⟧"
ITALIC_CLOSE = "⟦/ITALIC⟧"

#: Every token the grammar knows, in the order the module documents them.
TOKENS: tuple[str, ...] = (BOLD_OPEN, BOLD_CLOSE, ITALIC_OPEN, ITALIC_CLOSE)

#: One printed run: ``(text, bold, italic)``.
Run = tuple[str, bool, bool]

#: A matched pair of the *same* kind, innermost text in group 2.
_PAIR_RE = re.compile(r"⟦(BOLD|ITALIC)⟧(.*?)⟦/\1⟧", re.DOTALL)

#: Any single token, longest first so no token can be read as part of another.
_TOKEN_RE = re.compile(
    "|".join(re.escape(token) for token in sorted(TOKENS, key=len, reverse=True))
)

#: The state a token turns on, as the ``(bold, italic)`` pair runs carry.
_OPENS: dict[str, tuple[bool, bool]] = {BOLD_OPEN: (True, False), ITALIC_OPEN: (False, True)}
_CLOSES: dict[str, tuple[bool, bool]] = {BOLD_CLOSE: (True, False), ITALIC_CLOSE: (False, True)}

#: No emphasis.
PLAIN: tuple[bool, bool] = (False, False)


def counts(text: str) -> dict[str, int]:
    """How many of each token *text* holds (every token, even at zero)."""
    return {token: text.count(token) for token in TOKENS}


def is_balanced(text: str) -> bool:
    """True when every opener is closed by its own kind, in order.

    Crossing pairs (``⟦BOLD⟧…⟦/ITALIC⟧…⟦/BOLD⟧``) are not balanced, so a walk
    keeps a stack rather than counting.
    """
    stack: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if token in _OPENS:
            stack.append(token)
        elif not stack or _OPENS[stack.pop()] != _CLOSES[token]:
            return False
    return not stack


def strip(text: str) -> str:
    """*text* without any token: the words alone, in order."""
    for token in TOKENS:
        text = text.replace(token, "")
    return text


def to_markdown(text: str) -> str:
    """*text* as real Markdown emphasis, with no token left behind.

    Substitution repeats because a pair can hold another pair; whatever is still
    unmatched afterwards has no emphasis to express and is dropped, so ``⟦``
    never reaches a reader.
    """
    out = text
    while True:
        replaced = _PAIR_RE.sub(
            lambda m: (
                ("**" if m.group(1) == "BOLD" else "*")
                + m.group(2)
                + ("**" if m.group(1) == "BOLD" else "*")
            ),
            out,
        )
        if replaced == out:
            return strip(out)
        out = replaced


def to_runs(text: str) -> list[Run]:
    """*text* as printed runs, one per stretch of one style.

    A token claims no characters of its own, so nothing is ever lost: an
    unmatched token merely leaves the style as it was.
    """
    runs: list[Run] = []
    stack: list[tuple[bool, bool]] = []
    position = 0
    for match in _TOKEN_RE.finditer(text):
        _append(runs, text[position : match.start()], stack[-1] if stack else PLAIN)
        position = match.end()
        token = match.group(0)
        if token in _OPENS:
            stack.append(_OPENS[token])
        elif stack:
            stack.pop()
    _append(runs, text[position:], stack[-1] if stack else PLAIN)
    return runs


def _append(runs: list[Run], text: str, state: tuple[bool, bool]) -> None:
    """Add one stretch, merging it into the previous run when the style matches."""
    if not text:
        return
    if runs and runs[-1][1:] == state:
        runs[-1] = (runs[-1][0] + text, *state)
        return
    runs.append((text, *state))


def agrees_with(source: str, translation: str) -> bool:
    """True when *translation*'s tokens are well formed and match *source*'s counts.

    This is the only licence to print the translation's emphasis: the model may
    place a pair wherever the translated words earn it, but a translation that
    invented or lost a token is not describing the source's emphasis at all.
    """
    return is_balanced(translation) and counts(translation) == counts(source)


__all__ = [
    "BOLD_CLOSE",
    "BOLD_OPEN",
    "ITALIC_CLOSE",
    "ITALIC_OPEN",
    "PLAIN",
    "TOKENS",
    "Run",
    "agrees_with",
    "counts",
    "is_balanced",
    "strip",
    "to_markdown",
    "to_runs",
]
