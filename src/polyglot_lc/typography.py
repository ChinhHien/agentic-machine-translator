"""What a PDF font's *name* says about the type it sets.

The extractor records how a block was set (weight and slant) and the rebuild
prints the translation in the same face, so both read these names. They have to
agree: the extractor saying "regular" while the rebuild says "bold" is a heading
that comes back thinner than the page it replaces.

The awkward case is TeX. ``CMBX12`` is Computer Modern *bold extended* and
``CMR12`` is its roman; neither name contains the word "bold", so a book
typeset with LaTeX — Think Java, Think Data Structures, most lecture notes —
lost every heading's weight, and ``CMR12`` was read as a *sans* family, which
printed its pages in Arial instead of a serif face.
"""

from __future__ import annotations

import re

#: Words in a font's name that describe a *kind* of type, not its family.
SERIF_MARKERS = (
    "serif",
    "times",
    "georgia",
    "garamond",
    "antiqua",
    "palatino",
    "cambria",
    "century",
    "cmr",
    "cmbx",
    "cmb",
    "cmti",
    "cmmi",
    "cmsy",
    "cmex",
    "latinmodern",
    "lmroman",
    "lmr",
    "nimbusrom",
    "texgyre",
)
MONO_MARKERS = (
    "mono",
    "courier",
    "consol",
    "menlo",
    "terminal",
    "fixed",
    "cmtt",
    "lmtt",
)

#: TeX's weight and slant, spelled into the family name.
_TEX_BOLD_RE = re.compile(r"cmbx|cmb\d|lmbx|lmcbx|(?<![a-z])bx\d")
_TEX_ITALIC_RE = re.compile(r"cmti|cmbxti|cmitti|lmti|lmsl")
#: TeX's own sans faces stay sans.
_TEX_SANS_RE = re.compile(r"cmss|lmss")


def font_style(name: str) -> tuple[bool, bool]:
    """``(bold, italic)`` read off a font's name, TeX families included."""
    lowered = str(name or "").lower()
    bold = "bold" in lowered or bool(_TEX_BOLD_RE.search(lowered))
    italic = (
        "italic" in lowered
        or "oblique" in lowered
        or bool(_TEX_ITALIC_RE.search(lowered))
    )
    return bold, italic


def font_kind(key: str) -> str:
    """``"mono"``, ``"serif"`` or ``"sans"`` for a family key."""
    if any(marker in key for marker in MONO_MARKERS):
        return "mono"
    if _TEX_SANS_RE.search(key):
        return "sans"
    if any(marker in key for marker in SERIF_MARKERS):
        return "serif"
    return "sans"


__all__ = ["MONO_MARKERS", "SERIF_MARKERS", "font_kind", "font_style"]
