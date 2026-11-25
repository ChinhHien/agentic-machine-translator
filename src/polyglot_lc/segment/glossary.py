"""Glossary extraction: proper nouns, capitalized terms, repeated terminology."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")

#: Code, not prose: a term that only appears in a fence, an inline span or an
#: identifier (``keySet``) is not terminology the translator has to render.
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:_[A-Za-z0-9_]+|[a-z][A-Z][A-Za-z0-9]*)\b")

#: The words an identifier is built from: ``keySet`` is ``key`` + ``set``.
_IDENTIFIER_SPLIT_RE = re.compile(r"[a-z0-9]+(?=[A-Z])|[A-Z]?[a-z0-9]+|[A-Z]+")


def _prose(text: str) -> str:
    """*text* without code fences and inline code."""
    return _INLINE_CODE_RE.sub(" ", _FENCE_RE.sub(" ", text or ""))


def _identifiers(text: str) -> set[str]:
    """Every identifier and every word inside one, case-folded."""
    names: set[str] = set()
    for match in _IDENTIFIER_RE.finditer(text or ""):
        names.add(match.group(0).casefold())
        names.update(part.casefold() for part in _IDENTIFIER_SPLIT_RE.findall(match.group(0)))
    return names

# Common English words that are capitalized only at sentence start.
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "in",
    "on",
    "at",
    "to",
    "of",
    "for",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "we",
    "our",
    "you",
    "your",
    "they",
    "their",
    "he",
    "she",
    "his",
    "her",
    "not",
    "no",
    "yes",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "i",
    "my",
    "me",
    "am",
    "there",
    "here",
    "what",
    "which",
    "who",
    "how",
    "why",
    "all",
    "each",
    "every",
    "some",
    "any",
    "most",
    "more",
    "less",
    "after",
    "before",
    "between",
    "during",
    "into",
    "over",
    "under",
    "about",
    "such",
    "so",
    "than",
    "too",
    "very",
    "also",
    "however",
    "therefore",
    "figure",
    "table",
    "chapter",
    "section",
    "appendix",
    "note",
    "see",
}


@dataclass
class GlossaryTerm:
    """One glossary candidate."""

    term: str
    count: int
    reason: str  # proper_noun | capitalized | repeated

    def to_dict(self) -> dict[str, object]:
        return {"term": self.term, "count": self.count, "reason": self.reason}


def extract_glossary(
    text: str,
    min_count: int = 2,
    max_terms: int = 40,
) -> list[GlossaryTerm]:
    """Extract glossary candidates from *text*.

    Heuristics:
      * proper nouns / capitalized multi-word phrases (not sentence-initial),
      * acronyms (ALL-CAPS, 2+ chars),
      * terminology repeated >= min_count times.

    Counting happens on **prose only**: code fences, inline code and identifiers
    are dropped first, so ``key`` is not extracted from ``keySet`` and a sample
    inside backticks cannot become a required term. Terms are merged
    case-insensitively (``Set``/``set`` are one entry, spelled the way the
    document uses it most), which is what stopped one glossary from demanding
    two different translations for the same word.
    """
    prose = _prose(text)
    code_names = _identifiers(text)
    spellings: dict[str, Counter[str]] = {}
    counts: dict[str, int] = {}
    reasons: dict[str, str] = {}

    def add(term: str, reason: str, weight: int = 1) -> None:
        key = term.casefold()
        if len(term) < 3 or key in _STOP or key in code_names:
            return
        spellings.setdefault(key, Counter())[term] += weight
        counts[key] = counts.get(key, 0) + weight
        if reasons.get(key) != "proper_noun":
            reasons[key] = reason

    def bump(term: str, reason: str, total: int) -> None:
        """Record a frequency signal without counting the same occurrence twice."""
        key = term.casefold()
        if len(term) < 3 or key in _STOP or key in code_names:
            return
        spellings.setdefault(key, Counter())[term] += 1
        counts[key] = max(counts.get(key, 0), total)
        reasons.setdefault(key, reason)

    # Acronyms first (highest signal).
    for m in re.finditer(r"\b[A-Z]{2,}[0-9]*\b", prose):
        term = m.group(0)
        if term in {"II", "III", "AI"}:
            continue
        add(term, "proper_noun")

    # Capitalized words/phrases not at sentence start.
    # A word is sentence-initial if preceded by '.', '!', '?', '\n' + optional space.
    for m in re.finditer(r"(?<![.!?\n]\s)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", prose):
        add(m.group(0), "capitalized")

    # Repeated terminology (any capitalized or technical word >= min_count).
    words = Counter(w.group(0) for w in _WORD_RE.finditer(prose))
    for word, cnt in words.items():
        if cnt >= min_count and len(word) > 3:
            bump(word, "repeated", cnt)

    # Keep only terms seen at least min_count times or strong proper nouns.
    kept: list[GlossaryTerm] = []
    for key, total in counts.items():
        reason = reasons.get(key, "repeated")
        if total < min_count and reason != "proper_noun":
            continue
        surface = spellings[key].most_common(1)[0][0]
        kept.append(GlossaryTerm(term=surface, count=total, reason=reason))
    kept.sort(key=lambda t: (-t.count, t.term))
    return kept[:max_terms]


def glossary_to_md(terms: list[GlossaryTerm]) -> str:
    """Render glossary as markdown bullet list for prompts."""
    if not terms:
        return "(none)"
    return "\n".join(f"- {t.term} (x{t.count})" for t in terms)


#: Hard cap on glossary entries, so dynamic context cannot grow without bound
#: (insertion order preserved, deterministic).
MAX_GLOSSARY_TERMS = 200

#: The terminology a language pair starts from, before the Planner says
#: anything. Each chunk's prompt is built independently, so without one
#: agreed target form a book calls the same concept "hàm tạo", "hàm dựng" and
#: "hàm khởi tạo" in three chapters. Only core concepts belong here — an
#: identifier is not translated, and forcing a term the domain does not use is
#: worse than the drift.
DEFAULT_GLOSSARIES: dict[tuple[str, str], dict[str, str]] = {
    ("en", "vi"): {
        "constructor": "hàm khởi tạo",
        "instance variable": "biến thực thể",
    },
}


def default_glossary(src: str, tgt: str) -> dict[str, str]:
    """The built-in terminology for a language pair, as a fresh dictionary."""
    return dict(DEFAULT_GLOSSARIES.get((src.strip().lower(), tgt.strip().lower()), {}))


#: The glossary a document deserves grows with it: a fixed 40 leaves most of a
#: 1000-page book's terminology unpinned, while 200 terms on a memo would be
#: noise in every chunk's prompt. One term per :data:`TERMS_PER_CHARS`
#: characters — roughly one per page and a half — with 40 as the floor and the
#: cap above as the ceiling, so a prompt never grows without bound.
GLOSSARY_TERMS_FLOOR = 40
GLOSSARY_TERMS_CEILING = MAX_GLOSSARY_TERMS
TERMS_PER_CHARS = 8_000


def glossary_budget(chars: int) -> int:
    """How many glossary terms a document of *chars* characters deserves."""
    return max(
        GLOSSARY_TERMS_FLOOR,
        min(GLOSSARY_TERMS_CEILING, GLOSSARY_TERMS_FLOOR + max(0, chars) // TERMS_PER_CHARS),
    )


def render_glossary_mapping(mapping: dict[str, str], limit: int = MAX_GLOSSARY_TERMS) -> str:
    """Render a glossary mapping (source => target) as prompt bullets, capped at *limit*."""
    items = list(mapping.items())
    if len(items) > limit:
        items = items[:limit]
    return "\n".join(f"- {k} => {v}" for k, v in items)
