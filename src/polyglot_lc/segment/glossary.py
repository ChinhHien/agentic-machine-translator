"""Glossary extraction: proper nouns, capitalized terms, repeated terminology."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")

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
    """
    candidates: dict[str, GlossaryTerm] = {}

    # Acronyms first (highest signal).
    for m in re.finditer(r"\b[A-Z]{2,}[0-9]*\b", text):
        term = m.group(0)
        if term in {"II", "III", "AI"} or term.lower() in _STOP:
            continue
        candidates.setdefault(term, GlossaryTerm(term=term, count=0, reason="proper_noun"))
        candidates[term].count += 1

    # Capitalized words/phrases not at sentence start.
    # A word is sentence-initial if preceded by '.', '!', '?', '\\n' + optional space.
    for m in re.finditer(r"(?<![.!?\n]\s)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        phrase = m.group(0)
        if phrase.lower() in _STOP or len(phrase) < 3:
            continue
        candidates.setdefault(phrase, GlossaryTerm(term=phrase, count=0, reason="capitalized"))
        candidates[phrase].count += 1

    # Repeated terminology (any capitalized or technical word >= min_count).
    words = Counter(w.group(0) for w in _WORD_RE.finditer(text))
    for word, cnt in words.items():
        if cnt >= min_count and len(word) > 3 and word.lower() not in _STOP:
            if word not in candidates:
                candidates[word] = GlossaryTerm(term=word, count=cnt, reason="repeated")
            else:
                candidates[word].count = max(candidates[word].count, cnt)

    # Keep only terms seen at least min_count times or strong proper nouns.
    kept = [t for t in candidates.values() if t.count >= min_count or t.reason == "proper_noun"]
    kept.sort(key=lambda t: (-t.count, t.term))
    return kept[:max_terms]


def glossary_to_md(terms: list[GlossaryTerm]) -> str:
    """Render glossary as markdown bullet list for prompts."""
    if not terms:
        return "(none)"
    return "\n".join(f"- {t.term} (x{t.count})" for t in terms)


#: Hard cap on user-supplied glossary entries so dynamic context cannot grow
#: without bound with a large CSV (insertion order preserved, deterministic).
MAX_USER_GLOSSARY_TERMS = 200


def render_user_glossary(mapping: dict[str, str], limit: int = MAX_USER_GLOSSARY_TERMS) -> str:
    """Render a user glossary mapping as prompts bullets, capped at *limit*."""
    items = list(mapping.items())
    if len(items) > limit:
        items = items[:limit]
    return "\n".join(f"- {k} => {v}" for k, v in items)


def load_glossary_csv(path: str) -> dict[str, str]:
    """Load a user CSV glossary mapping source -> target (columns: src,tgt)."""
    import csv
    from pathlib import Path

    mapping: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Glossary file not found: {path}")
    with p.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                mapping[row[0].strip()] = row[1].strip()
    return mapping
