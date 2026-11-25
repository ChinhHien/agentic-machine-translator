"""Glossary extraction counts prose, and merges case variants of one term."""

from __future__ import annotations

from polyglot_lc.segment.glossary import extract_glossary


def test_case_variants_become_one_term():
    """``Dispatcher``/``dispatcher`` are one word: one entry, not two demands."""
    text = "Each Dispatcher routes. Another Dispatcher retries. Every dispatcher logs."
    terms = [
        term for term in extract_glossary(text, min_count=2) if term.term.casefold() == "dispatcher"
    ]
    assert len(terms) == 1, terms
    assert terms[0].count >= 2


def test_a_name_that_only_lives_in_code_is_not_a_term():
    text = "```python\nh = HashMap()\n```\n"
    assert [t for t in extract_glossary(text, min_count=1) if t.term.casefold() == "hashmap"] == []


def test_prose_terms_are_still_extracted():
    text = "The Dispatcher routes work. The Dispatcher retries work."
    terms = [t for t in extract_glossary(text, min_count=1) if t.term == "Dispatcher"]
    assert terms, "a repeated proper noun is still terminology"
