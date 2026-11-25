"""The glossary override: prose only, and ambiguous terms never fail a chunk.

The override exists because the model is not a reliable glossary lookup. It used
to fire on a term that only appeared inside code (``key`` in ``keySet``, a word
in a code fence), telling the translator to render an identifier — a defect no
retranslation can fix, and one that rejected good chunks in real runs.
"""

from __future__ import annotations

from polyglot_lc.agents.verifier import glossary_defects


def test_a_term_that_only_appears_in_code_is_not_a_defect():
    source = "Call `keySet()` to iterate.\n\n```python\nkey = 1\n```\n"
    assert glossary_defects(source, "Gọi keySet() để lặp.", {"key": "khóa"}) == []


def test_a_prose_term_that_is_also_an_identifier_is_only_minor():
    """``keySet`` makes "key" ambiguous, so it is reported, not enforced."""
    source = "The keySet is a set of keys. The key must be unique."
    defects = glossary_defects(source, "keySet là một tập. key phải là duy nhất.", {"key": "khóa"})
    assert [defect["severity"] for defect in defects] == ["minor"]


def test_a_prose_miss_stays_major():
    source = "The key opens the door."
    defects = glossary_defects(source, "Chìa mở cửa.", {"key": "khóa"})
    assert [defect["severity"] for defect in defects] == ["major"]
    assert defects[0]["expected"] == "khóa"


def test_the_planned_form_satisfies_the_check():
    source = "The key opens the door."
    assert glossary_defects(source, "Khóa mở cửa.", {"key": "khóa"}) == []
