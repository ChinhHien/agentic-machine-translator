"""Tests for the emphasis sentinel grammar (bold/italic spans in a line).

The grammar is deliberately dumb: four tokens, matched pairs, no Markdown. That
is what makes it checkable — a model can drop or move a ``**`` and nothing can
tell, but a missing ``⟦/BOLD⟧`` is a counted token.
"""

from __future__ import annotations

from polyglot_lc import emphasis


def test_counts_counts_each_token():
    text = "⟦BOLD⟧a⟦/BOLD⟧ ⟦ITALIC⟧b⟦/ITALIC⟧ ⟦BOLD⟧c⟦/BOLD⟧"
    assert emphasis.counts(text)["⟦BOLD⟧"] == 2
    assert emphasis.counts(text)["⟦/ITALIC⟧"] == 1
    assert emphasis.counts("plain")["⟦ITALIC⟧"] == 0


def test_is_balanced_rejects_a_lone_token_and_accepts_nesting():
    assert emphasis.is_balanced("⟦BOLD⟧a⟦/BOLD⟧") is True
    assert emphasis.is_balanced("⟦BOLD⟧a⟦ITALIC⟧b⟦/ITALIC⟧c⟦/BOLD⟧") is True
    assert emphasis.is_balanced("⟦BOLD⟧a") is False
    assert emphasis.is_balanced("⟦BOLD⟧a⟦/ITALIC⟧") is False


def test_strip_keeps_every_character():
    text = "⟦BOLD⟧Eocortex ST⟦/BOLD⟧ được thiết kế để"
    assert emphasis.strip(text) == "Eocortex ST được thiết kế để"


def test_to_markdown_makes_real_markdown_and_never_leaves_a_token():
    assert emphasis.to_markdown("a ⟦BOLD⟧b⟦/BOLD⟧ c") == "a **b** c"
    assert emphasis.to_markdown("a ⟦ITALIC⟧b⟦/ITALIC⟧ c") == "a *b* c"
    # Nested: the inner pair must survive the outer substitution.
    assert emphasis.to_markdown("⟦BOLD⟧a⟦ITALIC⟧b⟦/ITALIC⟧c⟦/BOLD⟧") == "**a*b*c**"
    # A lone token carries no emphasis to convert: it is dropped, not printed.
    assert "⟦" not in emphasis.to_markdown("a ⟦BOLD⟧b")


def test_to_runs_marks_only_the_emphasised_span():
    runs = emphasis.to_runs("A ⟦BOLD⟧B⟦/BOLD⟧ C")
    assert runs == [("A ", False, False), ("B", True, False), (" C", False, False)]


def test_to_runs_never_leaves_a_token():
    assert "⟦" not in "".join(text for text, _, _ in emphasis.to_runs("⟦BOLD⟧a⟦/ITALIC⟧"))


def test_agrees_with_requires_equal_counts():
    assert emphasis.agrees_with("⟦BOLD⟧a⟦/BOLD⟧", "⟦BOLD⟧a⟦/BOLD⟧") is True
    assert emphasis.agrees_with("plain", "⟦BOLD⟧a⟦/BOLD⟧") is False
    assert emphasis.agrees_with("⟦BOLD⟧a⟦/BOLD⟧", "a ⟦BOLD⟧b") is False
