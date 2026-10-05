"""Tests for the translation memory."""

from __future__ import annotations

from polyglot_lc.memory.tm import TranslationMemory


def test_store_and_exact_lookup():
    tm = TranslationMemory()
    tm.store("Hello world", "Xin chào thế giới", "en", "vi")
    hits = tm.lookup("Hello world", "en", "vi")
    assert hits == [("Hello world", "Xin chào thế giới")]


def test_fuzzy_lookup():
    tm = TranslationMemory(min_similarity=0.8)
    tm.store("The quick brown fox jumps", "Cáo nâu nhanh", "en", "vi")
    hits = tm.lookup("The quick brown fox jumps over", "en", "vi")
    assert hits and hits[0][1] == "Cáo nâu nhanh"


def test_no_match_below_threshold():
    tm = TranslationMemory(min_similarity=0.9)
    tm.store("alpha beta gamma", "a b c", "en", "vi")
    assert tm.lookup("completely different content here", "en", "vi") == []


def test_empty_input_ignored():
    tm = TranslationMemory()
    tm.store("  ", "target")
    assert len(tm) == 0


def test_lang_pair_isolation():
    tm = TranslationMemory()
    tm.store("hi", "xin chào", "en", "vi")
    assert tm.lookup("hi", "en", "ja") == []
    assert tm.lookup("hi", "en", "vi")
