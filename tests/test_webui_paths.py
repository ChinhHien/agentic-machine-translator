"""Tests for webui.paths: repository root and safe name resolution."""

from __future__ import annotations

import pytest

from webui.paths import REPO_ROOT, is_within, safe_child


@pytest.mark.parametrize("bad", ["", "..", "../x", "a/b", "a\\b", "/etc/passwd"])
def test_safe_child_rejects_traversal(tmp_path, bad):
    with pytest.raises(ValueError):
        safe_child(tmp_path, bad)


@pytest.mark.parametrize("good", ["a.csv", "Think_Data.pdf", "a b.md", "dấu-cách.csv"])
def test_safe_child_accepts_plain_names(tmp_path, good):
    assert safe_child(tmp_path, good) == tmp_path / good


def test_repo_root_is_the_project_root():
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_is_within(tmp_path):
    assert is_within(tmp_path, tmp_path / "x")
    assert not is_within(tmp_path, tmp_path.parent / "x")
