"""Tests for webui.store: the config, run, skill and glossary stores."""

from __future__ import annotations

from webui.store import EDITABLE_KEYS, READONLY_KEYS, ConfigStore, is_secret

ENV = (
    "# cấu hình mẫu\n"
    "CHUNK_MAX_TOKENS=700\n"
    "\n"
    "TRANSLATOR_API_KEY=sk-CANARY-DO-NOT-LEAK\n"
    "SOME_UNRELATED=1\n"
)


def _store(tmp_path, text=ENV):
    p = tmp_path / ".env"
    p.write_bytes(text.encode("utf-8"))
    return ConfigStore(tmp_path, p)


def test_whitelist_never_exposes_a_secret_name():
    assert EDITABLE_KEYS and READONLY_KEYS
    for key in EDITABLE_KEYS + READONLY_KEYS:
        assert not is_secret(key), key


def test_env_keys_lists_names_only(tmp_path):
    s = _store(tmp_path)
    assert s.env_keys() == {"CHUNK_MAX_TOKENS", "TRANSLATOR_API_KEY", "SOME_UNRELATED"}


def test_no_secret_value_is_ever_returned(tmp_path):
    s = _store(tmp_path)
    blob = repr(s.env_keys())
    assert "sk-CANARY-DO-NOT-LEAK" not in blob
    assert is_secret("TRANSLATOR_API_KEY") and is_secret("x_token")
    assert is_secret("DB_PASSWORD")
    assert not is_secret("CHUNK_MAX_TOKENS")


def test_save_preserves_comments_blank_and_foreign_lines(tmp_path):
    s = _store(tmp_path)
    saved, rejected = s.save({"CHUNK_MAX_TOKENS": 900})
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert saved == ["CHUNK_MAX_TOKENS"] and rejected == []
    assert "# cấu hình mẫu" in text and "SOME_UNRELATED=1" in text
    assert "CHUNK_MAX_TOKENS=900" in text and "700" not in text
    assert "sk-CANARY-DO-NOT-LEAK" in text  # dòng key giữ nguyên, không bị sửa


def test_save_appends_missing_key_and_rejects_unknown(tmp_path):
    s = _store(tmp_path)
    saved, rejected = s.save({"PARTIAL_OUTPUT": 1, "UNKNOWN_SETTING": 1})
    assert saved == ["PARTIAL_OUTPUT"]
    assert rejected == [{"key": "UNKNOWN_SETTING", "reason": "not editable"}]
    assert (tmp_path / ".env").read_text(encoding="utf-8").rstrip().endswith("PARTIAL_OUTPUT=1")


def test_save_preserves_crlf_line_endings(tmp_path):
    s = _store(tmp_path, ENV.replace("\n", "\r\n"))
    s.save({"CHUNK_MAX_TOKENS": 900})
    raw = (tmp_path / ".env").read_bytes()
    assert b"CHUNK_MAX_TOKENS=900\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")  # không có LF trần


def test_save_rejects_a_secret_even_if_someone_lists_it(tmp_path):
    s = _store(tmp_path)
    saved, rejected = s.save({"MY_API_KEY": "x"})
    assert saved == [] and rejected == [{"key": "MY_API_KEY", "reason": "secret"}]


def test_token_budgets_are_not_mistaken_for_credentials():
    """`*_TOKENS` is a budget, not a credential; the spec whitelist says so."""
    assert not is_secret("CHUNK_MAX_TOKENS")
    assert not is_secret("CHUNK_OVERLAP_TOKENS")
    assert is_secret("OPENAI_APIKEY")  # no underscore: caught by the suffix rule


def test_save_renders_booleans_as_one_and_zero(tmp_path):
    s = _store(tmp_path)
    s.save({"PARTIAL_OUTPUT": True, "RESUME": False})
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "PARTIAL_OUTPUT=1" in text and "RESUME=0" in text


def test_missing_env_file_is_treated_as_empty(tmp_path):
    s = ConfigStore(tmp_path, tmp_path / "nope.env")
    assert s.env_keys() == set()
    saved, rejected = s.save({"RESUME": True})
    assert saved == ["RESUME"] and rejected == []
    assert (tmp_path / "nope.env").read_text(encoding="utf-8").strip() == "RESUME=1"
