"""Tests for webui.store: the config, run, skill and glossary stores."""

from __future__ import annotations

import json

import pytest

from webui.store import (
    EDITABLE_KEYS,
    READONLY_KEYS,
    ConfigStore,
    GlossaryStore,
    RunRecord,
    RunsStore,
    SkillsStore,
    is_secret,
)

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


# ------------------------------- config store -------------------------------
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


# --------------------------------- run store --------------------------------
def test_run_record_round_trip_and_order(tmp_path):
    s = RunsStore(tmp_path)
    for rid, ts in (("b", 2.0), ("a", 1.0)):
        s.write_record(RunRecord(run_id=rid, started=ts, status="done", file="f.md",
                                 src="en", tgt="vi", domain="tech", chunks=3,
                                 exit_code=0, output_path="o.md", mock=True, pid=None,
                                 command=["x"], note=""))
    assert [r.run_id for r in s.list_records()] == ["b", "a"]
    assert s.read_record("a").chunks == 3
    assert s.read_record("missing") is None


def test_read_events_counts_malformed_lines(tmp_path):
    s = RunsStore(tmp_path)
    d = s.dir_for("r1")
    d.mkdir(parents=True)
    (d / "run_20260101_000000.jsonl").write_text(
        '{"step": "ingest", "status": "ok"}\n'
        "not json\n"
        '{"step": "plan", "status": "ok"}\n',
        encoding="utf-8",
    )
    events, nxt, malformed = s.read_events("r1", 0)
    assert [e["step"] for e in events] == ["ingest", "plan"]
    assert nxt == 2 and malformed == 1
    assert s.read_events("r1", 2) == ([], 2, 1)


def test_run_without_a_log_has_no_events(tmp_path):
    s = RunsStore(tmp_path)
    s.write_record(RunRecord(run_id="r2", started=1.0, status="running", file="f.md",
                             src="en", tgt="vi", domain="tech", chunks=1,
                             exit_code=None, output_path="o.md", mock=True, pid=1,
                             command=[], note=""))
    assert s.log_path("r2") is None
    assert s.read_events("r2", 0) == ([], 0, 0)


# -------------------------------- skill store -------------------------------
def test_list_skills_skips_directories_and_bad_json(tmp_path):
    (tmp_path / "en-vi").mkdir(parents=True)
    (tmp_path / "en-vi" / "general.json").write_text(
        json.dumps({"instruction": "i", "mean_score": 8.5, "domain": "general",
                    "src_lang": "en", "tgt_lang": "vi"}), encoding="utf-8")
    (tmp_path / "en-vi" / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "en-vi" / "session_abc.json").write_text("{}", encoding="utf-8")
    (tmp_path / "en-vi" / "a-directory.json").mkdir()

    st = SkillsStore(tmp_path)
    names = [(s["pair"], s["domain"]) for s in st.list_skills()]
    assert names == [("en-vi", "general")]
    assert st.get_skill("en-vi", "general")["mean_score"] == 8.5
    assert st.get_skill("en-vi", "broken") is None


def test_skill_history_is_read(tmp_path):
    (tmp_path / "en-vi").mkdir(parents=True)
    (tmp_path / "en-vi" / "general.json").write_text("{}", encoding="utf-8")
    (tmp_path / "en-vi" / "general.history.jsonl").write_text(
        '{"score": 7.0, "ts": 1}\n{"score": 9.0, "ts": 2}\n', encoding="utf-8")
    assert [h["score"] for h in SkillsStore(tmp_path).history("en-vi", "general")] == [7.0, 9.0]


def test_skill_history_tolerates_a_missing_file(tmp_path):
    (tmp_path / "en-vi").mkdir(parents=True)
    assert SkillsStore(tmp_path).history("en-vi", "general") == []


# ------------------------------ glossary store ------------------------------
def test_glossary_round_trip_with_bom(tmp_path):
    (tmp_path / "g.csv").write_bytes("\ufeffsource,target\nNMT, dịch máy\n".encode("utf-8"))
    st = GlossaryStore(tmp_path)
    assert st.read("g.csv")[1] == ["NMT", "dịch máy"]
    st.write("h.csv", [["a", "b"]])
    assert st.read("h.csv") == [["a", "b"]]
    assert {f["name"] for f in st.list_files()} == {"g.csv", "h.csv"}


def test_glossary_rejects_a_name_that_escapes(tmp_path):
    st = GlossaryStore(tmp_path)
    with pytest.raises(ValueError):
        st.read("../secrets.csv")
