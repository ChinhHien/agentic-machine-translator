"""Tests for webui.probe: effective configuration and per-file chunk planning."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from webui.paths import REPO_ROOT
from webui.store import EDITABLE_KEYS, READONLY_KEYS

PROBE = REPO_ROOT / "webui" / "probe.py"

#: Deterministic agent settings for the child, so the chunk count cannot depend
#: on whatever `.env` happens to hold on this machine.
PLAN_ENV = {
    **os.environ,
    "LLM_MOCK": "1",
    "TRANSLATOR_BASE_URL": "http://mock.local/v1",
    "TRANSLATOR_MODEL": "mock-model",
    "TRANSLATOR_API_KEY": "sk-mock",
}


def _run(*args, env=None):
    return subprocess.run(
        [sys.executable, str(PROBE), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,  # the caller asserts on returncode itself
    )


def test_config_reports_the_whitelist_and_its_source(tmp_path):
    proc = _run("--config")
    assert proc.returncode == 0, proc.stderr

    data = json.loads(proc.stdout)
    assert set(data["values"]) == set(EDITABLE_KEYS) | set(READONLY_KEYS)
    assert set(data["sources"]) == set(data["values"])
    assert set(data["sources"].values()) <= {"env", "dotenv", "default"}
    # No credential-shaped name is ever reported, not even as a name.
    assert not [name for name in data["values"] if name.upper().endswith("API_KEY")]


def test_config_does_not_need_credentials(tmp_path, monkeypatch):
    """Reading settings must work with an empty environment: no keys required."""
    bare = {
        "PATH": os.environ.get("PATH", ""),
        "SystemRoot": os.environ.get("SystemRoot", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    proc = _run("--config", env=bare)
    assert proc.returncode == 0, proc.stderr
    assert set(json.loads(proc.stdout)["values"]) == set(EDITABLE_KEYS) | set(READONLY_KEYS)


def test_plan_chunk_count_matches_the_real_chunker(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text("# T\n\n" + "một câu. " * 400, encoding="utf-8")

    proc = _run("--plan", str(doc), "--src", "vi", "--tgt", "en",
                "--max-tokens", "200", env=PLAN_ENV)
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)

    from polyglot_lc.ingest import load_document
    from polyglot_lc.segment.chunker import chunk_document

    expected = len(chunk_document(load_document(doc), max_tokens=200, model="mock-model"))
    assert plan["chunks"] == expected > 1
    assert plan["model"] == "mock-model"
    assert plan["output_path"].endswith("d.en.md")


def test_plan_on_a_missing_file_exits_nonzero_with_stderr_only():
    proc = _run("--plan", "does-not-exist.md", "--src", "vi", "--tgt", "en")
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""
    assert proc.stderr.strip()
    # Naming the input distinguishes probe's own error from the interpreter
    # merely failing to find probe.py itself.
    assert "does-not-exist.md" in proc.stderr
