"""Tests for webui.probe: settings, chunk planning, text and the memory report.

The probe is a terminal tool that runs the pipeline in a child process. It is
the *only* place the web UI can observe the pipeline from, so what it refuses
to report matters as much as what it does: `--config` never returns a value for
a name that looks like a credential, and `--memory` never writes to the bank.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from webui.paths import REPO_ROOT

PROBE = REPO_ROOT / "webui" / "probe.py"


def _probe_module():
    """The probe's own field maps, read as source rather than imported.

    ``probe.py`` runs with its own directory on ``sys.path``, so it cannot
    import the package it lives in; parsing it here is what keeps this test
    honest about the names the probe reports.
    """
    import ast

    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    found: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            found[node.target.id] = {
                key.value: True for key in node.value.keys if isinstance(key, ast.Constant)
            }
    return found


EDITABLE_FIELDS = _probe_module()["EDITABLE_FIELDS"]
READONLY_FIELDS = _probe_module()["READONLY_FIELDS"]

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
    assert set(data["values"]) == set(EDITABLE_FIELDS) | set(READONLY_FIELDS) | {"LLM_MOCK"}
    assert set(data["sources"]) == set(data["values"])
    assert set(data["sources"].values()) <= {"env", "dotenv", "default"}
    # No credential-shaped name is ever reported, not even as a name.
    assert not [name for name in data["values"] if name.upper().endswith(("API_KEY", "TOKEN"))]


def test_config_does_not_need_credentials(tmp_path, monkeypatch):
    """Reading settings must work with an empty environment: no keys required."""
    bare = {
        "PATH": os.environ.get("PATH", ""),
        "SystemRoot": os.environ.get("SystemRoot", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    proc = _run("--config", env=bare)
    assert proc.returncode == 0, proc.stderr
    values = json.loads(proc.stdout)["values"]
    assert set(values) == set(EDITABLE_FIELDS) | set(READONLY_FIELDS) | {"LLM_MOCK"}


def test_text_mode_returns_the_document_the_translator_will_read(tmp_path):
    """The side-by-side pane compares against *this* text, so it has to be the
    extractor's own output — not a second reading of the same PDF."""
    from pdfdoc import write_pdf

    doc = write_pdf(tmp_path / "d.pdf", "# Title\n\nA paragraph of plain prose.\n")
    proc = _run("--text", str(doc), env=PLAN_ENV)
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(proc.stdout)
    assert payload["title"] == "d.pdf"
    assert payload["blocks"] >= 2
    assert "Title" in payload["text"] and "plain prose" in payload["text"]
    assert payload["chars"] == len(payload["text"])


def test_text_mode_on_a_missing_file_reports_the_path(tmp_path):
    proc = _run("--text", "nope.pdf", env=PLAN_ENV)
    assert proc.returncode != 0 and proc.stdout.strip() == ""
    assert "nope.pdf" in proc.stderr


def test_memory_mode_reads_an_empty_bank_without_creating_anything(tmp_path):
    """A read-only report must not touch the bank: pointing it at an empty
    directory has to come back clean and leave that directory empty."""
    bank = tmp_path / "bank"
    bank.mkdir()
    proc = _run("--memory", env={**PLAN_ENV, "REASONING_BANK_PATH": str(bank)})
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(proc.stdout)
    assert payload["total"] == 0 and payload["pairs"] == []
    assert payload["path"] == str(bank)
    assert list(bank.iterdir()) == [], "the report wrote into the bank"


def test_memory_mode_reports_what_the_bank_holds(tmp_path):
    """A bank with a collection in it is summarised per language pair."""
    import chromadb

    bank = tmp_path / "bank"
    client = chromadb.PersistentClient(path=str(bank))
    collection = client.get_or_create_collection(
        "reasoning_traces", metadata={"hnsw:space": "cosine"}
    )
    for index, score in enumerate((0.8, 0.9)):
        collection.upsert(
            ids=[f"t{index}"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=[f"en->vi: strategy {index}"],
            metadatas=[
                {
                    "src_lang": "en",
                    "tgt_lang": "vi",
                    "strategy": f"strategy {index}",
                    "score": score,
                    "ts": index,
                }
            ],
        )

    proc = _run("--memory", env={**PLAN_ENV, "REASONING_BANK_PATH": str(bank)})
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(proc.stdout)
    assert payload["total"] == 2 and payload["sampled"] == 2
    (pair,) = payload["pairs"]
    assert pair["pair"] == "en-vi"
    assert pair["traces"] == 2
    assert pair["mean_score"] == 0.85
    # Newest first, and the whole point of the panel: it shows the strategies.
    assert [row["score"] for row in pair["recent"]] == [0.9, 0.8]
    assert pair["recent"][0]["strategy"].endswith("strategy 1")


def test_plan_chunk_count_matches_the_real_chunker(tmp_path):
    # Input is PDF-only, so the document has to be rendered into a real one.
    from pdfdoc import write_pdf

    doc = write_pdf(
        tmp_path / "d.pdf",
        "\n\n".join(
            f"Paragraph {i} describes a distinct step of the workflow in plain prose."
            for i in range(12)
        ),
    )

    proc = _run(
        "--plan", str(doc), "--src", "vi", "--tgt", "en", "--max-tokens", "30", env=PLAN_ENV
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)

    from polyglot_lc.ingest import load_document
    from polyglot_lc.segment.chunker import chunk_document

    expected = len(chunk_document(load_document(doc), max_tokens=30, model="mock-model"))
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
