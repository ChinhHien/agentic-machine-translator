"""Shared fixtures: mock env vars, sample documents, temp dirs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

SAMPLE_MD = """# Demo Doc

## Intro

Neural Machine Translation (NMT) is widely used. The Berlin-based team
released the system in 2024.

## Details

The NMT pipeline splits documents. NMT quality depends on context.

```python
print("never translate me")
```

Final paragraph mentions NMT again.
"""

AGENT_NAMES = ("PLANNER", "TRANSLATOR", "CRITIC", "REFINER", "VERIFIER")


@pytest.fixture()
def mock_env(monkeypatch, tmp_path):
    """Set mock env for all agents + embedding, isolated dirs."""
    monkeypatch.setenv("LLM_MOCK", "1")
    for name in AGENT_NAMES:
        monkeypatch.setenv(f"{name}_BASE_URL", "http://mock.local/v1")
        monkeypatch.setenv(f"{name}_MODEL", "mock-model")
        monkeypatch.setenv(f"{name}_API_KEY", "sk-mock")
        monkeypatch.setenv(f"{name}_TEMPERATURE", "0.2")
    monkeypatch.setenv("EMBED_BASE_URL", "http://mock.local/v1")
    monkeypatch.setenv("EMBED_MODEL", "mock-embed")
    monkeypatch.setenv("EMBED_API_KEY", "sk-mock")
    monkeypatch.setenv("REASONING_BANK_PATH", str(tmp_path / "rb"))
    monkeypatch.setenv("SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_JSONL", "0")
    # Neutralize optional app-level overrides (e.g. a tuned local .env with
    # CRITIC_THRESHOLD=9.0) so default-value tests stay deterministic.
    for var in ("CRITIC_THRESHOLD", "CHUNK_MAX_TOKENS", "CHUNK_OVERLAP_TOKENS",
                "REFINE_ROUNDS", "RB_TOP_K", "MAX_CONCURRENCY",
                "REFINE_MIN_SCORE"):
        monkeypatch.delenv(var, raising=False)
    # Retry settings: pin deterministic defaults (a tuned local .env may
    # otherwise change retry counts/timings mid-suite).
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY", "2")
    monkeypatch.setenv("LLM_RETRY_MAX_DELAY", "60")
    monkeypatch.setenv("LLM_RETRY_BUFFER_SECONDS", "2")
    return tmp_path


@pytest.fixture()
def sample_md_file(tmp_path):
    p = tmp_path / "sample.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    return p


@pytest.fixture()
def sample_doc(sample_md_file):
    from polyglot_lc.ingest import load_document

    return load_document(sample_md_file)


@pytest.fixture()
def rb_dir(tmp_path):
    return tmp_path / "reasoning_bank"
