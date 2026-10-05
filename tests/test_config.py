"""Tests for config loading and validation."""

from __future__ import annotations

import pytest

from polyglot_lc.config import (
    AGENT_NAMES,
    ConfigError,
    get_agent_config,
    get_all_agent_configs,
    get_app_config,
    get_embed_config,
)


def test_agent_names():
    assert AGENT_NAMES == ("PLANNER", "TRANSLATOR", "CRITIC", "REFINER", "VERIFIER")


def test_valid_agent_config(mock_env):
    cfg = get_agent_config("PLANNER")
    assert cfg.name == "PLANNER"
    assert cfg.base_url == "http://mock.local/v1"
    assert cfg.model == "mock-model"
    assert cfg.api_key == "sk-mock"
    assert cfg.temperature == 0.2
    assert cfg.is_mock is True


def test_unknown_agent_raises(mock_env):
    with pytest.raises(ConfigError, match="Unknown agent"):
        get_agent_config("INTERN")


def test_missing_base_url_raises(monkeypatch):
    monkeypatch.delenv("PLANNER_BASE_URL", raising=False)
    monkeypatch.setenv("PLANNER_MODEL", "m")
    with pytest.raises(ConfigError, match="PLANNER_BASE_URL"):
        get_agent_config("PLANNER")


def test_missing_model_raises(monkeypatch):
    monkeypatch.setenv("PLANNER_BASE_URL", "http://x/v1")
    monkeypatch.delenv("PLANNER_MODEL", raising=False)
    with pytest.raises(ConfigError, match="PLANNER_MODEL"):
        get_agent_config("PLANNER")


def test_invalid_temperature_raises(monkeypatch):
    monkeypatch.setenv("PLANNER_BASE_URL", "http://x/v1")
    monkeypatch.setenv("PLANNER_MODEL", "m")
    monkeypatch.setenv("PLANNER_TEMPERATURE", "hot")
    with pytest.raises(ConfigError, match="TEMPERATURE"):
        get_agent_config("PLANNER")


def test_invalid_base_url_scheme_raises(monkeypatch):
    monkeypatch.setenv("PLANNER_BASE_URL", "ftp://nope")
    monkeypatch.setenv("PLANNER_MODEL", "m")
    with pytest.raises(ConfigError):
        get_agent_config("PLANNER")


def test_get_all_configs_validates_everything(mock_env):
    cfgs = get_all_agent_configs()
    assert set(cfgs) == set(AGENT_NAMES)


def test_embed_config(mock_env):
    cfg = get_embed_config()
    assert cfg.model == "mock-embed"
    assert cfg.is_mock is True


def test_missing_embed_raises(monkeypatch):
    monkeypatch.delenv("EMBED_BASE_URL", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    with pytest.raises(ConfigError, match="EMBED_BASE_URL"):
        get_embed_config()


def test_app_config_defaults(mock_env):
    cfg = get_app_config()
    assert cfg.critic_threshold == 7.0
    assert cfg.refine_rounds == 2
    assert cfg.chunk_max_tokens == 700


def test_app_config_overrides(monkeypatch, mock_env):
    monkeypatch.setenv("CRITIC_THRESHOLD", "8.5")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "256")
    cfg = get_app_config()
    assert cfg.critic_threshold == 8.5
    assert cfg.chunk_max_tokens == 256


def test_app_config_bad_int_raises(monkeypatch, mock_env):
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "many")
    with pytest.raises(ConfigError, match="CHUNK_MAX_TOKENS"):
        get_app_config()
