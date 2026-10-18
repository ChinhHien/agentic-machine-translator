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


# ------------------------- workflow v2 defaults -----------------------------
def test_workflow_defaults(mock_env):
    cfg = get_app_config()
    assert cfg.unified_planner is True
    assert cfg.adaptive_quality is True
    assert cfg.fast_path_max_complexity == 0.35
    assert cfg.fast_path_min_score == 9.0
    assert cfg.fast_path_skip_llm_verify is False
    assert cfg.resume is False
    assert cfg.partial_output is False
    assert cfg.consistency_check is True
    assert cfg.consistency_max_patches == 5
    assert cfg.checkpoint_dir is not None


def test_workflow_overrides(mock_env, monkeypatch):
    monkeypatch.setenv("UNIFIED_PLANNER", "0")
    monkeypatch.setenv("ADAPTIVE_QUALITY", "0")
    monkeypatch.setenv("FAST_PATH_MIN_SCORE", "8")
    monkeypatch.setenv("FAST_PATH_MAX_COMPLEXITY", "0.5")
    monkeypatch.setenv("FAST_PATH_SKIP_LLM_VERIFY", "1")
    monkeypatch.setenv("RESUME", "1")
    monkeypatch.setenv("PARTIAL_OUTPUT", "1")
    monkeypatch.setenv("CONSISTENCY_CHECK", "0")
    monkeypatch.setenv("CONSISTENCY_MAX_PATCHES", "2")
    cfg = get_app_config()
    assert cfg.unified_planner is False
    assert cfg.adaptive_quality is False
    assert cfg.fast_path_min_score == 8.0
    assert cfg.fast_path_max_complexity == 0.5
    assert cfg.fast_path_skip_llm_verify is True
    assert cfg.resume is True
    assert cfg.partial_output is True
    assert cfg.consistency_check is False
    assert cfg.consistency_max_patches == 2


def test_checkpoint_dir_can_be_disabled(mock_env, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DIR", "off")
    assert get_app_config().checkpoint_dir is None
    monkeypatch.setenv("CHECKPOINT_DIR", "")
    assert get_app_config().checkpoint_dir is None
    monkeypatch.setenv("CHECKPOINT_DIR", "./elsewhere")
    assert str(get_app_config().checkpoint_dir) == "elsewhere"


def test_fast_path_complexity_out_of_range(mock_env, monkeypatch):
    monkeypatch.setenv("FAST_PATH_MAX_COMPLEXITY", "1.5")
    with pytest.raises(ConfigError, match="FAST_PATH_MAX_COMPLEXITY|fast_path"):
        get_app_config()


# ---------------------------- output format ---------------------------------
def test_output_format_defaults_to_markdown(mock_env):
    cfg = get_app_config()
    assert cfg.output_format == "md"
    assert cfg.partial_output is False


def test_output_format_does_not_depend_on_partial_output(mock_env, monkeypatch):
    """OUTPUT_FORMAT must be readable on its own.

    The two settings used to share one ``if`` block, so choosing a format did
    nothing unless PARTIAL_OUTPUT happened to be set as well.
    """
    monkeypatch.setenv("OUTPUT_FORMAT", "both")
    cfg = get_app_config()
    assert cfg.output_format == "both"
    assert cfg.partial_output is False


def test_an_unknown_output_format_is_rejected(mock_env, monkeypatch):
    monkeypatch.setenv("OUTPUT_FORMAT", "docx")
    with pytest.raises(ConfigError, match="OUTPUT_FORMAT|output_format"):
        get_app_config()


# ---------------------------- chunk retry wait -------------------------------
def test_chunk_retry_wait_has_a_default_and_is_env_overridable(mock_env, monkeypatch):
    """A chunk that fails waits before its next attempt; how long is tunable."""
    cfg = get_app_config()
    assert (cfg.chunk_retry_base_delay, cfg.chunk_retry_max_delay) == (5.0, 120.0)

    monkeypatch.setenv("CHUNK_RETRY_BASE_DELAY", "0.5")
    monkeypatch.setenv("CHUNK_RETRY_MAX_DELAY", "30")
    cfg = get_app_config()
    assert (cfg.chunk_retry_base_delay, cfg.chunk_retry_max_delay) == (0.5, 30.0)


def test_a_negative_chunk_retry_wait_is_rejected(mock_env, monkeypatch):
    monkeypatch.setenv("CHUNK_RETRY_BASE_DELAY", "0")
    with pytest.raises(ConfigError, match="CHUNK_RETRY_BASE_DELAY|chunk_retry_base_delay"):
        get_app_config()
