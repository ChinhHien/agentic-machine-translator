"""Application configuration: per-agent LLM settings from environment.

Every agent (PLANNER, TRANSLATOR, CRITIC, REFINER, VERIFIER) reads its own
``<NAME>_BASE_URL``, ``<NAME>_MODEL``, ``<NAME>_API_KEY`` and
``<NAME>_TEMPERATURE`` from the environment (or a ``.env`` file), so providers
can be mixed freely (OpenAI, DeepSeek, Anthropic via an OpenAI-compatible
proxy, Ollama, vLLM, ...).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

load_dotenv()

AGENT_NAMES: tuple[str, ...] = (
    "PLANNER",
    "TRANSLATOR",
    "CRITIC",
    "REFINER",
    "VERIFIER",
)

# Hybrid agent grouping.
#
# Group 1 "read-source": Translator, Critic and Refiner all re-read the same
# source chunk (original markdown) and cross-reference it with the plan,
# draft and critique. Keeping them on ONE provider/model family lets the
# provider's native prompt/context cache serve the shared, stable prefix
# (original text + glossary + strategy) instead of re-billing it per agent.
#
# Planner (runs once, doc-level) and Verifier (runs after the final draft,
# reads translation + criteria only) are independent and may use a different
# provider/model.
SOURCE_GROUP_AGENTS: tuple[str, ...] = ("TRANSLATOR", "CRITIC", "REFINER")
AGENT_GROUPS: dict[str, str] = {
    "PLANNER": "planner",
    "TRANSLATOR": "source",
    "CRITIC": "source",
    "REFINER": "source",
    "VERIFIER": "verifier",
}

#: Env prefix holding the shared provider/model for the read-source group.
SOURCE_GROUP_PREFIX = "SOURCE_GROUP_"

# Zero-setup demo defaults used when LLM_MOCK=1 and a value is not provided.
MOCK_BASE_URL = "http://mock.local/v1"
MOCK_TEMPERATURE = 0.2


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def format_validation_error(err: ValidationError, ctx: str) -> str:
    """Render a pydantic ValidationError as a short, human-readable string."""
    parts: list[str] = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e.get("loc", ()))
        parts.append(f"{loc or '<root>'}: {e.get('msg', 'invalid')}")
    return f"{ctx}: " + "; ".join(parts)


class AgentConfig(BaseModel):
    """Validated settings for one agent's LLM endpoint."""

    name: str
    base_url: str
    model: str
    api_key: str = ""
    temperature: float = 0.3
    #: Logical group ("planner" | "source" | "verifier") for reporting/routing.
    group: str = ""
    timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    # Backup endpoint used when the primary keeps failing (rate limits,
    # timeouts, connection errors). Configured via ``<NAME>_BACKUP_*`` vars.
    backup_base_url: str = ""
    backup_model: str = ""
    backup_api_key: str = ""

    @field_validator("base_url", "backup_base_url")
    @classmethod
    def _check_base_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v:
            return ""
        if not v.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return v

    @field_validator("model")
    @classmethod
    def _check_model(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be empty")
        return v

    @model_validator(mode="after")
    def _check_backup_pair(self) -> "AgentConfig":
        """Backup endpoint needs BOTH base_url and model (or neither)."""
        if bool(self.backup_base_url) != bool(self.backup_model):
            raise ValueError(
                "backup endpoint requires both <NAME>_BACKUP_BASE_URL and "
                "<NAME>_BACKUP_MODEL"
            )
        return self

    @property
    def has_backup(self) -> bool:
        """True when a complete backup endpoint is configured."""
        return bool(self.backup_base_url and self.backup_model)

    @property
    def is_mock(self) -> bool:
        """True when mock mode is enabled globally (no real API calls)."""
        return os.getenv("LLM_MOCK", "0").lower() in {"1", "true", "yes"}


def _mock_enabled() -> bool:
    return os.getenv("LLM_MOCK", "0").lower() in {"1", "true", "yes", "on"}


def _read_source_group_env() -> dict[str, str]:
    """Shared provider/model defaults for the read-source group (Translator,
    Critic, Refiner). Individual ``<NAME>_*`` vars always win when present, so
    per-agent overrides remain fully supported."""
    p = SOURCE_GROUP_PREFIX
    return {
        "base_url": os.getenv(f"{p}BASE_URL", "").strip(),
        "model": os.getenv(f"{p}MODEL", "").strip(),
        "api_key": os.getenv(f"{p}API_KEY", "").strip(),
        "temperature": os.getenv(f"{p}TEMPERATURE", "").strip(),
    }


def _agent_from_env(name: str) -> AgentConfig:
    """Build one AgentConfig from ``<NAME>_*`` env vars, raising clearly.

    For the read-source group (Translator/Critic/Refiner), values not set on
    the agent fall back to the shared ``SOURCE_GROUP_*`` vars so the whole
    group can share one caching-capable provider/model in one place.

    In mock mode (``LLM_MOCK=1``) missing BASE_URL/MODEL fall back to demo
    defaults so ``LLM_MOCK=1 polyglot translate ...`` needs no other setup.
    """
    prefix = f"{name}_"
    mock = _mock_enabled()
    base_url = os.getenv(f"{prefix}BASE_URL", "").strip()
    model = os.getenv(f"{prefix}MODEL", "").strip()
    api_key = os.getenv(f"{prefix}API_KEY", "").strip()
    temperature_raw = os.getenv(f"{prefix}TEMPERATURE", "").strip()
    backup_base_url = os.getenv(f"{prefix}BACKUP_BASE_URL", "").strip()
    backup_model = os.getenv(f"{prefix}BACKUP_MODEL", "").strip()
    backup_api_key = os.getenv(f"{prefix}BACKUP_API_KEY", "").strip()

    group = AGENT_GROUPS.get(name, "")
    group_used = False
    if name in SOURCE_GROUP_AGENTS:
        shared = _read_source_group_env()
        if not base_url and shared["base_url"]:
            base_url = shared["base_url"]
            group_used = True
        if not model and shared["model"]:
            model = shared["model"]
            group_used = True
        if not api_key and shared["api_key"]:
            api_key = shared["api_key"]
            group_used = True
        if not temperature_raw and shared["temperature"]:
            temperature_raw = shared["temperature"]
            group_used = True

    missing = [
        field
        for field, value in (("BASE_URL", base_url), ("MODEL", model))
        if not value
    ]
    if missing and not mock:
        missing_vars = ", ".join(f"{prefix}{m}" for m in missing)
        hint = ""
        if name in SOURCE_GROUP_AGENTS:
            group_vars = ", ".join(f"{SOURCE_GROUP_PREFIX}{m}" for m in missing)
            hint = (
                f" (or set {group_vars} to share one provider across the "
                "read-source group)"
            )
        raise ConfigError(
            f"Missing required environment variables for agent {name}: "
            + missing_vars
            + hint
            + ". Add them to your .env (see .env.example), or set LLM_MOCK=1 "
            "for a keyless demo."
        )
    if mock:
        base_url = base_url or MOCK_BASE_URL
        model = model or f"mock-{name.lower()}"
        api_key = api_key or "sk-mock"
        temperature_raw = temperature_raw or str(MOCK_TEMPERATURE)

    temperature = 0.3
    if temperature_raw:
        try:
            temperature = float(temperature_raw)
        except ValueError as exc:
            raise ConfigError(
                f"{prefix}TEMPERATURE={temperature_raw!r} is not a float"
            ) from exc

    try:
        return AgentConfig(
            name=name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature,
            group=group + ("+group" if group_used else ""),
            backup_base_url=backup_base_url,
            backup_model=backup_model,
            backup_api_key=backup_api_key,
        )
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, f"agent {name}")) from exc


class EmbedConfig(BaseModel):
    """Embedding endpoint used by the Reasoning Bank."""

    base_url: str
    model: str
    api_key: str = ""
    timeout: float = Field(default=60.0, gt=0)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return v

    @property
    def is_mock(self) -> bool:
        return os.getenv("LLM_MOCK", "0").lower() in {"1", "true", "yes"}


def _embed_from_env() -> EmbedConfig:
    mock = _mock_enabled()
    base_url = os.getenv("EMBED_BASE_URL", "").strip()
    model = os.getenv("EMBED_MODEL", "").strip()
    api_key = os.getenv("EMBED_API_KEY", "").strip()
    missing = [
        f for f, v in (("EMBED_BASE_URL", base_url), ("EMBED_MODEL", model)) if not v
    ]
    if missing and not mock:
        raise ConfigError(
            "Missing required environment variables: " + ", ".join(missing)
            + ". Add them to your .env (see .env.example), or set LLM_MOCK=1."
        )
    if mock:
        base_url = base_url or MOCK_BASE_URL
        model = model or "mock-embed"
        api_key = api_key or "sk-mock"
    try:
        return EmbedConfig(base_url=base_url, model=model, api_key=api_key)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, "embedding")) from exc


class AppConfig(BaseModel):
    """General application settings (paths, thresholds, chunking)."""

    reasoning_bank_path: Path = Path("./data/reasoning_bank")
    skills_dir: Path = Path("./skills")
    logs_dir: Path = Path("./logs")
    log_jsonl: bool = True
    chunk_max_tokens: int = Field(default=700, gt=0)
    chunk_overlap_tokens: int = Field(default=80, ge=0)
    critic_threshold: float = Field(default=7.0, ge=0, le=10)
    refine_rounds: int = Field(default=2, ge=0)
    rb_top_k: int = Field(default=4, ge=1)
    #: Bounded concurrency across independent chunks (each chunk keeps its
    #: internal Translate->Critic->Refiner->Verify order). Conservative default.
    max_concurrency: int = Field(default=2, ge=1, le=16)
    #: Adaptive refiner gate: a draft scoring >= this skips the Refiner and goes
    #: straight to the Verifier. Defaults to ``critic_threshold`` so behaviour is
    #: unchanged unless raised explicitly.
    refine_min_score: float = Field(default=7.0, ge=0, le=10)
    #: When True the summary tree asks the Planner LLM to summarize every
    #: chunk (one request per chunk). Those per-chunk summaries are never used
    #: downstream, so this defaults to False to avoid wasted requests/tokens.
    summary_tree_chunk_summaries: bool = False


def _get_float_env(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a float") from exc


def _get_int_env(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not an int") from exc


def _get_bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


class RetryConfig(BaseModel):
    """LLM resilience tuning; every field is env-overridable.

    Env vars: ``LLM_MAX_RETRIES``, ``LLM_RETRY_BASE_DELAY``,
    ``LLM_RETRY_MAX_DELAY``, ``LLM_RETRY_BUFFER_SECONDS``.
    """

    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=2.0, gt=0)
    max_delay: float = Field(default=60.0, gt=0)
    buffer_seconds: float = Field(default=2.0, ge=0, le=30)
    #: Fraction of the backoff a retry may skip, in ``[0, 1]``. ``0`` disables
    #: jitter (deterministic); production default comes from ``LLM_RETRY_JITTER``.
    jitter: float = Field(default=0.0, ge=0.0, le=1.0)


def get_retry_config() -> RetryConfig:
    """Return retry settings from env (with sane defaults)."""
    try:
        cfg = RetryConfig(
            max_retries=_get_int_env("LLM_MAX_RETRIES", 3),
            base_delay=_get_float_env("LLM_RETRY_BASE_DELAY", 2.0),
            max_delay=_get_float_env("LLM_RETRY_MAX_DELAY", 60.0),
            buffer_seconds=_get_float_env("LLM_RETRY_BUFFER_SECONDS", 2.0),
            jitter=_get_float_env("LLM_RETRY_JITTER", 0.2),
        )
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, "retry config")) from exc
    if cfg.base_delay > cfg.max_delay:
        raise ConfigError(
            f"LLM_RETRY_BASE_DELAY ({cfg.base_delay}) must be <= "
            f"LLM_RETRY_MAX_DELAY ({cfg.max_delay})"
        )
    return cfg


class CacheConfig(BaseModel):
    """Provider-native prompt/context caching policy.

    * ``auto`` (default): rely on the provider's automatic prefix caching.
      Requests are still ordered stable-prefix-first (source text, glossary,
      strategy) so the provider can cache the prefix, and cache usage metrics
      reported by the API are always recorded. No cache markers are sent.
    * ``explicit``: additionally attach an Anthropic-style ``cache_control``
      breakpoint after the stable prefix, for gateways that require explicit
      cache markers (e.g. Anthropic through an OpenAI-compatible proxy).
    * ``off``: disable explicit markers (metrics are still parsed when the API
      returns them).
    """

    mode: str = "auto"

    @property
    def send_markers(self) -> bool:
        return self.mode == "explicit"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


def get_cache_config() -> CacheConfig:
    """Return the prompt-cache policy from ``PROMPT_CACHE`` (default auto)."""
    raw = os.getenv("PROMPT_CACHE", "").strip().lower()
    if not raw:
        return CacheConfig(mode="auto")
    aliases = {"on": "explicit", "markers": "explicit", "off": "off",
               "auto": "auto", "explicit": "explicit"}
    mode = aliases.get(raw)
    if mode is None:
        raise ConfigError(
            f"PROMPT_CACHE={raw!r} is invalid; expected auto|explicit|off"
        )
    return CacheConfig(mode=mode)


def get_agent_config(name: str) -> AgentConfig:
    """Return the validated config for *name* (e.g. ``"PLANNER"``).

    Raises:
        ConfigError: with a clear message when required env vars are missing
            or invalid. The CLI surfaces this on the Errors panel.
    """
    key = name.strip().upper()
    if key not in AGENT_NAMES:
        raise ConfigError(
            f"Unknown agent {name!r}; expected one of {', '.join(AGENT_NAMES)}"
        )
    return _agent_from_env(key)


def get_all_agent_configs() -> dict[str, AgentConfig]:
    """Return validated configs for every agent (validates all or nothing)."""
    return {name: get_agent_config(name) for name in AGENT_NAMES}


def get_embed_config() -> EmbedConfig:
    """Return the validated embedding config for the Reasoning Bank."""
    return _embed_from_env()


def get_app_config() -> AppConfig:
    """Return general application settings from env (with sane defaults)."""
    kwargs: dict[str, Any] = {}
    if os.getenv("REASONING_BANK_PATH"):
        kwargs["reasoning_bank_path"] = Path(os.environ["REASONING_BANK_PATH"])
    if os.getenv("SKILLS_DIR"):
        kwargs["skills_dir"] = Path(os.environ["SKILLS_DIR"])
    if os.getenv("LOGS_DIR"):
        kwargs["logs_dir"] = Path(os.environ["LOGS_DIR"])
    if os.getenv("LOG_JSONL"):
        kwargs["log_jsonl"] = _get_bool_env("LOG_JSONL", True)
    if os.getenv("CHUNK_MAX_TOKENS"):
        kwargs["chunk_max_tokens"] = _get_int_env("CHUNK_MAX_TOKENS", 700)
    if os.getenv("CHUNK_OVERLAP_TOKENS"):
        kwargs["chunk_overlap_tokens"] = _get_int_env("CHUNK_OVERLAP_TOKENS", 80)
    if os.getenv("CRITIC_THRESHOLD"):
        kwargs["critic_threshold"] = _get_float_env("CRITIC_THRESHOLD", 7.0)
    if os.getenv("REFINE_ROUNDS"):
        kwargs["refine_rounds"] = _get_int_env("REFINE_ROUNDS", 2)
    if os.getenv("RB_TOP_K"):
        kwargs["rb_top_k"] = _get_int_env("RB_TOP_K", 4)
    if os.getenv("MAX_CONCURRENCY"):
        kwargs["max_concurrency"] = _get_int_env("MAX_CONCURRENCY", 2)
    if os.getenv("REFINE_MIN_SCORE"):
        kwargs["refine_min_score"] = _get_float_env("REFINE_MIN_SCORE", 7.0)
    if os.getenv("SUMMARY_TREE_CHUNK_SUMMARIES"):
        kwargs["summary_tree_chunk_summaries"] = _get_bool_env(
            "SUMMARY_TREE_CHUNK_SUMMARIES", False
        )
    try:
        return AppConfig(**kwargs)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, "app config")) from exc
