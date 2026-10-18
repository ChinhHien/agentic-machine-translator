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

#: Default wall-clock timeout per LLM request (seconds). Overridable globally
#: via ``LLM_TIMEOUT`` (applies to every agent/embedding that doesn't set its
#: own ``<NAME>_TIMEOUT`` / ``EMBED_TIMEOUT``). Kept modest so a hung endpoint
#: can't stall the whole pipeline for minutes on end.
DEFAULT_TIMEOUT = 45.0


def _default_timeout() -> float:
    """Global request timeout: ``LLM_TIMEOUT`` env or :data:`DEFAULT_TIMEOUT`."""
    raw = os.getenv("LLM_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT
    return _get_float_env("LLM_TIMEOUT", DEFAULT_TIMEOUT)


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
    timeout: float = Field(default=DEFAULT_TIMEOUT, gt=0)
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
    def _check_backup_pair(self) -> AgentConfig:
        """Backup endpoint needs BOTH base_url and model (or neither)."""
        if bool(self.backup_base_url) != bool(self.backup_model):
            raise ValueError(
                "backup endpoint requires both <NAME>_BACKUP_BASE_URL and <NAME>_BACKUP_MODEL"
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
    # Per-agent timeout override (e.g., VERIFIER_TIMEOUT=30); falls back to
    # LLM_TIMEOUT / DEFAULT_TIMEOUT when not set.
    timeout_raw = os.getenv(f"{prefix}TIMEOUT", "").strip()

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

    missing = [field for field, value in (("BASE_URL", base_url), ("MODEL", model)) if not value]
    if missing and not mock:
        missing_vars = ", ".join(f"{prefix}{m}" for m in missing)
        hint = ""
        if name in SOURCE_GROUP_AGENTS:
            group_vars = ", ".join(f"{SOURCE_GROUP_PREFIX}{m}" for m in missing)
            hint = f" (or set {group_vars} to share one provider across the read-source group)"
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
            raise ConfigError(f"{prefix}TEMPERATURE={temperature_raw!r} is not a float") from exc
    if timeout_raw:
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise ConfigError(f"{prefix}TIMEOUT={timeout_raw!r} is not a float") from exc
    else:
        # Global default so one hung endpoint can't stall the whole pipeline
        # for minutes. Per-agent <NAME>_TIMEOUT always wins when present.
        timeout = _default_timeout()

    try:
        return AgentConfig(
            name=name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature,
            group=group + ("+group" if group_used else ""),
            timeout=timeout,
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
    timeout: float = Field(default=DEFAULT_TIMEOUT, gt=0)

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
    missing = [f for f, v in (("EMBED_BASE_URL", base_url), ("EMBED_MODEL", model)) if not v]
    if missing and not mock:
        raise ConfigError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Add them to your .env (see .env.example), or set LLM_MOCK=1."
        )
    if mock:
        base_url = base_url or MOCK_BASE_URL
        model = model or "mock-embed"
        api_key = api_key or "sk-mock"
    timeout_raw = os.getenv("EMBED_TIMEOUT", "").strip()
    if timeout_raw:
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise ConfigError(f"EMBED_TIMEOUT={timeout_raw!r} is not a float") from exc
    else:
        timeout = _default_timeout()
    try:
        return EmbedConfig(base_url=base_url, model=model, api_key=api_key, timeout=timeout)
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
    max_concurrency: int = Field(default=1, ge=1, le=16)
    #: Adaptive refiner gate: a draft scoring >= this skips the Refiner and goes
    #: straight to the Verifier. Defaults to ``critic_threshold`` so behaviour is
    #: unchanged unless raised explicitly.
    refine_min_score: float = Field(default=7.0, ge=0, le=10)
    #: When True the summary tree asks the Planner LLM to summarize every
    #: chunk (one request per chunk). Those per-chunk summaries are never used
    #: downstream, so this defaults to False to avoid wasted requests/tokens.
    summary_tree_chunk_summaries: bool = False

    # ------------------------------------------------------------------
    # Workflow knobs (see the README's architecture section)
    # ------------------------------------------------------------------
    #: ONE Planner request produces summary + strategy + glossary + complexity +
    #: section notes. Set ``UNIFIED_PLANNER=0`` to fall back to the legacy
    #: summary-tree-then-plan path.
    unified_planner: bool = True
    #: Adaptive quality pipeline: route simple, confidently-translated chunks
    #: straight to the Verifier instead of Critic -> Refiner -> Verifier.
    adaptive_quality: bool = True
    #: Fast path gate — a chunk is simple below this complexity (0..1)...
    fast_path_max_complexity: float = Field(default=0.35, ge=0, le=1)
    #: ...and confident at or above this deterministic draft score (0..10).
    fast_path_min_score: float = Field(default=9.0, ge=0, le=10)
    #: Also skip the paid LLM verification on fast-path chunks (hard checks only).
    fast_path_skip_llm_verify: bool = False
    #: Where per-chunk checkpoints live. ``None`` (``CHECKPOINT_DIR=off``)
    #: disables crash-resume entirely.
    checkpoint_dir: Path | None = Path("./data/checkpoints")
    #: Reuse checkpoints from an interrupted run instead of re-translating.
    resume: bool = False
    #: What the run writes: ``md`` (translated Markdown, the default), ``pdf``
    #: (the source PDF rebuilt in place with the translation) or ``both``.
    output_format: str = "md"
    #: Unicode TTF used to print the translation into the rebuilt PDF; empty
    #: means "pick the first available font with Vietnamese coverage".
    pdf_font_path: str = ""
    #: Rewrite the output file as chunks complete (long runs only).
    partial_output: bool = False
    #: Cross-chunk terminology consistency pass after translation.
    consistency_check: bool = True
    #: Upper bound on consistency patch requests per document.
    consistency_max_patches: int = Field(default=5, ge=0, le=100)
    #: How long to wait before a chunk whose step hit provider errors is run
    #: again; doubles with each attempt (see ``CHUNK_RETRY_MAX_DELAY``).
    chunk_retry_base_delay: float = Field(default=5.0, gt=0)
    #: Ceiling on that wait. A chunk is never abandoned, so this is the pace at
    #: which an outage is waited out rather than a deadline for giving up.
    chunk_retry_max_delay: float = Field(default=120.0, gt=0)

    @field_validator("output_format")
    @classmethod
    def _check_output_format(cls, v: str) -> str:
        value = v.strip().lower()
        if value not in {"md", "pdf", "both"}:
            raise ValueError("must be md, pdf or both")
        return value


def _get_str_env(key: str, default: str) -> str:
    raw = os.getenv(key, "")
    return raw.strip() if raw.strip() else default


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
    ``LLM_RETRY_MAX_DELAY``, ``LLM_RETRY_BUFFER_SECONDS``, ``LLM_RETRY_JITTER``,
    ``LLM_RETRY_CONCURRENCY_WAIT``, ``LLM_RETRY_SLOT_REBUILD_AFTER`` and
    ``LLM_PROVIDER_CONCURRENCY``.
    """

    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=2.0, gt=0)
    max_delay: float = Field(default=60.0, gt=0)
    buffer_seconds: float = Field(default=2.0, ge=0, le=30)
    #: Fraction of the backoff a retry may skip, in ``[0, 1]``. ``0`` disables
    #: jitter (deterministic); production default comes from ``LLM_RETRY_JITTER``.
    jitter: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Shortest wait before retrying a request the provider rejected because
    #: another one from this process was still in flight. Exponential backoff
    #: (2 s) only re-collides with the request that is still running server
    #: side, so this floor is deliberately much longer than the base delay.
    concurrency_wait: float = Field(default=10.0, ge=0.0, le=120.0)
    #: Consecutive 429 "another request still holds the slot" rejections that
    #: trigger rebuilding the HTTP client with a fresh connection pool. A
    #: stalled request abandoned mid-flight can leave the provider counting it,
    #: and retries through the same keep-alive connection keep colliding.
    slot_rebuild_after: int = Field(default=3, ge=1, le=10)
    #: Most requests this process keeps in flight per provider ``base_url``.
    #: ``1`` matches a free tier that serves a single simultaneous request;
    #: raise it to the provider's real limit. This is a queue, not a throttle:
    #: queued calls wait their turn instead of failing.
    provider_concurrency: int = Field(default=1, ge=1, le=32)


def get_retry_config() -> RetryConfig:
    """Return retry settings from env (with sane defaults)."""
    try:
        cfg = RetryConfig(
            max_retries=_get_int_env("LLM_MAX_RETRIES", 3),
            base_delay=_get_float_env("LLM_RETRY_BASE_DELAY", 2.0),
            max_delay=_get_float_env("LLM_RETRY_MAX_DELAY", 60.0),
            buffer_seconds=_get_float_env("LLM_RETRY_BUFFER_SECONDS", 2.0),
            jitter=_get_float_env("LLM_RETRY_JITTER", 0.2),
            concurrency_wait=_get_float_env("LLM_RETRY_CONCURRENCY_WAIT", 10.0),
            slot_rebuild_after=_get_int_env("LLM_RETRY_SLOT_REBUILD_AFTER", 3),
            provider_concurrency=_get_int_env("LLM_PROVIDER_CONCURRENCY", 1),
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
    aliases = {
        "on": "explicit",
        "markers": "explicit",
        "off": "off",
        "auto": "auto",
        "explicit": "explicit",
    }
    mode = aliases.get(raw)
    if mode is None:
        raise ConfigError(f"PROMPT_CACHE={raw!r} is invalid; expected auto|explicit|off")
    return CacheConfig(mode=mode)


def get_agent_config(name: str) -> AgentConfig:
    """Return the validated config for *name* (e.g. ``"PLANNER"``).

    Raises:
        ConfigError: with a clear message when required env vars are missing
            or invalid. The CLI surfaces this on the Errors panel.
    """
    key = name.strip().upper()
    if key not in AGENT_NAMES:
        raise ConfigError(f"Unknown agent {name!r}; expected one of {', '.join(AGENT_NAMES)}")
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
        kwargs["max_concurrency"] = _get_int_env("MAX_CONCURRENCY", 1)
    if os.getenv("REFINE_MIN_SCORE"):
        kwargs["refine_min_score"] = _get_float_env("REFINE_MIN_SCORE", 7.0)
    if os.getenv("SUMMARY_TREE_CHUNK_SUMMARIES"):
        kwargs["summary_tree_chunk_summaries"] = _get_bool_env(
            "SUMMARY_TREE_CHUNK_SUMMARIES", False
        )
    if os.getenv("UNIFIED_PLANNER"):
        kwargs["unified_planner"] = _get_bool_env("UNIFIED_PLANNER", True)
    if os.getenv("ADAPTIVE_QUALITY"):
        kwargs["adaptive_quality"] = _get_bool_env("ADAPTIVE_QUALITY", True)
    if os.getenv("FAST_PATH_MAX_COMPLEXITY"):
        kwargs["fast_path_max_complexity"] = _get_float_env("FAST_PATH_MAX_COMPLEXITY", 0.35)
    if os.getenv("FAST_PATH_MIN_SCORE"):
        kwargs["fast_path_min_score"] = _get_float_env("FAST_PATH_MIN_SCORE", 9.0)
    if os.getenv("FAST_PATH_SKIP_LLM_VERIFY"):
        kwargs["fast_path_skip_llm_verify"] = _get_bool_env("FAST_PATH_SKIP_LLM_VERIFY", False)
    if "CHECKPOINT_DIR" in os.environ:
        raw_dir = os.environ["CHECKPOINT_DIR"].strip()
        kwargs["checkpoint_dir"] = (
            None if raw_dir.lower() in {"", "off", "none", "0"} else Path(raw_dir)
        )
    if os.getenv("RESUME"):
        kwargs["resume"] = _get_bool_env("RESUME", False)
    if os.getenv("OUTPUT_FORMAT"):
        kwargs["output_format"] = _get_str_env("OUTPUT_FORMAT", "md") or "md"
    if os.getenv("PDF_FONT_PATH"):
        kwargs["pdf_font_path"] = _get_str_env("PDF_FONT_PATH", "")
    if os.getenv("PARTIAL_OUTPUT"):
        kwargs["partial_output"] = _get_bool_env("PARTIAL_OUTPUT", False)
    if os.getenv("CONSISTENCY_CHECK"):
        kwargs["consistency_check"] = _get_bool_env("CONSISTENCY_CHECK", True)
    if os.getenv("CONSISTENCY_MAX_PATCHES"):
        kwargs["consistency_max_patches"] = _get_int_env("CONSISTENCY_MAX_PATCHES", 5)
    if os.getenv("CHUNK_RETRY_BASE_DELAY"):
        kwargs["chunk_retry_base_delay"] = _get_float_env("CHUNK_RETRY_BASE_DELAY", 5.0)
    if os.getenv("CHUNK_RETRY_MAX_DELAY"):
        kwargs["chunk_retry_max_delay"] = _get_float_env("CHUNK_RETRY_MAX_DELAY", 120.0)
    try:
        return AppConfig(**kwargs)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, "app config")) from exc
