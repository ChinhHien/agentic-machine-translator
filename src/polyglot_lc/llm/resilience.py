"""Resilience primitives for LLM calls: rate limits, timeouts, fallback.

Classification rules
--------------------
* :class:`RateLimitError`   — HTTP 429 / ``rate_limit_exceeded``: retry with
  the provider-supplied wait when parseable (``try again in 47.4s``), else
  exponential backoff (base * 2^n capped at max).
* :class:`TransientError`   — timeouts, connection errors, HTTP 408/409/425/
  429-free 5xx: retry with exponential backoff.
* :class:`PermanentError`   — 400/401/403/404/422, bad replies, anything a
  different endpoint cannot fix: fail fast, no retry, no fallback.

Every retry logs provider/model, HTTP status, attempt, wait seconds and the
reason. Once retries are exhausted the caller may fall back to a backup
endpoint (see :meth:`LLMClient.chat`).
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from typing import Any

from loguru import logger

from polyglot_lc.config import RetryConfig

# Enable verbose resilience logging via env var
VERBOSE_RESILIENCE = os.getenv("LLM_VERBOSE_RESILIENCE", "0").lower() in {"1", "true", "yes", "on"}

#: HTTP status codes worth retrying (transient server/network trouble).
RETRYABLE_STATUS = frozenset({408, 409, 425, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """Base for normalized provider failures (kept for backwards compat)."""


class RateLimitError(LLMError):
    """HTTP 429 / rate_limit_exceeded; ``retry_after`` is seconds when known."""

    def __init__(
        self, message: str, retry_after: float | None = None, status: int | None = None
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.status = status


class TransientError(LLMError):
    """Timeout / connection error / retryable HTTP status."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class PermanentError(LLMError):
    """Auth, bad request, malformed reply — retrying cannot fix these."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


_STATUS_RE = re.compile(r"\b(?:status[: ]|HTTP )?([1-5]\d{2})\b")
_TRY_AGAIN_RE = re.compile(
    r"try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|secs|second|seconds|m|min|mins|minute|minutes)?",
    re.IGNORECASE,
)
_RETRY_AFTER_RE = re.compile(r"retry[-_ ]?after\D{0,10}([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_RATE_LIMIT_HINTS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "quota",
    "429",
    "otpm",
    "tpm",
    "tokens per minute",
    "requests per minute",
)

#: Phrases that mean "you already have a request in flight", which is a
#: different problem from "you are sending too many". Exponential backoff does
#: not fix it: a 2 s retry collides with the request that is still running, and
#: the run burns its whole retry budget on collisions.
_CONCURRENCY_HINTS = (
    "concurrency",
    "concurrent request",
    "simultaneous",
    "parallel request",
)


def is_concurrency_limit(err: LLMError) -> bool:
    """True when *err* says another request from this process is still in flight."""
    text = str(err).lower()
    return any(hint in text for hint in _CONCURRENCY_HINTS)


def _extract_status(text: str) -> int | None:
    m = _STATUS_RE.search(text)
    return int(m.group(1)) if m else None


def parse_retry_seconds(text: str) -> float | None:
    """Best-effort parse of a provider wait hint, in seconds.

    Understands ``Please try again in 47.4s``, ``try again in 900ms`` and
    ``Retry-After: 12``; returns ``None`` when no hint is present.
    """
    m = _TRY_AGAIN_RE.search(text)
    if m:
        value = float(m.group(1))
        unit = (m.group(2) or "s").lower()
        if unit.startswith("ms"):
            return value / 1000.0
        if unit.startswith("m"):  # min/mins/minute/minutes
            return value * 60.0
        return value
    m = _RETRY_AFTER_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def _looks_like_rate_limit(text: str) -> bool:
    low = text.lower()
    return any(hint in low for hint in _RATE_LIMIT_HINTS)


def normalize_provider_error(exc: BaseException) -> LLMError:
    """Map any SDK/HTTP exception onto the three error classes above.

    The OpenAI SDK raises ``RateLimitError`` (distinct class) for 429 and
    ``APITimeoutError`` / ``APIConnectionError`` for network trouble; we use
    duck-typing on names so vendor SDK upgrades cannot break classification.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is None:
        # httpx-style exceptions carry the status on `.response`.
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    text = f"{type(exc).__name__}: {exc}"

    name = type(exc).__name__.lower()
    if "ratelimit" in name or "rate_limit" in name:
        return RateLimitError(
            text, retry_after=parse_retry_seconds(text), status=int(status) if status else 429
        )
    if "timeout" in name:
        return TransientError(text, status=int(status) if status else None)
    if "connection" in name or isinstance(exc, ConnectionError):
        return TransientError(text, status=int(status) if status else None)

    # Non-SDK exceptions (httpx, plain strings...): classify from the message.
    if status is not None:
        status = int(status)
        if status == 429:
            return RateLimitError(text, retry_after=parse_retry_seconds(text), status=status)
        if status in RETRYABLE_STATUS:
            return TransientError(text, status=status)
        return PermanentError(text, status=status)
    if _looks_like_rate_limit(text):
        return RateLimitError(
            text, retry_after=parse_retry_seconds(text), status=_extract_status(text)
        )
    if "timed out" in text.lower() or "timeout" in text.lower():
        return TransientError(text)
    if "connection" in text.lower() or "unreachable" in text.lower():
        return TransientError(text)
    return PermanentError(text)


def backoff_delay(retry_cfg: RetryConfig, attempt: int) -> float:
    """Exponential backoff for *attempt* (0-based): 2s, 4s, 8s ... capped."""
    return min(retry_cfg.max_delay, retry_cfg.base_delay * (2**attempt))


def jittered_delay(retry_cfg: RetryConfig, attempt: int) -> float:
    """Exponential backoff with bounded jitter (avoids thundering-herd retries).

    Uses *decorrelated* full-ish jitter: the wait is scaled by a random factor
    in ``[1 - jitter, 1]`` so it never exceeds the plain exponential value.
    ``jitter=0`` (the dataclass default) returns the exact backoff, which keeps
    unit tests deterministic; production uses ``LLM_RETRY_JITTER`` (default 0.2).
    """
    delay = backoff_delay(retry_cfg, attempt)
    if retry_cfg.jitter <= 0:
        return delay
    return max(0.0, delay * (1.0 - retry_cfg.jitter * random.random()))


def log_retry(
    err: LLMError,
    attempt: int,
    max_retries: int,
    wait: float,
    reason: str,
    *,
    provider: str,
    model: str,
) -> None:
    """One structured log line per retry: who, what status, how long, why."""
    logger.warning(
        "LLM retry | provider={} model={} status={} attempt={}/{} wait={:.1f}s reason={} ({})",
        provider,
        model,
        getattr(err, "status", None) or "?",
        attempt + 1,
        max_retries,
        wait,
        reason,
        type(err).__name__,
    )
    if VERBOSE_RESILIENCE:
        logger.debug(
            "LLM retry detail | provider={} model={} err_type={} err_msg={} "
            "retry_after={} buffer={} jitter={}",
            provider,
            model,
            type(err).__name__,
            str(err)[:200],
            getattr(err, "retry_after", None),
            RetryConfig().buffer_seconds,
            RetryConfig().jitter,
        )


class RetryStats:
    """Global counters for the final report (thread-safe: chunks run concurrently)."""

    _lock = threading.Lock()
    _rate_limit_retries = 0
    _transient_retries = 0
    _fallbacks = 0

    @classmethod
    def add_rate_limit_retry(cls) -> None:
        with cls._lock:
            cls._rate_limit_retries += 1

    @classmethod
    def add_transient_retry(cls) -> None:
        with cls._lock:
            cls._transient_retries += 1

    @classmethod
    def add_fallback(cls) -> None:
        with cls._lock:
            cls._fallbacks += 1

    @classmethod
    def snapshot(cls) -> dict[str, int]:
        with cls._lock:
            return {
                "rate_limit_retries": cls._rate_limit_retries,
                "transient_retries": cls._transient_retries,
                "fallbacks": cls._fallbacks,
            }

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._rate_limit_retries = 0
            cls._transient_retries = 0
            cls._fallbacks = 0


def _sleep(seconds: float) -> None:
    time.sleep(max(0.0, seconds))


def call_with_resilience(
    primary: Any,
    backup: Any,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    retry_cfg: RetryConfig | None = None,
    *,
    provider: str = "",
    model: str = "",
    backup_provider: str = "",
    backup_model: str = "",
    on_retry: Any = None,
) -> Any:
    """Run ``primary(*args, **kwargs)`` with the full resilience flow.

    *primary* and *backup* are zero-arg-ready callables (the backup, when
    given, receives the same ``(*args, **kwargs)``).

    Flow: rate limit -> wait (provider hint + buffer, else exponential
    backoff) and retry; transient errors -> backoff and retry; retries
    exhausted (or unrecoverable error) -> switch to *backup* when configured.
    Permanent errors never retry and never fall back.
    """
    cfg = retry_cfg or RetryConfig()
    kwargs = kwargs or {}

    def _call(fn: Any) -> Any:
        return fn(*args, **kwargs)

    def _log(err: LLMError, attempt: int, wait: float, reason: str) -> None:
        log_retry(err, attempt, cfg.max_retries, wait, reason, provider=provider, model=model)

    try:
        return _call(primary)
    except RateLimitError as err:
        if cfg.max_retries == 0:
            return _fallback_or_raise(
                backup, args, kwargs, err, provider, model, backup_provider, backup_model
            )
        last: LLMError = err
        for attempt in range(cfg.max_retries):
            hinted = err.retry_after
            wait = (
                min(cfg.max_delay, hinted + cfg.buffer_seconds)
                if hinted is not None
                else jittered_delay(cfg, attempt)
            )
            reason = (
                "rate limit: provider hint"
                if hinted is not None
                else "rate limit: exponential backoff"
            )
            if is_concurrency_limit(err):
                # Our own earlier request may still be running server side (an
                # abandoned deadline leaves it going), so a short backoff only
                # collides again. Wait for the slot to clear, then retry.
                floor = min(cfg.max_delay, max(wait, cfg.concurrency_wait))
                # Callers that collided together would otherwise retry in the
                # same instant and collide again, so jitter is *added* above the
                # floor — subtracting would break the minimum this exists for.
                wait = min(cfg.max_delay, floor * (1.0 + cfg.jitter * random.random()))
                reason = "rate limit: another request still holds the slot"
            _log(err, attempt, wait, reason)
            RetryStats.add_rate_limit_retry()
            if on_retry is not None:
                on_retry(str(err))
            _sleep(wait)
            try:
                return _call(primary)
            except LLMError as retry_err:
                # May stay a rate limit or change class mid-retries; either
                # way the remaining flow (retry/fallback) handles it below.
                last = retry_err
                if not isinstance(retry_err, RateLimitError):
                    break
                err = retry_err  # refresh the retry-after hint for next wait
        return _fallback_or_raise(
            backup, args, kwargs, last, provider, model, backup_provider, backup_model
        )
    except TransientError as err:
        if cfg.max_retries == 0:
            return _fallback_or_raise(
                backup, args, kwargs, err, provider, model, backup_provider, backup_model
            )
        last = err
        for attempt in range(cfg.max_retries):
            wait = jittered_delay(cfg, attempt)
            _log(err, attempt, wait, "transient error: exponential backoff")
            RetryStats.add_transient_retry()
            if on_retry is not None:
                on_retry(str(err))
            _sleep(wait)
            try:
                return _call(primary)
            except LLMError as retry_err:
                last = retry_err
                if not isinstance(retry_err, TransientError):
                    break
        return _fallback_or_raise(
            backup, args, kwargs, last, provider, model, backup_provider, backup_model
        )
    except PermanentError as err:
        # Application/business-level failure: a different endpoint cannot fix
        # it, so fail fast (caller surfaces it on the Errors panel).
        raise err from None


def _fallback_or_raise(
    backup: Any,
    args: tuple,
    kwargs: dict[str, Any],
    cause: LLMError,
    provider: str,
    model: str,
    backup_provider: str,
    backup_model: str,
) -> Any:
    """Switch to the backup endpoint, or re-raise when there is none."""
    if backup is None:
        raise cause from None
    logger.warning(
        "LLM fallback | primary={}/{} failed ({}) -> backup={}/{}",
        provider or "?",
        model or "?",
        type(cause).__name__,
        backup_provider or "?",
        backup_model or "?",
    )
    RetryStats.add_fallback()
    return backup(*args, **kwargs)
