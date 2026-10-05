"""LLM client over any OpenAI-compatible REST API.

Uses the official ``openai`` SDK with a custom ``base_url`` so OpenAI,
DeepSeek, an Anthropic OpenAI-compatible proxy, Ollama and vLLM all work with
the same code path. Supports mock mode (``LLM_MOCK=1``) for demos/tests.

Resilience: transient failures (rate limits, timeouts, connection errors) are
retried with provider-hinted or exponential backoff (see
``polyglot_lc.llm.resilience``); when retries are exhausted, chat calls fail
over to the per-agent backup endpoint configured via ``<NAME>_BACKUP_*`` vars.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from polyglot_lc.config import (
    AgentConfig,
    EmbedConfig,
    get_cache_config,
    get_retry_config,
)
from polyglot_lc.llm.resilience import (
    LLMError,
    call_with_resilience,
    normalize_provider_error,
)


class TokenUsage:
    """Thread-safe global counter for LLM tokens sent / received / cached.

    Every successful :meth:`LLMClient.chat` call adds to the totals regardless
    of which agent performed the call. Totals are also broken down per agent
    (PLANNER / TRANSLATOR / CRITIC / ...) via :meth:`snapshot_by_agent` and per
    provider/model via :meth:`snapshot_by_model`.

    ``cached`` counts input tokens the provider reported as served from its
    prompt/context cache (a subset of ``sent``); ``sent - cached`` is the
    uncached input. When a provider omits usage data the tokens are estimated
    (chars/4) and flagged via :class:`LLMResponse.usage_estimated`.
    """

    _lock = threading.Lock()
    _sent: int = 0
    _received: int = 0
    _cached: int = 0
    _calls: int = 0
    _estimated_calls: int = 0
    _cache_hits: int = 0
    _cache_misses: int = 0
    _cache_unknown: int = 0
    _by_agent: dict[str, dict[str, int]] = {}
    _by_model: dict[str, dict[str, int]] = {}

    @classmethod
    def add(
        cls,
        prompt_tokens: int,
        completion_tokens: int,
        agent: str = "",
        cached_tokens: int = 0,
        model: str = "",
        provider: str = "",
        estimated: bool = False,
        cache_status: str = "UNKNOWN",
    ) -> None:
        prompt = max(0, prompt_tokens)
        completion = max(0, completion_tokens)
        cached = max(0, min(cached_tokens, prompt))
        status = cache_status if cache_status in {"HIT", "MISS"} else "UNKNOWN"
        with cls._lock:
            cls._sent += prompt
            cls._received += completion
            cls._cached += cached
            cls._calls += 1
            if estimated:
                cls._estimated_calls += 1
            if status == "HIT":
                cls._cache_hits += 1
            elif status == "MISS":
                cls._cache_misses += 1
            else:
                cls._cache_unknown += 1
            for table, key in (
                (cls._by_agent, agent),
                (cls._by_model, f"{provider or '?'} / {model or '?'}"),
            ):
                if not key:
                    continue
                bucket = table.setdefault(
                    key, {"sent": 0, "received": 0, "cached": 0, "calls": 0}
                )
                bucket["sent"] += prompt
                bucket["received"] += completion
                bucket["cached"] += cached
                bucket["calls"] += 1

    @classmethod
    def snapshot(cls) -> dict[str, int]:
        with cls._lock:
            return {
                "sent": cls._sent,
                "received": cls._received,
                "cached": cls._cached,
                "calls": cls._calls,
                "estimated_calls": cls._estimated_calls,
                "cache_hits": cls._cache_hits,
                "cache_misses": cls._cache_misses,
                "cache_unknown": cls._cache_unknown,
            }

    @classmethod
    def snapshot_by_agent(cls) -> dict[str, dict[str, int]]:
        """Per-agent breakdown: ``{name: {sent, received, cached, calls}}``."""
        with cls._lock:
            return {name: dict(bucket) for name, bucket in cls._by_agent.items()}

    @classmethod
    def snapshot_by_model(cls) -> dict[str, dict[str, int]]:
        """Per provider/model breakdown, keyed ``"<base_url> / <model>"``."""
        with cls._lock:
            return {key: dict(bucket) for key, bucket in cls._by_model.items()}

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._sent = 0
            cls._received = 0
            cls._cached = 0
            cls._calls = 0
            cls._estimated_calls = 0
            cls._cache_hits = 0
            cls._cache_misses = 0
            cls._cache_unknown = 0
            cls._by_agent.clear()
            cls._by_model.clear()


def format_tokens(n: int) -> str:
    """Human-readable token count (e.g. ``12.4k``, ``1.25M``)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


@dataclass
class LLMResponse:
    """A single completion result plus usage metadata."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: int = 0
    via_backup: bool = False
    provider: str = ""
    #: True when the endpoint returned no usage and tokens were estimated.
    usage_estimated: bool = False
    #: "HIT" / "MISS" only when the provider reported cache metadata; "UNKNOWN"
    #: otherwise (absence of metadata is NOT treated as a miss).
    cache_status: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_status": self.cache_status,
            "latency_ms": self.latency_ms,
            "via_backup": self.via_backup,
            "provider": self.provider,
            "usage_estimated": self.usage_estimated,
        }


@dataclass
class EmbeddingResponse:
    """Embedding vectors for a list of inputs."""

    vectors: list[list[float]]
    model: str


_MOCK_CHAT_REPLIES: dict[str, str] = {}


def register_mock_reply(marker: str, reply: str) -> None:
    """Register a deterministic mock reply returned when prompt contains marker."""
    _MOCK_CHAT_REPLIES[marker] = reply


def clear_mock_replies() -> None:
    _MOCK_CHAT_REPLIES.clear()


class LLMClient:
    """Chat + embeddings client for one OpenAI-compatible endpoint.

    When the agent config defines a backup endpoint (``<NAME>_BACKUP_BASE_URL``
    + ``<NAME>_BACKUP_MODEL``), chat calls that exhaust their retries on the
    primary are executed once against the backup instead of failing.
    """

    def __init__(self, config: AgentConfig | EmbedConfig) -> None:
        self.config = config
        self._client: OpenAI | None = None
        self._backup: LLMClient | None = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sdk(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key or "sk-mock",
                timeout=getattr(self.config, "timeout", 120.0),
                max_retries=0,  # resilience layer handles retries
            )
        return self._client

    def _backup_client(self) -> LLMClient | None:
        """Lazy backup client (primary config only; backups never nest)."""
        if self._backup is None:
            cfg = self.config
            if getattr(cfg, "has_backup", False):
                self._backup = LLMClient(
                    AgentConfig(
                        name=cfg.name,
                        base_url=cfg.backup_base_url,
                        model=cfg.backup_model,
                        api_key=cfg.backup_api_key,
                        temperature=cfg.temperature,
                        timeout=cfg.timeout,
                    )
                )
        return self._backup

    def _mock_chat(self, messages: list[dict[str, Any]], json_mode: bool = False) -> str:
        prompt = _messages_text(messages)
        for marker, reply in _MOCK_CHAT_REPLIES.items():
            if marker in prompt:
                return reply
        if json_mode or "Reply ONLY with JSON" in prompt:
            return _mock_json_reply(prompt)
        role = prompt[:1].upper() or "X"
        # Deterministic pseudo-translation: prefix each line.
        lines = [ln for ln in prompt.splitlines() if ln.strip()]
        payload = lines[-1] if lines else prompt
        return f"[{role}->{payload[:80]}]"

    def _mock_embed(self, inputs: list[str]) -> list[list[float]]:
        return [mock_hash_embedding(t) for t in inputs]

    def _raw_chat(self, messages: list[dict[str, Any]], **kw: Any) -> LLMResponse:
        """Single un-retried chat call; raises normalized resilience errors."""
        start = time.monotonic()
        try:
            resp = self._sdk().chat.completions.create(
                model=self.config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=getattr(self.config, "temperature", 0.3),
                **kw,
            )
        except Exception as exc:  # noqa: BLE001 - normalized by resilience layer
            raise normalize_provider_error(exc) from exc
        latency = int((time.monotonic() - start) * 1000)
        choice = resp.choices[0] if resp.choices else None
        text = (choice.message.content if choice and choice.message else "") or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens, completion_tokens, cached_tokens, cache_reported = _parse_usage(usage)
        estimated = False
        # Estimated usage when the endpoint omits it (some Ollama/vLLM builds).
        if prompt_tokens == 0 and completion_tokens == 0:
            prompt_tokens = _estimate_tokens(_messages_text(messages))
            completion_tokens = _estimate_tokens(text)
            estimated = True
        return LLMResponse(
            text=text,
            model=self.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            latency_ms=latency,
            provider=self.config.base_url,
            usage_estimated=estimated,
            cache_status=_cache_status(cache_reported, cached_tokens),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        cache_prefix: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion; ``json_mode`` requests a JSON object reply.

        ``cache_prefix`` asks the client to mark the leading, stable message as
        a prompt/context-cache breakpoint when ``PROMPT_CACHE=explicit``. With
        the default ``auto`` mode no marker is sent (providers cache the stable
        prefix automatically) but reported cache metrics are still recorded.
        """
        if json_mode:
            kwargs.setdefault("response_format", {"type": "json_object"})
        agent = getattr(self.config, "name", "")
        model = self.config.model
        provider = self.config.base_url
        if cache_prefix and get_cache_config().send_markers:
            messages = _apply_cache_prefix(messages)
        if isinstance(self.config, AgentConfig) and self.config.is_mock:
            prompt = _messages_text(messages)
            text = self._mock_chat(messages, json_mode=json_mode)
            prompt_tokens = _estimate_tokens(prompt)
            completion_tokens = _estimate_tokens(text)
            TokenUsage.add(
                prompt_tokens, completion_tokens, agent=agent, model=model,
                provider=provider, estimated=True,
            )
            return LLMResponse(
                text=text,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=1,
                provider=provider,
                usage_estimated=True,
            )
        backup = self._backup_client()
        used_backup = False

        def backup_call(msgs: list[dict[str, Any]], **kw: Any) -> LLMResponse:
            nonlocal used_backup
            used_backup = True
            assert backup is not None
            return backup._raw_chat(msgs, **kw)

        resp = call_with_resilience(
            self._raw_chat,
            backup_call if backup is not None else None,
            args=(messages,),
            kwargs=kwargs,
            retry_cfg=get_retry_config(),
            provider=self.config.base_url,
            model=self.config.model,
            backup_provider=(backup.config.base_url if backup else ""),
            backup_model=(backup.config.model if backup else ""),
        )
        resp.via_backup = used_backup
        TokenUsage.add(
            resp.prompt_tokens, resp.completion_tokens, agent=agent,
            cached_tokens=resp.cached_tokens, model=model, provider=provider,
            estimated=resp.usage_estimated, cache_status=resp.cache_status,
        )
        return resp

    def embed(self, inputs: list[str]) -> EmbeddingResponse:
        """Embed a batch of texts (no backup: the reasoning bank is best-effort)."""
        if isinstance(self.config, EmbedConfig) and self.config.is_mock:
            return EmbeddingResponse(
                vectors=self._mock_embed(inputs), model=self.config.model
            )
        start = time.monotonic()
        try:
            resp = self._sdk().embeddings.create(
                model=self.config.model,
                input=inputs,
            )
        except Exception as exc:  # noqa: BLE001
            raise normalize_provider_error(exc) from exc
        del start
        vectors = [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]
        return EmbeddingResponse(vectors=vectors, model=self.config.model)


_EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _message_text(message: dict[str, Any]) -> str:
    """Flatten one message's content (string or content-block list) to text."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content or "")


def _messages_text(messages: list[dict[str, Any]]) -> str:
    """Concatenate all messages' text (used for mock replies + estimates)."""
    return "\n".join(_message_text(m) for m in messages)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~chars/4) used when usage data is unavailable."""
    return max(1, (len(text) + 3) // 4)


def _apply_cache_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the first (stable) message as an explicit cache breakpoint.

    Converts the leading message's string content into an Anthropic-style
    text block carrying ``cache_control: {"type": "ephemeral"}``. Only called
    when ``PROMPT_CACHE=explicit``; providers with automatic prefix caching
    ignore it and cache the stable prefix on their own.
    """
    if not messages:
        return messages
    first = dict(messages[0])
    content = first.get("content", "")
    if isinstance(content, str) and content:
        first["content"] = [
            {"type": "text", "text": content,
             "cache_control": {"type": "ephemeral"}}
        ]
        return [first, *messages[1:]]
    return messages


def _as_int(value: Any) -> int | None:
    """Best-effort int conversion; returns None for missing/non-numeric."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_field(obj: Any, name: str) -> Any:
    """Read *name* from an object or dict (SDK objects vs plain dicts)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _parse_usage(usage: Any) -> tuple[int, int, int, bool]:
    """Extract (prompt, completion, cached, cache_reported) across formats.

    Handles OpenAI (``prompt_tokens`` + ``prompt_tokens_details.cached_tokens``),
    DeepSeek (``prompt_cache_hit_tokens``) and Anthropic-compatible responses
    (``input_tokens`` + ``cache_read_input_tokens`` + ``cache_creation_input_tokens``).
    Never fabricates values: missing fields stay 0. ``cache_reported`` is True
    only when the provider actually returned cache metadata, so a zero cached
    count is a real MISS rather than an UNKNOWN.
    """
    prompt = _as_int(_get_field(usage, "prompt_tokens"))
    completion = _as_int(_get_field(usage, "completion_tokens"))
    cached = 0
    cache_reported = False

    details = _get_field(usage, "prompt_tokens_details")
    cached_detail = _as_int(_get_field(details, "cached_tokens"))
    if cached_detail is not None:
        cached = cached_detail
        cache_reported = True
    hit = _as_int(_get_field(usage, "prompt_cache_hit_tokens"))
    if hit is not None:
        cache_reported = True
        cached = max(cached, hit)
    read = _as_int(_get_field(usage, "cache_read_input_tokens"))
    if read is not None:
        cache_reported = True
        cached = max(cached, read)

    if prompt is None:
        # Anthropic-style: input_tokens excludes cache reads/creations.
        inp = _as_int(_get_field(usage, "input_tokens"))
        if inp is not None:
            creation = _as_int(_get_field(usage, "cache_creation_input_tokens")) or 0
            prompt = inp + (read or 0) + creation
    if prompt is None:
        prompt = 0
    if completion is None:
        completion = 0
    return prompt, completion, min(cached, prompt), cache_reported


def _cache_status(reported: bool, cached: int) -> str:
    """HIT when the provider reported cache reads, MISS when it reported none,
    UNKNOWN when it reported no cache metadata at all."""
    if not reported:
        return "UNKNOWN"
    return "HIT" if cached > 0 else "MISS"


def mock_hash_embedding(text: str) -> list[float]:
    """Deterministic feature-hashed embedding for mock/offline mode.

    Words + character trigrams are sign-hashed into a fixed-dim bag vector,
    so texts sharing vocabulary get higher cosine similarity and retrieval
    stays meaningful (and reproducible) without a network call.
    """
    vec = [0.0] * _EMBED_DIM
    feats: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        feats.append(tok)
        if len(tok) > 3:
            feats.extend(tok[i : i + 3] for i in range(len(tok) - 2))
    for feat in feats:
        h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % _EMBED_DIM] += 1.0 if (h >> 16) & 1 else -1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _mock_json_reply(prompt: str) -> str:
    """Deterministic JSON replies per agent role for mock mode."""
    if "translation planner" in prompt:
        return (
            '{"strategy": "mock: literal with consistent terminology", '
            '"few_shot": ["NMT -> NMT"], "pitfalls": ["mock pitfall"], '
            '"notes": "mock notes"}'
        )
    if "textual gradient" in prompt:
        return (
            '{"instruction": "mock: improved instruction", '
            '"few_shot": ["mock -> mock"], "rationale": "mock rationale"}'
        )
    if "QA verifier" in prompt:
        return '{"pass": true, "reasons": []}'
    if "translation critic" in prompt:
        return '{"score": 8.5, "issues": [], "suggestions": []}'
    return '{"note": "mock reply"}'


def parse_json_reply(text: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM reply, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"Reply is not JSON: {text[:120]!r}")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"Invalid JSON in reply: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("Reply JSON is not an object")
    return data
