"""Agent base class: one named LLM role with its own provider config."""

from __future__ import annotations

from typing import Any

from polyglot_lc.config import get_agent_config
from polyglot_lc.llm.client import LLMClient, LLMResponse, parse_json_reply


class Agent:
    """Base for all agents. Subclasses set prompts and parse replies."""

    name: str = "AGENT"

    def __init__(self, config=None) -> None:
        self.config = config or get_agent_config(self.name)
        self.client = LLMClient(self.config)
        self.calls: int = 0
        self.total_tokens: int = 0
        #: The model that answered the last call. A fallback endpoint can serve a
        #: different one than ``config.model``, and the status tab reports what
        #: actually did the work.
        self.last_model: str = self.config.model

    def chat(self, messages: list[dict[str, str]], **kw: Any) -> LLMResponse:
        """One chat call, tracked for metrics."""
        self.calls += 1
        resp = self.client.chat(messages, **kw)
        self.total_tokens += resp.prompt_tokens + resp.completion_tokens
        self.last_model = resp.model or self.config.model
        return resp

    def chat_json(
        self, messages: list[dict[str, Any]], cache_prefix: bool = False, **kw: Any
    ) -> dict[str, Any]:
        """Chat call expecting a JSON object reply.

        ``cache_prefix`` marks the stable leading message as a cache
        breakpoint when the provider requires explicit markers (see
        :class:`~polyglot_lc.config.CacheConfig`). Extra keyword arguments
        (e.g. ``max_tokens``) travel to the provider call.
        """
        resp = self.chat(messages, json_mode=True, cache_prefix=cache_prefix, **kw)
        return parse_json_reply(resp.text)
