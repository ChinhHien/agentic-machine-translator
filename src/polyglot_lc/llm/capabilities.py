"""Fixed per-call limits for an LLM endpoint.

There is no learning and no probing: a run uses the limits configured for it
(``<NAME>_TIMEOUT``) and the process-wide provider concurrency. What a model
can do is something the reader configures, not something the pipeline guesses
from earlier replies.
"""

from __future__ import annotations

from pydantic import BaseModel

from polyglot_lc.config import DEFAULT_TIMEOUT, get_retry_config


class EffectiveCaps(BaseModel):
    """The limits one call is made under."""

    concurrency: int | None = None
    timeout: float | None = None
    #: Name this endpoint accepts for the output-token cap.
    token_param: str = "max_tokens"


def resolve(
    base_url: str,
    model: str,
    *,
    agent_timeout: float | None = None,
    provider_concurrency: int | None = None,
) -> EffectiveCaps:
    """Return the configured limits for one call.

    *base_url* and *model* identify the endpoint in log messages only; neither
    changes the answer. *agent_timeout* is the agent's own ``<NAME>_TIMEOUT``
    and *provider_concurrency* is ``LLM_PROVIDER_CONCURRENCY``.
    """
    timeout = DEFAULT_TIMEOUT if agent_timeout is None else agent_timeout
    concurrency = (
        get_retry_config().provider_concurrency
        if provider_concurrency is None
        else provider_concurrency
    )
    return EffectiveCaps(concurrency=concurrency, timeout=timeout)
