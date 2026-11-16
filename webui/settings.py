"""Endpoint overrides saved from the web UI's settings dialog.

The pipeline reads its endpoints from the environment (``.env`` or the
process env). This module holds the one place the web UI may add to that:
a JSON file of per-agent endpoint overrides the reader entered in the
settings dialog. At launch the overrides that still answer are merged into
the child's env (a child env var wins over ``.env``); one that fails its
test is skipped, so the pipeline quietly falls back to the ``.env`` value.

Two shapes share the file:

``agents``  one entry per configurable agent. ``SOURCE_GROUP`` is the
            shared endpoint Translator/Critic/Refiner can be collapsed
            onto; ``BACKUP`` is the pipeline-wide fallback endpoint
            (``LLM_BACKUP_*``). An entry present is an override; its
            absence means "use whatever ``.env`` says".
``tests``   the last connection test each agent's entry went through, so
            reopening the dialog can show the green tick without re-testing.

Nothing here is authoritative about the pipeline itself: the file is only
a set of suggestions the run launcher re-validates at launch.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from webui.paths import REPO_ROOT

#: The agents the dialog can override, and the env prefix each one writes.
AGENT_ENV_PREFIXES: dict[str, str] = {
    "PLANNER": "PLANNER_",
    "TRANSLATOR": "TRANSLATOR_",
    "CRITIC": "CRITIC_",
    "REFINER": "REFINER_",
    "VERIFIER": "VERIFIER_",
    "SOURCE_GROUP": "SOURCE_GROUP_",
    "BACKUP": "LLM_BACKUP_",
}

#: The three agents an enabled source-group endpoint takes over from.
SOURCE_GROUP_AGENTS = ("TRANSLATOR", "CRITIC", "REFINER")

#: Where the file lives, under the repository root.
SETTINGS_RELATIVE = Path("data") / "webui-settings.json"

#: How long a dialog-triggered test may take (seconds).
TEST_TIMEOUT = 10.0

#: How long a launch-triggered re-test may take. A start already spends a
#: moment probing the document, but a launch should not hang on a dead host.
LAUNCH_TEST_TIMEOUT = 5.0

#: A key that is stored but never echoed back. The dialog shows the mask as a
#: placeholder; an empty field on save means "keep the stored key".
KEY_MASK = "••••••"


class SettingsError(ValueError):
    """A saved or submitted setting is malformed."""


def _format_validation_error(err: ValidationError, ctx: str) -> str:
    """Render a pydantic error briefly, without importing the pipeline package
    (this server stays free of ``polyglot_lc``; see ``webui.server``)."""
    parts = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e.get("loc", ()))
        parts.append(f"{loc or '<root>'}: {e.get('msg', 'invalid')}")
    return f"{ctx}: " + "; ".join(parts)


class EndpointOverride(BaseModel):
    """One agent's endpoint, exactly the three fields the dialog collects."""

    base_url: str
    model: str
    api_key: str = ""

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v

    @field_validator("model")
    @classmethod
    def _check_model(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("model must not be empty")
        return v


class Settings(BaseModel):
    """The whole file: which agents are overridden, and how chunks parallelize."""

    use_source_group: bool = False
    max_concurrency: int = Field(default=1, ge=1, le=16)
    #: Requests in flight per provider (``LLM_PROVIDER_CONCURRENCY``). The worker
    #: pool above is only useful when this lets more than one call through.
    provider_concurrency: int = Field(default=1, ge=1, le=32)
    agents: dict[str, EndpointOverride] = Field(default_factory=dict)

    @field_validator("agents")
    @classmethod
    def _check_agents(cls, v: dict[str, EndpointOverride]) -> dict[str, EndpointOverride]:
        unknown = set(v) - set(AGENT_ENV_PREFIXES)
        if unknown:
            raise ValueError(f"unknown agent(s): {', '.join(sorted(unknown))}")
        return v


def test_endpoint(
    base_url: str,
    model: str,
    api_key: str,
    *,
    timeout: float = TEST_TIMEOUT,
) -> dict[str, Any]:
    """Probe one OpenAI-compatible endpoint, cheaply first.

    ``GET /models`` costs nothing and answers for every OpenAI-compatible
    gateway; when it does not (a proxy that only forwards completions, a
    405/404 route), a one-token chat completion is tried instead — the
    smallest request that still proves the endpoint and the model both
    accept work. ``max_completion_tokens`` is retried once for providers
    that stopped accepting ``max_tokens``.

    The answer is a small plain dict, so it can travel as JSON unchanged:
    ``ok``, ``latency_ms``, ``method`` (``models`` | ``chat``) and, when the
    probe failed, a short ``error``. The key itself is never quoted back.
    """
    base = (base_url or "").strip().rstrip("/")
    model_name = (model or "").strip()
    started = time.perf_counter()
    if not base or not model_name:
        return {"ok": False, "latency_ms": 0, "method": "", "error": "thiếu base_url hoặc model"}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _elapsed() -> int:
        return int((time.perf_counter() - started) * 1000)

    # 1) The free probe.
    try:
        request = urllib.request.Request(f"{base}/models", headers=headers)
        with urllib.request.urlopen(request, timeout=timeout):
            return {"ok": True, "latency_ms": _elapsed(), "method": "models", "error": ""}
    except Exception:  # noqa: BLE001, S110 - any failure just falls through to the chat probe
        pass

    # 2) The one-token completion, then the same call in the newer token
    #    vocabulary. Each failure keeps the reason only until the next try
    #    succeeds; the last one is what the reader sees.
    error = "không kết nối được endpoint"
    for field in ("max_tokens", "max_completion_tokens"):
        body = json.dumps(
            {
                "model": model_name,
                "messages": [{"role": "user", "content": "ping"}],
                field: 1,
            }
        ).encode("utf-8")
        try:
            request = urllib.request.Request(
                f"{base}/chat/completions", data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=timeout):
                return {"ok": True, "latency_ms": _elapsed(), "method": "chat", "error": ""}
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = (exc.read() or b"").decode("utf-8", "replace")[:200]
            except Exception:  # noqa: BLE001, S110 - the body is a courtesy, not a contract
                pass
            error = f"HTTP {exc.code}: {detail}".strip()
        except Exception as exc:  # noqa: BLE001 - any transport failure is a probe failure
            error = str(exc) or exc.__class__.__name__
    return {"ok": False, "latency_ms": _elapsed(), "method": "", "error": error}


class SettingsStore:
    """The JSON file behind the dialog: load, save, delete, and mask for display."""

    def __init__(self, root: Path = REPO_ROOT) -> None:
        self.path = Path(root) / SETTINGS_RELATIVE

    # File ----------------------------------------------------------------
    def load(self) -> Settings:
        """The saved settings, or defaults when nothing valid is on disk."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return Settings()
        try:
            return Settings.model_validate(data)
        except ValidationError:
            return Settings()

    def save(self, settings: Settings) -> None:
        """Persist the settings (and keep any recorded test results)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {**settings.model_dump(), "tests": self._tests(settings)}, ensure_ascii=False
            ),
            encoding="utf-8",
        )

    def _tests(self, settings: Settings) -> dict[str, Any]:
        """Test results that still name a stored agent."""
        tests = self._read_tests()
        return {name: test for name, test in tests.items() if name in settings.agents}

    def _read_tests(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        tests = data.get("tests")
        return tests if isinstance(tests, dict) else {}

    # Mutation --------------------------------------------------------------
    def apply_update(self, data: dict[str, Any]) -> Settings:
        """Merge a dialog submission into the file and return what was saved.

        The submission carries *all* the dialog's cards, so an agent it leaves
        out is an override the reader emptied — it is dropped rather than kept
        on disk behind an empty field. An ``api_key`` left empty inside a kept
        entry means "unchanged": the dialog never echoes a key back, so an
        empty field is not "removed". Removing one entry outright is the
        delete route's job.
        """
        current = self.load()
        agents_in = data.get("agents")
        merged: dict[str, EndpointOverride] = {}
        if isinstance(agents_in, dict):
            for name, entry in agents_in.items():
                if name not in AGENT_ENV_PREFIXES:
                    raise SettingsError(f"unknown agent: {name}")
                if not isinstance(entry, dict):
                    raise SettingsError(f"agent {name}: expected an object")
                key = str(entry.get("api_key") or "").strip()
                if not key:
                    key = current.agents[name].api_key if name in current.agents else ""
                try:
                    merged[name] = EndpointOverride(
                        base_url=str(entry.get("base_url") or ""),
                        model=str(entry.get("model") or ""),
                        api_key=key,
                    )
                except ValidationError as exc:
                    raise SettingsError(_format_validation_error(exc, f"agent {name}")) from exc
        try:
            updated = Settings(
                use_source_group=bool(data.get("use_source_group", current.use_source_group)),
                max_concurrency=int(data.get("max_concurrency", current.max_concurrency)),
                provider_concurrency=int(
                    data.get("provider_concurrency", current.provider_concurrency)
                ),
                agents=merged,
            )
        except ValidationError as exc:
            raise SettingsError(_format_validation_error(exc, "settings")) from exc
        self.save(updated)
        return updated

    def delete_agent(self, name: str) -> bool:
        """Drop one agent's override (and its test result). ``False`` if absent."""
        if name not in AGENT_ENV_PREFIXES:
            raise SettingsError(f"unknown agent: {name}")
        current = self.load()
        if name not in current.agents:
            return False
        del current.agents[name]
        self.save(current)
        return True

    def record_test(self, name: str, result: dict[str, Any]) -> None:
        """Remember the last test an agent's endpoint went through."""
        if name not in AGENT_ENV_PREFIXES:
            return
        tests = self._read_tests()
        tests[name] = {**result, "checked_at": time.time()}
        current = self.load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {**current.model_dump(), "tests": tests},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # Reading for the page -------------------------------------------------
    def public(self) -> dict[str, Any]:
        """The settings as the dialog may see them: keys masked, tests included."""
        settings = self.load()
        return {
            "use_source_group": settings.use_source_group,
            "max_concurrency": settings.max_concurrency,
            "provider_concurrency": settings.provider_concurrency,
            "agents": {
                name: {
                    "base_url": endpoint.base_url,
                    "model": endpoint.model,
                    "has_key": bool(endpoint.api_key),
                }
                for name, endpoint in settings.agents.items()
            },
            "tests": self._read_tests(),
        }


def child_env(settings: Settings, ok: dict[str, bool]) -> tuple[dict[str, str], list[str]]:
    """Launch-time env for *settings*, given which agents passed their test.

    Returns ``(additions, removals)``. ``additions`` are the env vars to set
    over the child's environment; ``removals`` are names to strip from the
    server's own environment first (nothing today sets them, but a caller
    who exported ``TRANSLATOR_*`` in the shell would otherwise beat the
    group the reader just chose).

    A tested-ok source group also pins the group's values onto the three
    per-agent variables: the child resolves ``<NAME>_*`` before
    ``SOURCE_GROUP_*``, so only the per-agent names can carry the override
    past whatever ``.env`` says for Translator, Critic or Refiner alone.
    """
    additions: dict[str, str] = {}
    for name, endpoint in settings.agents.items():
        if not ok.get(name):
            continue
        prefix = AGENT_ENV_PREFIXES[name]
        additions[f"{prefix}BASE_URL"] = endpoint.base_url
        additions[f"{prefix}MODEL"] = endpoint.model
        if endpoint.api_key:
            additions[f"{prefix}API_KEY"] = endpoint.api_key
    if settings.max_concurrency:
        additions["MAX_CONCURRENCY"] = str(settings.max_concurrency)
    if settings.provider_concurrency:
        additions["LLM_PROVIDER_CONCURRENCY"] = str(settings.provider_concurrency)

    removals: list[str] = []
    group_ok = settings.use_source_group and bool(ok.get("SOURCE_GROUP"))
    if group_ok:
        for agent in SOURCE_GROUP_AGENTS:
            for suffix in ("BASE_URL", "MODEL", "API_KEY"):
                removals.append(f"{AGENT_ENV_PREFIXES[agent]}{suffix}")
        group = settings.agents.get("SOURCE_GROUP")
        if group is not None:
            for agent in SOURCE_GROUP_AGENTS:
                prefix = AGENT_ENV_PREFIXES[agent]
                additions[f"{prefix}BASE_URL"] = group.base_url
                additions[f"{prefix}MODEL"] = group.model
                if group.api_key:
                    additions[f"{prefix}API_KEY"] = group.api_key
    return additions, removals
