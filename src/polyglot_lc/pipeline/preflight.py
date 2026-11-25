"""One short call per agent before a document is translated.

A long run pays the Planner, then a translation and a verdict for every chunk.
Discovering at chunk 73 that the model name does not exist — or that the key is
wrong, or the endpoint is dead — costs the plan, the glossary and every chunk
before it. Three tiny requests answer those questions in seconds, through the
same client path the run will use (auth, model routing, response parsing).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from polyglot_lc.llm.client import LLMClient

#: The one thing the canary asks: short enough to cost almost nothing, real
#: enough to exercise auth, model routing and the response parser.
PROMPT = "Reply with exactly one word: ok"

#: Agents worth asking. Three cover every endpoint configuration a run can use:
#: the Planner, the source group (Translator/Critic/Refiner share one) and the
#: Verifier.
AGENTS: tuple[str, ...] = ("PLANNER", "TRANSLATOR", "VERIFIER")

#: A canary that hangs is an answer too; it must not inherit the minutes-long
#: per-request budget meant for a whole chunk.
CANARY_TIMEOUT = 30.0


@dataclass
class PreflightCheck:
    """One agent's answer to the canary."""

    agent: str
    ok: bool
    detail: str = ""
    latency_ms: int = 0


@dataclass
class PreflightReport:
    """What the canary found, kept whole so the run record can quote it."""

    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def summary(self) -> str:
        failed = [check for check in self.checks if not check.ok]
        if not failed:
            return ""
        return "kiểm tra trước khi chạy thất bại: " + "; ".join(
            f"{check.agent}: {check.detail}" for check in failed
        )

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "agent": check.agent,
                "ok": check.ok,
                "detail": check.detail,
                "latency_ms": check.latency_ms,
            }
            for check in self.checks
        ]


def run_preflight(
    configs: dict[str, Any],
    agents: tuple[str, ...] = AGENTS,
) -> PreflightReport:
    """Ask each agent for one word through the real client path.

    The mock harness is skipped: its canned reply proves nothing about a
    provider, and the suite that counts calls must not pay three of them.
    """
    if os.getenv("LLM_MOCK", "0").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info("Preflight skipped: mock provider")
        return PreflightReport(
            checks=[PreflightCheck(agent=name, ok=True, detail="mock") for name in agents]
        )
    report = PreflightReport()
    for name in agents:
        config = configs.get(name)
        if config is None:
            continue
        configured = float(getattr(config, "timeout", 0) or CANARY_TIMEOUT)
        timeout = min(configured, CANARY_TIMEOUT)
        start = time.monotonic()
        try:
            client = LLMClient(config.model_copy(update={"timeout": timeout}))
            reply = client.chat([{"role": "user", "content": PROMPT}])
            text = (reply.text or "").strip()
            report.checks.append(
                PreflightCheck(
                    agent=name,
                    ok=bool(text),
                    detail="" if text else "nhà cung cấp trả về nội dung rỗng",
                    latency_ms=int((time.monotonic() - start) * 1000),
                )
            )
        except Exception as exc:  # noqa: BLE001 - any failure is the answer
            report.checks.append(
                PreflightCheck(
                    agent=name,
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}"[:200],
                    latency_ms=int((time.monotonic() - start) * 1000),
                )
            )
    return report
