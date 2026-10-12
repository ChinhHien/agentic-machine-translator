"""Planner agent: the named LLM role that owns document-level planning.

Prompt construction deliberately lives with the workflow that owns it, so a
planner prompt and its parser stay in one file:

* the single document plan — :mod:`polyglot_lc.analysis.unified_planner`
* the legacy section/document summaries — :mod:`polyglot_lc.segment.summary_tree`
* the SkillOpt textual gradient — :mod:`polyglot_lc.skillopt.optimizer`

This class only names the role, so PLANNER_* provider/model settings resolve
independently of the read-source group and the verifier.
"""

from __future__ import annotations

from polyglot_lc.agents.base import Agent


class PlannerAgent(Agent):
    """Document-level planning role (its own ``PLANNER_*`` provider config)."""

    name = "PLANNER"
