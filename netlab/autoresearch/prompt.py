"""Prompt construction for autoresearch LLM interactions.

Provides:
- build_hypothesis_prompt(): assembles system + user prompt for hypothesis generation
- build_reflection_prompt(): assembles prompt for reflection on recent experiments
- parse_hypothesis_response(): extracts YAML params from LLM response
- render_memory_section(): formats research memory for prompt inclusion
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

import yaml

if TYPE_CHECKING:
    from netlab.autoresearch.experiment_log import LogEntry
    from netlab.autoresearch.hypothesis import HypothesisTemplate


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into valid parameters."""


# ---------------------------------------------------------------------------
# Memory protocol -- duck-typed so we don't depend on the not-yet-built F-8.
# ---------------------------------------------------------------------------


@runtime_checkable
class InsightLike(Protocol):
    @property
    def claim(self) -> str: ...

    @property
    def confidence(self) -> str: ...

    @property
    def evidence_for(self) -> list: ...

    @property
    def evidence_against(self) -> list: ...


@runtime_checkable
class DeadEndLike(Protocol):
    @property
    def params_summary(self) -> str: ...

    @property
    def failure_type(self) -> str: ...

    @property
    def reason(self) -> str: ...

    @property
    def lesson(self) -> str: ...


@runtime_checkable
class ResearchMemoryLike(Protocol):
    @property
    def active_insights(self) -> list: ...

    @property
    def dead_ends(self) -> list: ...

    @property
    def strategy(self) -> str: ...


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_YAML_FENCE_RE = re.compile(r"```(?:yaml|YAML)\s*\n(.*?)```", re.DOTALL)


def _render_param_space(template: HypothesisTemplate) -> str:
    """Render parameter definitions for inclusion in the prompt."""
    lines: list[str] = []
    for name, pdef in template.params.items():
        parts = [f"- {name} ({pdef.type})"]
        if pdef.range is not None:
            parts.append(f"range [{pdef.range[0]}, {pdef.range[1]}]")
        if pdef.step is not None:
            parts.append(f"step {pdef.step}")
        if pdef.values is not None:
            parts.append(f"values {pdef.values}")
        if pdef.default is not None:
            parts.append(f"default {pdef.default}")
        if pdef.description:
            parts.append(f"-- {pdef.description}")
        lines.append(", ".join(parts))
    return "\n".join(lines)


def _render_best(best: Optional[LogEntry]) -> str:
    """Render the current best experiment for inclusion in the prompt."""
    if best is None:
        return "No successful experiments yet."
    parts = [f"Best so far: {best.exp_id}"]
    if best.objective_score is not None:
        parts.append(f"score={best.objective_score:.4f}")
    if best.params:
        params_str = ", ".join(f"{k}={v}" for k, v in sorted(best.params.items()))
        parts.append(f"params={{{params_str}}}")
    return ", ".join(parts)


def render_memory_section(memory: ResearchMemoryLike) -> str:
    """Render insights + dead_ends + strategy for inclusion in hypothesis prompt.

    Returns empty string if memory is empty (no section headers injected).
    """
    insights = memory.active_insights
    dead_ends = memory.dead_ends
    strategy = memory.strategy

    if not insights and not dead_ends and not strategy:
        return ""

    sections: list[str] = ["## Your Research Notes"]
    sections.append(
        "(These are your own observations. You may revise freely "
        "based on new evidence.)"
    )
    sections.append("")

    if insights:
        sections.append("### Verified Insights")
        for ins in insights:
            conf = ins.confidence
            ef = len(ins.evidence_for)
            ea = len(ins.evidence_against)
            sections.append(
                f"- [{conf}] {ins.claim} (evidence: {ef} for, {ea} against)"
            )
        sections.append("")

    if dead_ends:
        sections.append("### Known Dead Ends")
        for de in dead_ends:
            sections.append(
                f"- {de.params_summary}: {de.reason} "
                f"(type: {de.failure_type}, lesson: {de.lesson})"
            )
        sections.append("")

    if strategy:
        sections.append("### Your Current Strategy")
        sections.append(strategy)
        sections.append("")

    return "\n".join(sections)


def build_hypothesis_prompt(
    program_md: str,
    template: HypothesisTemplate,
    history: str,
    memory_section: str,
    best: Optional[LogEntry],
) -> tuple[str, str]:
    """Assemble system and user prompts for hypothesis generation.

    Returns (system_prompt, user_prompt).
    """
    system_prompt = program_md

    # Build user prompt sections in order:
    # 1. Parameter space
    # 2. Memory section (if non-empty)
    # 3. History
    # 4. Current best
    # 5. Instructions
    user_parts: list[str] = []

    user_parts.append("## Parameter Space")
    user_parts.append(_render_param_space(template))
    user_parts.append("")

    if memory_section:
        user_parts.append(memory_section)
        user_parts.append("")

    user_parts.append("## Experiment History")
    user_parts.append(history)
    user_parts.append("")

    user_parts.append("## Current Best")
    user_parts.append(_render_best(best))
    user_parts.append("")

    user_parts.append("## Instructions")
    user_parts.append(
        "Based on the above, propose a new set of parameter values to try. "
        "Explain your reasoning, then provide the parameters in a YAML block like:"
    )
    user_parts.append("```yaml")
    user_parts.append("params:")
    user_parts.append("  param_name: value")
    user_parts.append("```")

    user_prompt = "\n".join(user_parts)
    return system_prompt, user_prompt


def build_reflection_prompt(
    recent_entries: list[LogEntry],
    memory: ResearchMemoryLike,
    best: Optional[LogEntry],
) -> tuple[str, str]:
    """Assemble system and user prompts for reflection on recent experiments.

    Returns (system_prompt, user_prompt).
    """
    system_prompt = (
        "You are a research assistant reflecting on recent experimental results. "
        "Your task is to update your research notes: identify insights, "
        "flag dead ends, and revise your strategy."
    )

    user_parts: list[str] = []

    # Recent experiments
    user_parts.append("## Recent Experiment Results")
    for entry in recent_entries:
        parts = [f"- {entry.exp_id}: status={entry.status}"]
        if entry.objective_score is not None:
            parts.append(f"score={entry.objective_score:.4f}")
        if entry.params:
            params_str = ", ".join(f"{k}={v}" for k, v in sorted(entry.params.items()))
            parts.append(f"params={{{params_str}}}")
        if entry.error_detail:
            parts.append(f"error={entry.error_detail!r}")
        user_parts.append(", ".join(parts))
    user_parts.append("")

    # Current memory
    mem_section = render_memory_section(memory)
    if mem_section:
        user_parts.append("## Current Memory")
        user_parts.append(mem_section)
        user_parts.append("")

    # Current best
    user_parts.append("## Current Best")
    user_parts.append(_render_best(best))
    user_parts.append("")

    # Task sections
    user_parts.append("## Tasks")
    user_parts.append(
        "Review the results above and update your research notes. "
        "Respond with a JSON object containing three sections:"
    )
    user_parts.append("")
    user_parts.append("### INSIGHTS")
    user_parts.append(
        "List any new insights or updates to existing insights. "
        "Each insight needs a claim, evidence_for (list of exp_ids), "
        "and optionally evidence_against."
    )
    user_parts.append("")
    user_parts.append("### DEAD ENDS")
    user_parts.append(
        "List any parameter combinations that should be avoided, "
        "with the reason and lesson learned."
    )
    user_parts.append("")
    user_parts.append("### STRATEGY")
    user_parts.append(
        "Write a brief strategy for the next experiments (what to try, "
        "what to avoid, what hypotheses to test)."
    )

    user_prompt = "\n".join(user_parts)
    return system_prompt, user_prompt


def parse_hypothesis_response(response: str) -> dict[str, Any]:
    """Extract YAML params from LLM response.

    Tries:
    1. Find a ```yaml ... ``` fenced block and parse its content.
    2. If no fenced block, try to parse the entire response as YAML.
    3. If both fail, raise ParseError.

    If parsed result has a "params" key, returns the value under "params".
    Otherwise returns the whole dict.
    """
    parsed: Any = None

    # Try fenced YAML block first
    match = _YAML_FENCE_RE.search(response)
    if match:
        yaml_text = match.group(1)
        try:
            parsed = yaml.safe_load(yaml_text)
        except yaml.YAMLError as exc:
            raise ParseError(f"Failed to parse YAML in fenced block: {exc}") from exc
    else:
        # Try parsing the entire response as YAML
        try:
            parsed = yaml.safe_load(response)
        except yaml.YAMLError as err:
            raise ParseError(
                "No YAML fenced block found and response is not valid YAML"
            ) from err

    if not isinstance(parsed, dict):
        raise ParseError(f"Parsed YAML is not a dict (got {type(parsed).__name__})")

    if "params" in parsed:
        result = parsed["params"]
        if not isinstance(result, dict):
            raise ParseError(
                f"'params' key is not a dict (got {type(result).__name__})"
            )
        return result

    return parsed
