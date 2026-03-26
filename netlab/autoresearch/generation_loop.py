"""Inner Loop 1: Scenario generation and validation.

Translates a connectivity idea into a validated ngraph scenario YAML
through an iterative LLM + inspect loop. Each iteration is cheap (~40ms)
because it only builds and inspects the graph — no simulation.

Two generation modes:
  - Parameterized: LLM specifies config parameters, scenario_generator
    produces YAML. The loop validates the generator output.
  - Freeform: LLM writes raw YAML for novel topology ideas. The loop
    validates via ngraph inspect and structural invariant checks.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .backend import LLMBackend


@dataclass
class InspectResult:
    """Structured output from ngraph scenario inspection."""

    success: bool
    node_count: int = 0
    link_count: int = 0
    risk_groups: list[str] = field(default_factory=list)
    demand_count: int = 0
    workflow_steps: int = 0
    hierarchy: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    raw_output: str = ""

    def summary(self) -> str:
        if not self.success:
            return f"FAILED: {'; '.join(self.errors)}"
        return (
            f"nodes={self.node_count}, links={self.link_count}, "
            f"risk_groups={self.risk_groups}, demands={self.demand_count}, "
            f"workflow_steps={self.workflow_steps}"
        )


@dataclass
class GenerationResult:
    """Output of the generation loop."""

    success: bool
    scenario_yaml: str = ""
    scenario_path: Path | None = None
    inspect: InspectResult | None = None
    iterations_used: int = 0
    error: str = ""


def inspect_scenario(scenario_path: Path, ngraph_bin: str) -> InspectResult:
    """Run ngraph inspect on a scenario file and parse the output.

    Uses the ngraph CLI for validation (catches DSL errors, expansion
    issues, and schema violations that the Python API might miss).
    """
    try:
        proc = subprocess.run(
            [ngraph_bin, "inspect", str(scenario_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return InspectResult(success=False, errors=["ngraph inspect timed out"])
    except FileNotFoundError:
        return InspectResult(
            success=False, errors=[f"ngraph binary not found: {ngraph_bin}"]
        )

    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        # Extract the most useful error lines
        error_lines = [
            line.strip()
            for line in output.splitlines()
            if "error" in line.lower() or "Error" in line or "invalid" in line.lower()
        ]
        if not error_lines:
            error_lines = output.strip().splitlines()[-5:]
        return InspectResult(
            success=False,
            errors=error_lines,
            raw_output=output,
        )

    # Parse structured data from the overview table.
    # Format: "   Metric             | Value"
    # Only match lines where the metric name starts the line (not hierarchy lines
    # like "- root | Nodes=10, Links=12" which also contain | and Nodes).
    result = InspectResult(success=True, raw_output=output)
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("-") or "=" in stripped.split("|")[-1]:
            continue  # Skip hierarchy lines and separator lines
        if stripped.startswith("Nodes") and "|" in stripped:
            result.node_count = _parse_int(stripped.split("|")[-1])
        elif stripped.startswith("Links") and "|" in stripped:
            result.link_count = _parse_int(stripped.split("|")[-1])
        elif stripped.startswith("Workflow steps") and "|" in stripped:
            result.workflow_steps = _parse_int(stripped.split("|")[-1])
        elif (
            stripped.startswith("Demand") and "|" in stripped and "demands" in stripped
        ):
            val_part = stripped.split("|")[-1]
            if "(" in val_part:
                inner = val_part.split("(")[1]
                result.demand_count = _parse_int(inner.split("demand")[0])
        elif stripped.startswith("Total Nodes:"):
            if result.node_count == 0:
                result.node_count = _parse_int(stripped.split(":")[-1])
        elif stripped.startswith("Total Links:"):
            if result.link_count == 0:
                result.link_count = _parse_int(stripped.split(":")[-1])
        elif line.startswith("Risk groups"):
            # "2 total; 0 disabled"
            result.risk_groups = []  # populated below from detailed section

    # Extract risk group names from the detailed section
    in_rg_section = False
    for line in output.splitlines():
        stripped = line.strip()
        if "RISK GROUPS" in stripped:
            in_rg_section = True
            continue
        if in_rg_section and stripped.startswith("Total:"):
            continue
        if in_rg_section and stripped and not stripped.startswith("-"):
            # Lines like "  path_a (enabled)" or "  rg_fiber (enabled)"
            name = stripped.split("(")[0].strip()
            if name and name != "Total:":
                result.risk_groups.append(name)
        if in_rg_section and stripped == "":
            # Empty line after risk groups section
            if result.risk_groups:
                in_rg_section = False

    return result


def _parse_int(s: str) -> int:
    try:
        return int(s.strip().replace(",", ""))
    except ValueError:
        return 0


_GENERATION_SYSTEM_PROMPT = """\
You are a network topology engineer generating ngraph scenario YAML files.

You will receive a connectivity idea and must produce a complete ngraph
scenario YAML. After each attempt, you will receive the ngraph inspect
output showing what was actually built. Compare it against the original
intent and fix any mismatches.

ngraph YAML format:
- nodes: named hierarchy (e.g., abc1/rsw, bb/abc1/pl1)
- links: source, target, capacity, cost, optional risk_groups and attrs
- risk_groups: named groups with attrs for failure domain modeling
- demands: regex patterns for source/target, volume, mode, flow_policy
- failures: named policies with modes and rules (scope: node/link/risk_group)
- workflow: ordered steps (MaximumSupportedDemand, TrafficMatrixPlacement, etc.)

All links are bidirectional by default (ngraph adds reverse automatically).
Use `risk_groups: [name]` on link definitions to assign to failure domains.
"""

_GENERATION_PROMPT_TEMPLATE = """\
Generate a complete ngraph scenario YAML for this connectivity idea:

{idea}

{feedback}

Return ONLY the YAML content, no explanation. Start with `seed:`.
"""

_REVISION_PROMPT_TEMPLATE = """\
The scenario you generated was inspected by ngraph. Here is the result:

{inspect_summary}

The original connectivity idea was:
{idea}

{validation_errors}

Please fix the scenario YAML to match the intent. Return ONLY the YAML content.
"""


def run_generation_loop(
    idea: str,
    backend: LLMBackend,
    ngraph_bin: str | None = None,
    max_iterations: int = 20,
    work_dir: Path | None = None,
) -> GenerationResult:
    """Run the scenario generation loop.

    Iterates: LLM generates YAML → ngraph inspect → compare → revise
    until the scenario passes validation or the budget is exhausted.

    Args:
        idea: Natural language description of the connectivity idea.
        backend: LLM backend for generation.
        ngraph_bin: Path to ngraph binary. Auto-detected if None.
        max_iterations: Maximum generation attempts.
        work_dir: Directory for temporary files. Uses tempdir if None.

    Returns:
        GenerationResult with the validated scenario or error details.
    """
    if ngraph_bin is None:
        ngraph_bin = shutil.which("ngraph")
        if ngraph_bin is None:
            return GenerationResult(
                success=False, error="ngraph binary not found on PATH"
            )

    cleanup_work_dir = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="genloop_"))
        cleanup_work_dir = True

    work_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = work_dir / "scenario.yml"

    last_inspect: InspectResult | None = None
    yaml_text = ""

    try:
        for iteration in range(max_iterations):
            # Generate or revise
            if iteration == 0:
                prompt = _GENERATION_PROMPT_TEMPLATE.format(idea=idea, feedback="")
            else:
                validation_errors = ""
                if last_inspect and last_inspect.errors:
                    validation_errors = "Errors:\n" + "\n".join(
                        f"- {e}" for e in last_inspect.errors
                    )
                prompt = _REVISION_PROMPT_TEMPLATE.format(
                    inspect_summary=last_inspect.summary() if last_inspect else "N/A",
                    idea=idea,
                    validation_errors=validation_errors,
                )

            response = backend.generate(prompt, system=_GENERATION_SYSTEM_PROMPT)

            # Extract YAML from response
            yaml_text = _extract_yaml(response)
            if not yaml_text:
                last_inspect = InspectResult(
                    success=False,
                    errors=["Could not extract valid YAML from LLM response"],
                )
                continue

            # Validate YAML syntax
            try:
                yaml.safe_load(yaml_text)
            except yaml.YAMLError as e:
                last_inspect = InspectResult(
                    success=False,
                    errors=[f"YAML syntax error: {e}"],
                )
                continue

            # Write and inspect
            scenario_path.write_text(yaml_text)
            last_inspect = inspect_scenario(scenario_path, ngraph_bin)

            if last_inspect.success:
                # Post-inspect viability checks
                viability_errors = _check_viability(last_inspect)
                if viability_errors:
                    last_inspect.success = False
                    last_inspect.errors = viability_errors
                    continue

                return GenerationResult(
                    success=True,
                    scenario_yaml=yaml_text,
                    scenario_path=scenario_path,
                    inspect=last_inspect,
                    iterations_used=iteration + 1,
                )

        # Budget exhausted
        return GenerationResult(
            success=False,
            scenario_yaml=yaml_text,
            inspect=last_inspect,
            iterations_used=max_iterations,
            error=f"Failed to generate valid scenario in {max_iterations} iterations",
        )

    finally:
        if cleanup_work_dir and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def _check_viability(inspect: InspectResult) -> list[str]:
    """Check that an inspected scenario is minimally viable.

    Returns a list of error strings. Empty list means viable.
    """
    errors: list[str] = []
    if inspect.node_count < 2:
        errors.append(f"Need at least 2 nodes, got {inspect.node_count}")
    if inspect.link_count < 1:
        errors.append(f"Need at least 1 link, got {inspect.link_count}")
    if inspect.demand_count < 1:
        errors.append(f"Need at least 1 demand, got {inspect.demand_count}")
    if inspect.workflow_steps < 1:
        errors.append(f"Need at least 1 workflow step, got {inspect.workflow_steps}")
    return errors


def _extract_yaml(response: str) -> str:
    """Extract YAML content from an LLM response.

    Handles fenced code blocks (```yaml ... ```) and raw YAML.
    """
    # Try fenced block first
    lines = response.splitlines()
    in_block = False
    block_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("```") and not in_block:
            in_block = True
            continue
        if line.strip() == "```" and in_block:
            in_block = False
            continue
        if in_block:
            block_lines.append(line)

    if block_lines:
        return "\n".join(block_lines).strip()

    # Fall back to raw response (skip any leading non-YAML text)
    for i, line in enumerate(lines):
        if line.strip().startswith("seed:") or line.strip().startswith("network:"):
            return "\n".join(lines[i:]).strip()

    return response.strip()
