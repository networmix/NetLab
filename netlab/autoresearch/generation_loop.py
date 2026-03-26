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

from netlab.runtime import require_executable

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
    """Output of the generation loop.

    On success, contains both the validated scenario and the simulation
    results (since the simulation is run as part of validation).
    """

    success: bool
    scenario_yaml: str = ""
    scenario_path: Path | None = None
    results_path: Path | None = None
    results_data: dict | None = None
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


_GENERATION_SYSTEM_PROMPT_HEADER = """\
You are a network topology engineer generating ngraph scenario YAML files.

You will receive a connectivity idea and must produce a complete ngraph
scenario YAML. After each attempt, you will receive the ngraph inspect
output showing what was actually built. Compare it against the original
intent and fix any mismatches.

Return ONLY valid YAML. No markdown fences, no explanation.
"""

# Loaded at module import from the skills directory if available,
# otherwise falls back to a built-in minimal reference.
_DSL_REFERENCE: str | None = None


def _load_dsl_reference() -> str:
    """Load the ngraph DSL skill reference for use as system prompt context."""
    global _DSL_REFERENCE
    if _DSL_REFERENCE is not None:
        return _DSL_REFERENCE

    # Try loading from the skills directory
    from pathlib import Path

    skill_paths = [
        Path(__file__).parent.parent.parent.parent
        / "skills"
        / "netgraph-dsl"
        / "SKILL.md",
        Path.home()
        / "ws"
        / "project_netgraph"
        / "skills"
        / "netgraph-dsl"
        / "SKILL.md",
    ]
    for skill_path in skill_paths:
        if skill_path.exists():
            _DSL_REFERENCE = skill_path.read_text()
            return _DSL_REFERENCE

    # Fallback: built-in minimal reference
    _DSL_REFERENCE = """\
CRITICAL RULES:
- Top-level keys: seed, network, risk_groups, demands, failures, workflow
- nodes and links go INSIDE the network key
- All links are bidirectional (ngraph adds reverse automatically)
- Use risk_groups: [name] on link defs to assign failure domains
- Node attrs enable failure targeting: {role: bb} matches scope: node

Failure policy structure (EXACT nesting required):
failures:
  policy_name:
    modes:
      - weight: 1.0
        rules:
          - scope: node        # node | link | risk_group
            mode: choice       # choice | all | random
            count: 1
            match:
              conditions:
                - attr: role
                  op: "=="     # == | != | contains | in
                  value: bb

TrafficMatrixPlacement workflow step (all fields required):
  - type: TrafficMatrixPlacement
    name: tm_step
    demand_set: tm
    failure_policy: policy_name
    iterations: 10
    parallelism: 1
    placement_rounds: auto
    seed: 42
    include_flow_details: true
    alpha_from_step: msd_baseline
    alpha_from_field: data.alpha_star
"""
    return _DSL_REFERENCE


def _get_generation_system_prompt() -> str:
    """Build the full system prompt for scenario generation."""
    return _GENERATION_SYSTEM_PROMPT_HEADER + "\n" + _load_dsl_reference()


_GENERATION_PROMPT_TEMPLATE = """\
Generate a complete ngraph scenario YAML for this connectivity idea:

{idea}

{feedback}

Return ONLY the YAML content, no explanation. Start with `seed:`.
"""

_REVISION_PROMPT_TEMPLATE = """\
The scenario you generated failed validation:

{inspect_summary}

The original connectivity idea was:
{idea}

{validation_errors}

Common issues:
- Demand source/target regex must match existing node names
- Failure rule mode must be "choice" (not "random") with count: N
- All nodes referenced in links must be defined in the nodes section
- WorkflowType is TrafficMatrixPlacement (not TrafficMatrixPerformance)

Fix the scenario YAML. Return ONLY the YAML content.
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
        try:
            ngraph_bin = require_executable("ngraph", env_var="NETLAB_NGRAPH_BIN")
        except RuntimeError as exc:
            return GenerationResult(success=False, error=str(exc))

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

            try:
                response = backend.generate(
                    prompt, system=_get_generation_system_prompt()
                )
            except (RuntimeError, OSError) as exc:
                last_inspect = InspectResult(
                    success=False,
                    errors=[f"LLM backend error: {exc}"],
                )
                continue

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

                # Run simulation as definitive validation.
                # For LLM-generated scenarios (10-20 nodes), this takes <1s.
                # Catches issues inspect misses: unresolved demand patterns,
                # invalid failure policies, workflow reference errors.
                sim_result = _run_simulation(scenario_path, ngraph_bin, work_dir)
                if not sim_result.success:
                    last_inspect = InspectResult(
                        success=False,
                        errors=[f"Simulation failed: {sim_result.error}"],
                    )
                    continue

                return GenerationResult(
                    success=True,
                    scenario_yaml=yaml_text,
                    scenario_path=scenario_path,
                    results_path=sim_result.results_path,
                    results_data=sim_result.results_data,
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


@dataclass
class _SimResult:
    """Internal result from a trial simulation run."""

    success: bool
    results_path: Path | None = None
    results_data: dict | None = None
    error: str = ""


def _run_simulation(scenario_path: Path, ngraph_bin: str, work_dir: Path) -> _SimResult:
    """Run ngraph on the scenario as a validation step.

    Returns the results if successful, or an error message if not.
    Timeout is short (60s) since LLM-generated scenarios are small.
    """
    import json

    results_dir = work_dir / "results"
    results_dir.mkdir(exist_ok=True)
    try:
        proc = subprocess.run(
            [ngraph_bin, "run", str(scenario_path), "-o", str(results_dir)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return _SimResult(success=False, error="Simulation timed out (60s)")

    if proc.returncode != 0:
        # Extract useful error from stderr
        stderr = proc.stderr.strip()
        error_lines = [
            line
            for line in stderr.splitlines()
            if "error" in line.lower() or "Error" in line
        ]
        error_msg = "; ".join(error_lines[-3:]) if error_lines else stderr[-300:]
        return _SimResult(success=False, error=error_msg)

    # Find and load results
    results_files = list(results_dir.glob("*.results.json"))
    if not results_files:
        return _SimResult(success=False, error="No results file produced")

    results_path = results_files[0]
    try:
        with results_path.open() as f:
            results_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return _SimResult(success=False, error=f"Failed to load results: {e}")

    return _SimResult(
        success=True,
        results_path=results_path,
        results_data=results_data,
    )


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
