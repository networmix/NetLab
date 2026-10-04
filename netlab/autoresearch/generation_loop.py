"""Generate, validate and simulate candidates through the shared execution API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from netlab.artifacts import write_text_atomic
from netlab.simulation import Inspection, SimulationBatch
from netlab.tasks import TaskQueue

from .backend import LLMBackend


@dataclass
class GenerationResult:
    success: bool
    scenario_yaml: str = ""
    scenario_path: Path | None = None
    results_path: Path | None = None
    results_data: dict | None = None
    inspect: Inspection | None = None
    iterations_used: int = 0
    error: str = ""


_GENERATION_SYSTEM_PROMPT_HEADER = """\
You are a network topology engineer generating ngraph scenario YAML files.

You will receive a connectivity idea and must produce a complete ngraph
scenario YAML. After each attempt, you will receive the structured inspection
summary showing what was actually built. Compare it against the original
intent and fix any mismatches.

Return ONLY valid YAML. No markdown fences, no explanation.
"""

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
    seed: 42
    include_flow_details: true
    alpha_from_step: msd_baseline
    alpha_from_field: data.alpha_star
"""


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
    max_iterations: int = 20,
    work_dir: Path | None = None,
    timeout_s: float = 60,
) -> GenerationResult:
    """Revise candidates using actual validation/execution errors.

    Without ``work_dir``, return in-memory YAML/results with no artifact paths.
    Each attempt owns a separate directory, preventing stale-result reuse.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if work_dir is None:
        with TemporaryDirectory(prefix="netlab-generation-") as temporary:
            result = run_generation_loop(
                idea, backend, max_iterations, Path(temporary), timeout_s
            )
            result.scenario_path = None
            result.results_path = None
            return result

    error = ""
    yaml_text = ""
    inspection = None
    with TaskQueue() as queue:
        batch = SimulationBatch(queue)
        for iteration in range(1, max_iterations + 1):
            prompt = (
                _GENERATION_PROMPT_TEMPLATE.format(idea=idea, feedback="")
                if iteration == 1
                else _REVISION_PROMPT_TEMPLATE.format(
                    idea=idea,
                    inspect_summary=inspection.summary()
                    if inspection
                    else "Invalid scenario",
                    validation_errors=error,
                )
            )
            try:
                response = backend.generate(
                    prompt,
                    system=_GENERATION_SYSTEM_PROMPT_HEADER + "\n" + _DSL_REFERENCE,
                )
            except (RuntimeError, OSError) as exc:
                error = f"LLM backend error: {exc}"
                continue
            yaml_text = _extract_yaml(response)
            scenario_path = work_dir / f"attempt_{iteration:03d}" / "scenario.yml"
            write_text_atomic(scenario_path, yaml_text)
            outcome = batch.submit(
                scenario_path, timeout=timeout_s, require_traffic=True
            ).result()
            inspection = outcome.inspection
            if outcome.success:
                return GenerationResult(
                    success=True,
                    scenario_yaml=yaml_text,
                    scenario_path=scenario_path,
                    results_path=scenario_path.with_suffix(".results.json"),
                    results_data=outcome.results,
                    inspect=inspection,
                    iterations_used=iteration,
                )
            error = outcome.error
    return GenerationResult(
        success=False,
        scenario_yaml=yaml_text,
        inspect=inspection,
        iterations_used=max_iterations,
        error=f"Failed after {max_iterations} attempts: {error}",
    )


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
