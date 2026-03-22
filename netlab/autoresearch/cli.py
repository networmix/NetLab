"""CLI handlers for the ``netlab autoresearch`` subcommand.

Provides:
- autoresearch_init: scaffold a new autoresearch project directory.
- autoresearch_run: load project, construct runner, execute experiment loop.
- _build_backend: factory for LLM backend from CLI args.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import textwrap
from pathlib import Path

import yaml

from netlab.autoresearch.backend import (
    ClaudeCLIBackend,
    CodexCLIBackend,
    LLMBackend,
    MockBackend,
    OpenAICompatibleBackend,
)
from netlab.autoresearch.runner import AutoResearchRunner, RunConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default templates
# ---------------------------------------------------------------------------

_DEFAULT_PROGRAM_MD = textwrap.dedent("""\
    You are an autonomous network researcher. Your goal is to find parameter
    values that optimize the objective function defined in objective.yml.

    Each iteration you will:
    1. Review the experiment history and current best result.
    2. Propose a new set of parameters (a hypothesis).
    3. The system will run the experiment and report back.

    Be systematic: explore the parameter space, form hypotheses about
    which parameters matter most, and refine your approach over time.
""")

_DEFAULT_OBJECTIVE_YAML = textwrap.dedent("""\
    direction: maximize
    primary_metric: alpha_star
    metrics:
      alpha_star:
        path: "steps.msd_baseline.data.alpha_star"
""")


def _build_default_template(base_scenario_path: Path) -> str:
    """Build a minimal hypothesis_template.yml.

    If the base scenario contains ${{...}} placeholders, create params
    matching those names with reasonable defaults. Otherwise, create a
    single ``link_capacity`` placeholder template.
    """
    import re

    text = base_scenario_path.read_text(encoding="utf-8")
    placeholders = set(re.findall(r"\$\{\{(\w+)\}\}", text))

    if placeholders:
        params: dict[str, dict] = {}
        for name in sorted(placeholders):
            params[name] = {
                "type": "float",
                "range": [0.0, 100.0],
                "default": 1.0,
                "description": f"Auto-detected placeholder: {name}",
            }
        return yaml.dump({"params": params}, default_flow_style=False, sort_keys=False)

    # No placeholders found — provide a minimal default
    return textwrap.dedent("""\
        params:
          link_capacity:
            type: float
            range: [0.5, 10.0]
            default: 2.0
            description: "Default template param (replace with your own)"
    """)


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------


def _build_backend(args: argparse.Namespace) -> LLMBackend:
    """Construct an LLM backend from CLI arguments."""
    backend_name: str = args.backend

    if backend_name == "mock":
        return _build_mock_backend(args)
    elif backend_name == "claude-cli":
        return ClaudeCLIBackend()
    elif backend_name == "codex-cli":
        return CodexCLIBackend()
    elif backend_name == "openai":
        import os

        base_url = getattr(args, "openai_base_url", None) or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com"
        )
        model = getattr(args, "openai_model", None) or os.environ.get(
            "OPENAI_MODEL", "gpt-4"
        )
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return OpenAICompatibleBackend(base_url=base_url, model=model, api_key=api_key)
    else:
        print(
            f"Unknown backend: {backend_name!r}. "
            "Use 'mock', 'claude-cli', 'codex-cli', or 'openai'.",
            file=sys.stderr,
        )
        sys.exit(1)


def _build_mock_backend(args: argparse.Namespace) -> MockBackend:
    """Build a MockBackend that generates plausible YAML responses.

    Reads the hypothesis template from the project directory to produce
    responses whose params match the template definitions. Generates
    ``max_experiments`` scripted responses using default values with
    small perturbations.
    """
    project_dir = Path(args.project_dir)
    template_path = project_dir / "hypothesis_template.yml"

    if not template_path.exists():
        # Fallback: return responses with a generic param
        n = getattr(args, "max_experiments", 10)
        responses = [
            textwrap.dedent(f"""\
                Trying default with variation {i}.

                ```yaml
                params:
                  link_capacity: {2.0 + i * 0.5}
                ```
            """)
            for i in range(n)
        ]
        return MockBackend(responses)

    with open(template_path) as f:
        data = yaml.safe_load(f)

    params_data = data.get("params") or data.get("parameters") or {}
    n = getattr(args, "max_experiments", 10)

    import random

    rng = random.Random(getattr(args, "seed", 42))

    responses: list[str] = []
    for i in range(n):
        param_lines: list[str] = []
        for name, spec in params_data.items():
            ptype = spec.get("type", "float")
            if ptype == "enum":
                values = spec.get("values", ["default"])
                val = rng.choice(values)
                param_lines.append(f"  {name}: {val}")
            elif ptype == "int":
                lo, hi = spec.get("range", [1, 100])
                step = spec.get("step", 1)
                val = rng.randrange(int(lo), int(hi) + 1, int(step))
                param_lines.append(f"  {name}: {val}")
            else:  # float
                lo, hi = spec.get("range", [0.0, 10.0])
                val = round(rng.uniform(float(lo), float(hi)), 4)
                param_lines.append(f"  {name}: {val}")

        params_block = "\n".join(param_lines)
        response = textwrap.dedent(f"""\
            Experiment {i + 1}: trying a new configuration.

            ```yaml
            params:
            {params_block}
            ```
        """)
        responses.append(response)

    return MockBackend(responses)


def autoresearch_init(args: argparse.Namespace) -> None:
    """Create a new autoresearch project directory."""
    base_scenario: Path = args.base_scenario
    output_dir: Path = args.output

    if not base_scenario.exists():
        print(f"Base scenario does not exist: {base_scenario}", file=sys.stderr)
        sys.exit(1)

    # Validate the base scenario has a MaximumSupportedDemand workflow step
    try:
        scenario_text = base_scenario.read_text(encoding="utf-8")
        scenario_data = yaml.safe_load(scenario_text)
    except Exception as exc:
        print(f"Failed to read base scenario: {exc}", file=sys.stderr)
        sys.exit(1)

    workflow = scenario_data.get("workflow", [])
    has_msd = False
    if isinstance(workflow, list):
        for step in workflow:
            if isinstance(step, dict) and step.get("type") == "MaximumSupportedDemand":
                has_msd = True
                break
    elif isinstance(workflow, dict):
        for _name, step_data in workflow.items():
            if (
                isinstance(step_data, dict)
                and step_data.get("type") == "MaximumSupportedDemand"
            ):
                has_msd = True
                break

    if not has_msd:
        print(
            "Base scenario must have a MaximumSupportedDemand workflow step. "
            "No step with type 'MaximumSupportedDemand' found in workflow.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create project directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy base scenario
    shutil.copy2(base_scenario, output_dir / "base_scenario.yml")

    # Write default files
    (output_dir / "program.md").write_text(_DEFAULT_PROGRAM_MD, encoding="utf-8")
    (output_dir / "objective.yml").write_text(_DEFAULT_OBJECTIVE_YAML, encoding="utf-8")
    (output_dir / "hypothesis_template.yml").write_text(
        _build_default_template(base_scenario), encoding="utf-8"
    )

    # Create empty directories
    (output_dir / "memory").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)

    print(f"Autoresearch project initialized at: {output_dir}")


def autoresearch_run(args: argparse.Namespace) -> None:
    """Run the autoresearch experiment loop."""
    project_dir = Path(args.project_dir)

    if not project_dir.exists():
        print(f"Project directory does not exist: {project_dir}", file=sys.stderr)
        sys.exit(1)

    if not project_dir.is_dir():
        print(f"Not a directory: {project_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine generation mode from config.yml (if present)
    config_yml_path = project_dir / "config.yml"
    generation_mode = "template"
    if config_yml_path.exists():
        with open(config_yml_path) as f:
            project_config = yaml.safe_load(f) or {}
        generation_mode = project_config.get("generation_mode", "template")

    # Validate required files exist
    required = [
        "program.md",
        "objective.yml",
        "hypothesis_template.yml",
    ]
    if generation_mode == "template":
        required.append("base_scenario.yml")
    for name in required:
        if not (project_dir / name).exists():
            print(f"Missing required file: {project_dir / name}", file=sys.stderr)
            sys.exit(1)

    backend = _build_backend(args)

    config = RunConfig(
        project_dir=project_dir,
        backend=backend,
        max_experiments=args.max_experiments,
        timeout_s=args.timeout,
        seed=args.seed,
    )

    try:
        runner = AutoResearchRunner(config)
    except ValueError as exc:
        print(f"Project validation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    runner.run()

    if runner.status == "circuit_breaker":
        print(
            "Run halted: circuit breaker tripped (too many consecutive failures).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Run completed. Status: {runner.status}")
