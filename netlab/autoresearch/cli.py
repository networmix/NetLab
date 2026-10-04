"""Create research projects and run them from the CLI."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import textwrap
from pathlib import Path

import yaml

from netlab.autoresearch.backend import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_OPENAI_MODEL,
    SUPPORTED_BACKENDS,
    ClaudeCLIBackend,
    CodexCLIBackend,
    LLMBackend,
    MockBackend,
    OpenAICompatibleBackend,
)
from netlab.autoresearch.hypothesis import validate_template_workflow
from netlab.autoresearch.runner import AutoResearchRunner, RunConfig

logger = logging.getLogger(__name__)


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
    """Define parameters for the base scenario's ``${{...}}`` placeholders.

    Use ``link_capacity`` when the scenario has no placeholders.
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

    return textwrap.dedent("""\
        params:
          link_capacity:
            type: float
            range: [0.5, 10.0]
            default: 2.0
            description: "Default template param (replace with your own)"
    """)


def _build_backend(args: argparse.Namespace) -> LLMBackend:
    """Construct an LLM backend from CLI arguments."""
    backend_name: str = args.backend
    backend_bin = getattr(args, "backend_bin", None)

    if backend_name == "mock":
        return _build_mock_backend(args)
    elif backend_name == "claude-cli":
        model = _resolve_model_arg(
            args,
            generic_attr="model",
            env_var="CLAUDE_MODEL",
            default=DEFAULT_CLAUDE_MODEL,
        )
        return ClaudeCLIBackend(model=model, command=backend_bin)
    elif backend_name == "codex-cli":
        model = _resolve_model_arg(
            args,
            generic_attr="model",
            env_var="CODEX_MODEL",
            default=DEFAULT_CODEX_MODEL,
        )
        return CodexCLIBackend(model=model, command=backend_bin)
    elif backend_name == "openai":
        base_url = getattr(args, "openai_base_url", None) or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com"
        )
        model = _resolve_model_arg(
            args,
            generic_attr="model",
            env_var="OPENAI_MODEL",
            default=DEFAULT_OPENAI_MODEL,
        )
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return OpenAICompatibleBackend(base_url=base_url, model=model, api_key=api_key)
    else:
        print(
            f"Unknown backend: {backend_name!r}. "
            f"Use {', '.join(repr(name) for name in SUPPORTED_BACKENDS)}.",
            file=sys.stderr,
        )
        sys.exit(1)


def _resolve_model_arg(
    args: argparse.Namespace,
    *,
    generic_attr: str,
    env_var: str,
    default: str,
) -> str:
    """Choose the model from the CLI flag, then the environment, then the default."""
    generic_value = getattr(args, generic_attr, None)
    if generic_value:
        return generic_value

    return os.environ.get(env_var, default)


def _build_mock_backend(args: argparse.Namespace) -> MockBackend:
    """Generate scripted responses from template defaults with small perturbations."""
    project_dir = Path(args.project_dir)
    template_path = project_dir / "hypothesis_template.yml"

    if not template_path.exists():
        # Use a generic parameter when the project has no template.
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

    params_data = data.get("params") or {}
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
            else:
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

    try:
        validate_template_workflow(base_scenario.read_text(encoding="utf-8"))
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(base_scenario, output_dir / "base_scenario.yml")

    (output_dir / "program.md").write_text(_DEFAULT_PROGRAM_MD, encoding="utf-8")
    (output_dir / "objective.yml").write_text(_DEFAULT_OBJECTIVE_YAML, encoding="utf-8")
    (output_dir / "hypothesis_template.yml").write_text(
        _build_default_template(base_scenario), encoding="utf-8"
    )

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

    config_yml_path = project_dir / "config.yml"
    generation_mode = "template"
    if config_yml_path.exists():
        with open(config_yml_path) as f:
            project_config = yaml.safe_load(f) or {}
        generation_mode = project_config.get("generation_mode", "template")

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
