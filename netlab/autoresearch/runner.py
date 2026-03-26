"""AutoResearch runner: core experiment loop.

Provides:
- RunConfig: dataclass holding all runner configuration.
- AutoResearchRunner: init validation, main loop, subprocess execution,
  circuit breaker, resume detection, deduplication.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from netlab.autoresearch.backend import LLMBackend
from netlab.autoresearch.experiment_log import ExperimentLog, LogEntry
from netlab.autoresearch.hypothesis import (
    Hypothesis,
    HypothesisMerger,
    HypothesisTemplate,
)
from netlab.autoresearch.memory import ResearchMemory
from netlab.autoresearch.objective import ObjectiveFunction
from netlab.autoresearch.prompt import (
    ParseError,
    build_hypothesis_prompt,
    build_reflection_prompt,
    parse_hypothesis_response,
    render_memory_section,
)

logger = logging.getLogger(__name__)


def _default_ngraph_bin() -> str:
    """Resolve ngraph binary from the same venv as the current interpreter."""
    return os.path.join(os.path.dirname(sys.executable), "ngraph")


@dataclass
class RunConfig:
    project_dir: Path
    backend: LLMBackend
    max_experiments: int = 50
    timeout_s: int = 600
    seed: int = 42
    circuit_breaker_threshold: int = 5
    reflection_interval: int = 5


class AutoResearchRunner:
    """Runs the autoresearch experiment loop."""

    def __init__(self, config: RunConfig) -> None:
        self._config = config
        self._project_dir = Path(config.project_dir)
        self._status = "initialized"
        self._ngraph_call_count = 0

        # Resolve ngraph binary path (overridable for testing)
        self._ngraph_bin = _default_ngraph_bin()

        # Load project files
        self._program_md = (self._project_dir / "program.md").read_text(
            encoding="utf-8"
        )

        self._template = HypothesisTemplate(
            self._project_dir / "hypothesis_template.yml"
        )
        self._objective = ObjectiveFunction(self._project_dir / "objective.yml")

        # Detect generation mode from config.yml
        config_path = self._project_dir / "config.yml"
        if config_path.exists():
            with open(config_path) as f:
                project_config = yaml.safe_load(f) or {}
            self._generation_mode = project_config.get("generation_mode", "template")
            self._generator_module = project_config.get("generator_module", "")
            self._generator_function = project_config.get(
                "generator_function", "generate_scenario"
            )
            self._config_class = project_config.get("config_class", "")
        else:
            self._generation_mode = "template"
            self._generator_module = ""
            self._generator_function = ""
            self._config_class = ""

        # Mode-specific initialization
        if self._generation_mode == "template":
            base_scenario_path = self._project_dir / "base_scenario.yml"
            self._base_scenario_text = base_scenario_path.read_text(encoding="utf-8")

            # Validate base scenario has MSD workflow step
            self._validate_workflow()

            # Create merger and validate placeholders
            self._merger = HypothesisMerger(self._base_scenario_text, self._template)
            placeholder_errors = self._merger.validate_placeholders()
            if placeholder_errors:
                raise ValueError(
                    f"Placeholder validation failed: {'; '.join(placeholder_errors)}"
                )
            self._generator_fn = None
            self._config_class_ref = None
        elif self._generation_mode == "programmatic":
            self._base_scenario_text = ""
            self._merger = None
            self._generator_fn = self._load_generator()
            self._config_class_ref = self._load_config_class()
        else:
            raise ValueError(f"Unknown generation_mode: {self._generation_mode}")

        # Set up experiment log and results dir
        self._log = ExperimentLog(
            self._project_dir, direction=self._objective.direction
        )
        self._results_dir = self._project_dir / "results"
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # Research memory
        memory_dir = self._project_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory = ResearchMemory(memory_dir)
        self._memory.load()

        # State for tracking best
        self._best_entry: Optional[LogEntry] = None
        self._experiments_run = 0
        self._successful_since_reflection = 0

    def _validate_workflow(self) -> None:
        """Validate that the base scenario has a MaximumSupportedDemand workflow step."""
        scenario_data = yaml.safe_load(self._base_scenario_text)
        workflow = scenario_data.get("workflow", [])

        has_msd = False
        if isinstance(workflow, list):
            for step in workflow:
                if (
                    isinstance(step, dict)
                    and step.get("type") == "MaximumSupportedDemand"
                ):
                    has_msd = True
                    break
        elif isinstance(workflow, dict):
            for _step_name, step_data in workflow.items():
                if (
                    isinstance(step_data, dict)
                    and step_data.get("type") == "MaximumSupportedDemand"
                ):
                    has_msd = True
                    break

        if not has_msd:
            raise ValueError(
                "Base scenario must have a MaximumSupportedDemand workflow step. "
                "No step with type 'MaximumSupportedDemand' found in workflow."
            )

    def _load_generator(self):
        """Dynamically load the generator function from the configured module."""
        import importlib

        module = importlib.import_module(self._generator_module)
        return getattr(module, self._generator_function)

    def _load_config_class(self):
        """Dynamically load the config class for the generator."""
        if not self._config_class:
            return None
        import importlib

        parts = self._config_class.rsplit(".", 1)
        module = importlib.import_module(parts[0])
        return getattr(module, parts[1])

    def _generate_programmatic(self, hypothesis: Hypothesis) -> dict:
        """Generate scenario using the programmatic generator."""
        assert self._generator_fn is not None, "generator_fn not loaded"
        params = hypothesis.params
        if self._config_class_ref is not None:
            import dataclasses

            field_types = {
                f.name: f.type for f in dataclasses.fields(self._config_class_ref)
            }
            config_kwargs = {}
            for key, value in params.items():
                # Handle layout tuples encoded as strings (e.g. "16x4_16x4")
                if key.startswith("layout_") and isinstance(value, str):
                    parts = value.replace("x", ",").replace("_", ",").split(",")
                    value = tuple(int(p) for p in parts)
                # Coerce string enum values to the target field type
                elif isinstance(value, str) and key in field_types:
                    ft = field_types[key]
                    if ft == "int" or ft is int:
                        value = int(value)
                    elif ft == "float" or ft is float:
                        value = float(value)
                config_kwargs[key] = value
            config = self._config_class_ref(**config_kwargs)
            return self._generator_fn(config)
        else:
            return self._generator_fn(**params)

    @property
    def status(self) -> str:
        return self._status

    @property
    def ngraph_call_count(self) -> int:
        """Number of actual ngraph subprocess invocations (for testing)."""
        return self._ngraph_call_count

    def run(self) -> None:
        """Execute the main research loop."""
        self._status = "running"

        # Load existing history and re-derive best
        entries = self._log.load()
        self._best_entry = self._log.best_entry()

        if entries:
            logger.info("Resuming from experiment %d", len(entries))
            # Mandatory reflection on resume if memory has existing content
            if (
                self._memory.active_insights
                or self._memory.dead_ends
                or self._memory.strategy
            ):
                self._run_reflection(entries)

        # Build set of known param hashes for deduplication
        seen_hashes: dict[str, LogEntry] = {}
        for entry in entries:
            if entry.params_hash:
                seen_hashes[entry.params_hash] = entry

        # Main loop
        while self._experiments_run < self._config.max_experiments:
            # Check circuit breaker from log tail
            # Reload entries each iteration to get fresh consecutive_failures count
            consecutive_fails = self._log.consecutive_failures()
            if consecutive_fails >= self._config.circuit_breaker_threshold:
                self._status = "circuit_breaker"
                logger.warning(
                    "Circuit breaker tripped after %d consecutive failures",
                    consecutive_fails,
                )
                return

            # Determine experiment ID
            exp_id = self._log.next_experiment_id()

            # Build prompt
            history_text = self._log.windowed_history()
            system_prompt, user_prompt = build_hypothesis_prompt(
                program_md=self._program_md,
                template=self._template,
                history=history_text,
                memory_section=render_memory_section(self._memory),
                best=self._best_entry,
            )

            # Call LLM backend
            try:
                response = self._config.backend.generate(user_prompt, system_prompt)
            except Exception as exc:
                self._log_error_entry(
                    exp_id=exp_id,
                    status="backend_error",
                    error_detail=str(exc),
                )
                self._experiments_run += 1
                continue

            # Parse response
            try:
                params = parse_hypothesis_response(response)
            except ParseError as exc:
                self._log_error_entry(
                    exp_id=exp_id,
                    status="parse_error",
                    error_detail=str(exc),
                )
                self._experiments_run += 1
                continue

            # Create hypothesis and validate
            hypothesis = Hypothesis(params, self._template)
            validation_errors = hypothesis.validate()
            if validation_errors:
                self._log_error_entry(
                    exp_id=exp_id,
                    status="invalid_hypothesis",
                    error_detail="; ".join(validation_errors),
                    params=params,
                    params_hash=hypothesis.params_hash,
                )
                self._experiments_run += 1
                continue

            # Deduplication check
            if hypothesis.params_hash in seen_hashes:
                cached_entry = seen_hashes[hypothesis.params_hash]
                self._log_cached_entry(exp_id, hypothesis, cached_entry)
                self._experiments_run += 1
                continue

            # Generate scenario
            try:
                if self._generation_mode == "template":
                    assert self._merger is not None
                    scenario_dict = self._merger.merge(hypothesis)
                else:  # programmatic
                    scenario_dict = self._generate_programmatic(hypothesis)
            except (ValueError, Exception) as exc:
                self._log_error_entry(
                    exp_id=exp_id,
                    status="generation_error",
                    error_detail=str(exc),
                    params=params,
                    params_hash=hypothesis.params_hash,
                )
                self._experiments_run += 1
                continue

            # Inject seed
            scenario_dict["seed"] = self._config.seed

            # Write scenario to results/exp_NNN/scenario.yml
            exp_dir = self._results_dir / exp_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            scenario_path = exp_dir / "scenario.yml"
            with open(scenario_path, "w") as f:
                yaml.dump(scenario_dict, f, default_flow_style=False)

            # Execute ngraph
            start_time = time.monotonic()
            run_result = self._execute_ngraph(scenario_path, exp_dir)
            execution_time = time.monotonic() - start_time

            if run_result["status"] != "success":
                entry = LogEntry(
                    exp_id=exp_id,
                    params=hypothesis.params,
                    params_hash=hypothesis.params_hash,
                    status=run_result["status"],
                    metrics=None,
                    objective_score=None,
                    error_detail=run_result.get("error_detail"),
                    execution_time_s=round(execution_time, 2),
                    seed=self._config.seed,
                    timestamp=_now_iso(),
                )
                self._log.append(entry)
                seen_hashes[hypothesis.params_hash] = entry
                self._experiments_run += 1
                continue

            # Load results and evaluate
            results_path = exp_dir / "scenario.results.json"
            try:
                with open(results_path) as f:
                    results_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                entry = LogEntry(
                    exp_id=exp_id,
                    params=hypothesis.params,
                    params_hash=hypothesis.params_hash,
                    status="crash",
                    metrics=None,
                    objective_score=None,
                    error_detail=f"Failed to load results: {exc}",
                    execution_time_s=round(execution_time, 2),
                    seed=self._config.seed,
                    timestamp=_now_iso(),
                )
                self._log.append(entry)
                seen_hashes[hypothesis.params_hash] = entry
                self._experiments_run += 1
                continue

            # Evaluate with objective function
            try:
                obj_result = self._objective.evaluate(results_data)
            except (KeyError, ValueError) as exc:
                entry = LogEntry(
                    exp_id=exp_id,
                    params=hypothesis.params,
                    params_hash=hypothesis.params_hash,
                    status="validation_error",
                    metrics=None,
                    objective_score=None,
                    error_detail=f"Objective evaluation failed: {exc}",
                    execution_time_s=round(execution_time, 2),
                    seed=self._config.seed,
                    timestamp=_now_iso(),
                )
                self._log.append(entry)
                seen_hashes[hypothesis.params_hash] = entry
                self._experiments_run += 1
                continue

            # Compute BAC and merge into metrics (non-fatal on failure)
            all_metrics = dict(obj_result.all_metrics)
            try:
                from metrics.bac import compute_bac

                # Try tm_combined first (per-mode workflow), fall back to tm_placement
                step = (
                    "tm_combined"
                    if "tm_combined" in results_data.get("steps", {})
                    else "tm_placement"
                )
                bac = compute_bac(results_data, step_name=step)
                all_metrics["bac_auc"] = round(bac.auc_normalized, 6)
                if 0.99 in bac.quantiles_pct:
                    all_metrics["bac_p99"] = round(bac.quantiles_pct[0.99], 6)
            except Exception:
                pass  # BAC extraction is best-effort

            # Log success
            entry = LogEntry(
                exp_id=exp_id,
                params=hypothesis.params,
                params_hash=hypothesis.params_hash,
                status="success",
                metrics=all_metrics,
                objective_score=obj_result.score,
                error_detail=None,
                execution_time_s=round(execution_time, 2),
                seed=self._config.seed,
                timestamp=_now_iso(),
            )
            self._log.append(entry)
            seen_hashes[hypothesis.params_hash] = entry

            # Update best
            prev_best = self._best_entry
            if (
                self._best_entry is None
                or (
                    self._objective.direction == "maximize"
                    and obj_result.score
                    > (self._best_entry.objective_score or float("-inf"))
                )
                or (
                    self._objective.direction == "minimize"
                    and obj_result.score
                    < (self._best_entry.objective_score or float("inf"))
                )
            ):
                self._best_entry = entry
                self._write_best_hypothesis(entry)

            self._experiments_run += 1
            self._successful_since_reflection += 1

            # Trigger reflection on interval or when a previous best is superseded
            new_best_superseded = prev_best is not None and self._best_entry is entry
            if (
                new_best_superseded
                or self._successful_since_reflection >= self._config.reflection_interval
            ):
                all_entries = self._log.load()
                self._run_reflection(all_entries)
                self._successful_since_reflection = 0

        self._status = "completed"

    def _run_reflection(self, all_entries: list[LogEntry]) -> None:
        """Run a reflection cycle. Non-fatal: logs warnings on any failure."""
        try:
            # Use last reflection_interval entries as recent context
            recent = all_entries[-self._config.reflection_interval :]
            system_prompt, user_prompt = build_reflection_prompt(
                recent_entries=recent,
                memory=self._memory,
                best=self._best_entry,
            )
            response = self._config.backend.generate(user_prompt, system_prompt)
            err = self._memory.parse_reflection_output(response, self._log)
            if err:
                logger.warning("Reflection parse issues: %s", err)
            self._memory.save()
            logger.info("Reflection completed successfully")
        except Exception as exc:
            logger.warning("Reflection failed (non-fatal): %s", exc)

    def _execute_ngraph(self, scenario_path: Path, exp_dir: Path) -> dict:
        """Run ngraph inspect + run as subprocess. Returns status dict."""
        ngraph_bin = self._ngraph_bin

        # Run inspect first
        try:
            inspect_result = subprocess.run(
                [ngraph_bin, "inspect", str(scenario_path)],
                capture_output=True,
                text=True,
                timeout=self._config.timeout_s,
            )
            if inspect_result.returncode != 0:
                stderr = inspect_result.stderr
                if stderr and len(stderr) > 500:
                    stderr = stderr[-500:]
                return {
                    "status": "crash",
                    "error_detail": f"ngraph inspect failed: {stderr}",
                }
        except subprocess.TimeoutExpired:
            return {"status": "timeout_no_result"}

        # Run the scenario
        self._ngraph_call_count += 1
        try:
            run_result = subprocess.run(
                [ngraph_bin, "run", str(scenario_path), "-o", str(exp_dir)],
                capture_output=True,
                text=True,
                timeout=self._config.timeout_s,
            )
            if run_result.returncode != 0:
                stderr = run_result.stderr
                if stderr and len(stderr) > 500:
                    stderr = stderr[-500:]
                return {
                    "status": "crash",
                    "error_detail": f"ngraph run failed: {stderr}",
                }
        except subprocess.TimeoutExpired:
            return {"status": "timeout_no_result"}

        return {"status": "success"}

    def _log_error_entry(
        self,
        exp_id: str,
        status: str,
        error_detail: str,
        params: Optional[dict] = None,
        params_hash: Optional[str] = None,
    ) -> None:
        """Log an error entry to the experiment log."""
        entry = LogEntry(
            exp_id=exp_id,
            params=params or {},
            params_hash=params_hash or "",
            status=status,
            metrics=None,
            objective_score=None,
            error_detail=error_detail,
            execution_time_s=None,
            seed=self._config.seed,
            timestamp=_now_iso(),
        )
        self._log.append(entry)

    def _log_cached_entry(
        self,
        exp_id: str,
        hypothesis: Hypothesis,
        cached_entry: LogEntry,
    ) -> None:
        """Log a cached (deduplicated) entry reusing previous results."""
        entry = LogEntry(
            exp_id=exp_id,
            params=hypothesis.params,
            params_hash=hypothesis.params_hash,
            status="cached",
            metrics=cached_entry.metrics,
            objective_score=cached_entry.objective_score,
            error_detail=None,
            execution_time_s=None,
            seed=self._config.seed,
            timestamp=_now_iso(),
        )
        self._log.append(entry)

    def _write_best_hypothesis(self, entry: LogEntry) -> None:
        """Write the best hypothesis params to best_hypothesis.yml."""
        best_data = {
            "exp_id": entry.exp_id,
            "params": entry.params,
            "objective_score": entry.objective_score,
            "metrics": entry.metrics,
        }
        best_path = self._project_dir / "best_hypothesis.yml"
        with open(best_path, "w") as f:
            yaml.dump(best_data, f, default_flow_style=False)


def _now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()
