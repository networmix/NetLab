"""Generate, simulate, and score research candidates with resumable experiment logs."""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from netlab.artifacts import (
    fingerprint,
    package_versions,
    sha256_file,
    write_text_atomic,
)
from netlab.autoresearch.backend import LLMBackend
from netlab.autoresearch.experiment_log import ExperimentLog, LogEntry
from netlab.autoresearch.hypothesis import (
    Hypothesis,
    HypothesisMerger,
    HypothesisTemplate,
    validate_template_workflow,
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
from netlab.simulation import run_simulation

logger = logging.getLogger(__name__)


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
        for name in (
            "timeout_s",
            "circuit_breaker_threshold",
            "reflection_interval",
        ):
            if getattr(config, name) < 1:
                raise ValueError(f"{name} must be positive")
        if config.max_experiments < 0:
            raise ValueError("max_experiments must be nonnegative")
        self._config = config
        self._project_dir = Path(config.project_dir)
        self._status = "initialized"
        self._ngraph_call_count = 0

        self._program_md = (self._project_dir / "program.md").read_text(
            encoding="utf-8"
        )

        self._template = HypothesisTemplate(
            self._project_dir / "hypothesis_template.yml"
        )
        self._objective = ObjectiveFunction(self._project_dir / "objective.yml")

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

        if self._generation_mode == "template":
            base_scenario_path = self._project_dir / "base_scenario.yml"
            self._base_scenario_text = base_scenario_path.read_text(encoding="utf-8")

            validate_template_workflow(self._base_scenario_text)

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

        self._log = ExperimentLog(self._project_dir)
        inputs = {
            name: sha256_file(self._project_dir / name)
            for name in (
                "program.md",
                "hypothesis_template.yml",
                "objective.yml",
                "base_scenario.yml",
                "config.yml",
            )
            if (self._project_dir / name).exists()
        }
        for name, obj in (
            ("generator", self._generator_fn),
            ("config_class", self._config_class_ref),
        ):
            if obj is not None:
                source = inspect.getsourcefile(obj)
                if source is None:
                    raise ValueError(f"Cannot fingerprint {name}")
                inputs[name] = sha256_file(Path(source))
        identity = fingerprint(
            {"inputs": inputs, "seed": config.seed, "packages": package_versions()}
        )
        previous = self._log.config_hash()
        if previous != identity and (previous is not None or self._log.load()):
            raise ValueError(
                "Research inputs or dependencies changed, or the log has no provenance. Use a new project directory."
            )
        if previous is None:
            self._log.write_config_hash(identity)
        self._results_dir = self._project_dir / "results"
        self._results_dir.mkdir(parents=True, exist_ok=True)

        memory_dir = self._project_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory = ResearchMemory(memory_dir)
        self._memory.load()

        self._best_entry: Optional[LogEntry] = None
        self._experiments_run = 0
        self._successful_since_reflection = 0

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
            config = self._config_class_ref(**params)
            return self._generator_fn(config)
        else:
            return self._generator_fn(**params)

    @property
    def status(self) -> str:
        return self._status

    @property
    def ngraph_call_count(self) -> int:
        """Number of uncached simulation attempts."""
        return self._ngraph_call_count

    def run(self) -> None:
        """Execute the main research loop."""
        self._status = "running"

        entries = self._log.load()
        self._best_entry = self._log.best_entry()
        if self._best_entry is not None:
            self._write_best_hypothesis(self._best_entry)

        if entries:
            logger.info("Resuming from experiment %d", len(entries))
            # Mandatory reflection on resume if memory has existing content
            if (
                self._memory.active_insights
                or self._memory.dead_ends
                or self._memory.strategy
            ):
                self._run_reflection(entries)

        seen_hashes: dict[str, LogEntry] = {}
        for entry in entries:
            if entry.params_hash and entry.status in {"success", "cached"}:
                seen_hashes[entry.params_hash] = entry

        while self._experiments_run < self._config.max_experiments:
            consecutive_fails = self._log.consecutive_failures()
            if consecutive_fails >= self._config.circuit_breaker_threshold:
                self._status = "circuit_breaker"
                logger.warning(
                    "Circuit breaker tripped after %d consecutive failures",
                    consecutive_fails,
                )
                return

            exp_id = self._log.next_experiment_id()

            history_text = self._log.windowed_history()
            system_prompt, user_prompt = build_hypothesis_prompt(
                program_md=self._program_md,
                template=self._template,
                history=history_text,
                memory_section=render_memory_section(self._memory),
                best=self._best_entry,
            )

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

            hypothesis = Hypothesis(params, self._template)
            validation_errors = hypothesis.validate()
            if validation_errors:
                self._log_error_entry(
                    exp_id=exp_id,
                    status="invalid_hypothesis",
                    error_detail="; ".join(validation_errors),
                    params=params,
                )
                self._experiments_run += 1
                continue

            if hypothesis.params_hash in seen_hashes and seen_hashes[
                hypothesis.params_hash
            ].status in {"success", "cached"}:
                cached_entry = seen_hashes[hypothesis.params_hash]
                self._log_cached_entry(exp_id, hypothesis, cached_entry)
                self._experiments_run += 1
                continue

            try:
                if self._generation_mode == "template":
                    assert self._merger is not None
                    scenario_dict = self._merger.merge(hypothesis)
                else:
                    scenario_dict = self._generate_programmatic(hypothesis)
            except Exception as exc:
                self._log_error_entry(
                    exp_id=exp_id,
                    status="generation_error",
                    error_detail=str(exc),
                    params=params,
                    params_hash=hypothesis.params_hash,
                )
                self._experiments_run += 1
                continue

            scenario_dict["seed"] = self._config.seed

            exp_dir = self._results_dir / exp_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            scenario_path = exp_dir / "scenario.yml"
            write_text_atomic(scenario_path, yaml.safe_dump(scenario_dict))

            start_time = time.monotonic()
            self._ngraph_call_count += 1
            run_result = run_simulation(scenario_path, timeout=self._config.timeout_s)
            execution_time = time.monotonic() - start_time

            if not run_result.success:
                entry = LogEntry(
                    exp_id=exp_id,
                    params=hypothesis.params,
                    params_hash=hypothesis.params_hash,
                    status="timeout_no_result"
                    if run_result.status == "timeout"
                    else "crash",
                    metrics=None,
                    objective_score=None,
                    error_detail=run_result.error,
                    execution_time_s=round(execution_time, 2),
                    seed=self._config.seed,
                    timestamp=_now_iso(),
                )
                self._log.append(entry)
                seen_hashes[hypothesis.params_hash] = entry
                self._experiments_run += 1
                continue

            results_data = run_result.results

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
                from netlab.metrics.bac import compute_bac

                # Try tm_combined first (per-mode workflow), fall back to tm_placement
                step = (
                    "tm_combined"
                    if "tm_combined" in results_data.get("steps", {})
                    else "tm_placement"
                )
                bac = compute_bac(results_data, step_name=step)
                all_metrics["bac_auc"] = bac.auc_normalized
                all_metrics["bw_p99_pct"] = bac.bw_at_probability_pct[99.0]
            except (ValueError, KeyError) as exc:
                logger.warning("BAC unavailable: %s", exc)

            entry = LogEntry(
                exp_id=exp_id,
                params=hypothesis.params,
                params_hash=hypothesis.params_hash,
                status="success" if obj_result.status == "feasible" else "infeasible",
                metrics=all_metrics,
                objective_score=obj_result.score,
                error_detail=None,
                execution_time_s=round(execution_time, 2),
                seed=self._config.seed,
                timestamp=_now_iso(),
            )
            self._log.append(entry)
            seen_hashes[hypothesis.params_hash] = entry

            prev_best = self._best_entry
            if obj_result.status == "feasible" and (
                self._best_entry is None
                or self._best_entry.objective_score is None
                or obj_result.score > self._best_entry.objective_score
            ):
                self._best_entry = entry
                self._write_best_hypothesis(entry)

            self._experiments_run += 1
            self._successful_since_reflection += 1

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
        write_text_atomic(
            best_path, yaml.safe_dump(best_data, default_flow_style=False)
        )


def _now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()
