"""Batch planning and orchestration, independent of command-line parsing."""

from __future__ import annotations

import math
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import (
    package_versions,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)
from .simulation import SimulationBatch, SimulationResult, invalidate_results
from .tasks import TaskQueue
from .topology import build_seed_scenario, generate_graph


@dataclass(frozen=True)
class PipelineConfig:
    masters: list[Path]
    seeds: list[int]
    output_dir: Path
    graphs_dir: Path | None = None
    build_jobs: int = 1
    run_jobs: int = 1
    build_timeout: float | None = None
    run_timeout: float | None = 600
    force: bool = False
    force_run: bool = False

    def __post_init__(self) -> None:
        if not self.masters or not self.seeds:
            raise ValueError("At least one master configuration and seed are required")
        if len({p.stem for p in self.masters}) != len(self.masters):
            raise ValueError("Master configuration names must be unique")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique")
        if self.build_jobs < 1 or self.run_jobs < 1:
            raise ValueError("Job counts must be positive")
        for timeout in (self.build_timeout, self.run_timeout):
            if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
                raise ValueError("Timeouts must be positive or None")


@dataclass
class PipelineResult:
    scenarios: list[Path] = field(default_factory=list)
    simulations: dict[Path, SimulationResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return not self.errors


def discover_configs(path: Path) -> list[Path]:
    if path.is_file() and path.suffix in {".yaml", ".yml"}:
        return [path.resolve()]
    if not path.is_dir():
        raise ValueError(f"Not a YAML file or configuration directory: {path}")
    configs = sorted(
        p.resolve()
        for p in path.iterdir()
        if p.suffix in {".yaml", ".yml"} and p.is_file()
    )
    if not configs:
        raise ValueError(f"No YAMLs under {path}")
    return configs


def _invalidate_scenario(path: Path, *, remove_input: bool = False) -> None:
    invalidate_results(path.with_suffix(".results.json"))
    if remove_input:
        path.unlink(missing_ok=True)


def _scenario_path(output: Path, master: Path, seed: int) -> Path:
    name = f"{master.stem}__seed{seed}"
    return output / master.stem / name / f"{name}_scenario.yml"


def run_pipeline(config: PipelineConfig, *, simulate: bool = True) -> PipelineResult:
    """Generate graphs, assemble scenarios, optionally simulate, then record outcomes.

    Independent tasks continue after a failure. Every failed stage is included in
    the returned result and provenance; the CLI can therefore return nonzero.
    """
    result = PipelineResult()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packages": package_versions(),
        "seeds": config.seeds,
        "settings": {
            "build_jobs": config.build_jobs,
            "run_jobs": config.run_jobs,
            "build_timeout": config.build_timeout,
            "run_timeout": config.run_timeout,
            "force": config.force,
            "force_run": config.force_run,
        },
        "configs": {str(p): sha256_file(p) for p in config.masters},
        "libraries": {
            str(p): sha256_file(p) for p in sorted(Path("lib").glob("*.yml"))
        },
        "graphs": {},
    }
    graphs: dict[Path, Path] = {}
    with TaskQueue(config.build_jobs) as queue:
        generating: dict[Path, Future[Path]] = {}
        for master in config.masters:
            if config.graphs_dir is None:
                generating[master] = queue.submit(
                    generate_graph,
                    master,
                    output_dir / master.stem,
                    config.force,
                    timeout=config.build_timeout,
                )
            else:
                graphs[master] = (
                    config.graphs_dir.resolve() / f"{master.stem}_integrated_graph.json"
                )
        for master, future in generating.items():
            try:
                graphs[master] = future.result()
            except Exception as exc:
                result.errors[f"{master.stem}/generate"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                for seed in config.seeds:
                    _invalidate_scenario(
                        _scenario_path(output_dir, master, seed), remove_input=True
                    )

        building: dict[Path, Future[Path]] = {}
        previous: dict[Path, str | None] = {}
        for master, graph in graphs.items():
            try:
                provenance["graphs"][str(graph)] = sha256_file(graph)
            except OSError as exc:
                result.errors[f"{master.stem}/graph"] = str(exc)
                for seed in config.seeds:
                    _invalidate_scenario(
                        _scenario_path(output_dir, master, seed), remove_input=True
                    )
                continue
            for seed in config.seeds:
                scenario = _scenario_path(output_dir, master, seed)
                previous[scenario] = (
                    sha256_file(scenario) if scenario.exists() else None
                )
                building[scenario] = queue.submit(
                    build_seed_scenario,
                    master,
                    graph,
                    scenario,
                    seed,
                    timeout=config.build_timeout,
                )
        for scenario, future in building.items():
            try:
                result.scenarios.append(future.result())
                if previous[scenario] != sha256_file(scenario):
                    _invalidate_scenario(scenario)
            except Exception as exc:
                _invalidate_scenario(scenario, remove_input=True)
                result.errors[str(scenario)] = f"{type(exc).__name__}: {exc}"

    if simulate:
        with TaskQueue(config.run_jobs) as queue:
            batch = SimulationBatch(queue)
            jobs = [
                batch.submit(
                    p,
                    timeout=config.run_timeout,
                    force=config.force or config.force_run,
                )
                for p in result.scenarios
            ]
            for job in jobs:
                outcome = job.result()
                result.simulations[job.scenario_path] = outcome
                if not outcome.success:
                    result.errors[str(job.scenario_path)] = outcome.error

    for master in config.masters:
        rows = ["Scenario\tStatus\tError"]
        for scenario, outcome in result.simulations.items():
            if scenario.parent.parent.name == master.stem:
                error = outcome.error.replace("\n", " ").replace("\t", " ")
                rows.append(f"{scenario}\t{outcome.status}\t{error}")
        write_text_atomic(
            output_dir / "_run_summaries" / f"{master.stem}.tsv", "\n".join(rows) + "\n"
        )
    provenance["errors"] = result.errors
    provenance["scenarios"] = {str(p): sha256_file(p) for p in result.scenarios}
    provenance["results"] = {
        str(p): outcome.status for p, outcome in result.simulations.items()
    }
    write_json_atomic(output_dir / "provenance.json", provenance)
    return result
