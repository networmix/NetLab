"""NetGraph's Python API, shared by batch experiments and research loops."""

from __future__ import annotations

import io
import json
import math
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from ngraph import prepare, run, write_json
from ngraph.scenario import Scenario

from .artifacts import (
    cache_matches,
    cache_path,
    fingerprint,
    package_versions,
    record_cache,
    write_json_atomic,
)
from .results import NO_FEASIBLE_ALPHA, feasible_alpha
from .run_record import load_run_record, run_record_path, validate_step_durations
from .tasks import TaskQueue


@dataclass(frozen=True)
class Inspection:
    node_count: int
    link_count: int
    risk_groups: list[str]
    demand_count: int
    workflow_steps: int

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> Inspection:
        return cls(
            node_count=len(scenario.network.nodes),
            link_count=len(scenario.network.links),
            risk_groups=sorted(scenario.network.risk_groups),
            demand_count=sum(
                len(demand_set.demands) for demand_set in scenario.demand_sets.values()
            ),
            workflow_steps=len(scenario.workflow),
        )

    def summary(self) -> str:
        return ", ".join(f"{key}={value}" for key, value in asdict(self).items())


@dataclass
class SimulationResult:
    status: Literal["success", "cached", "invalid", "failed", "timeout"]
    inspection: Inspection | None = None
    results: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_s: float = 0.0
    step_durations_sec: dict[str, float] = field(default_factory=dict)

    def run_record(self, scenario_path: Path | None = None) -> dict[str, Any]:
        return {
            "scenario": str(scenario_path) if scenario_path is not None else None,
            "status": self.status,
            "error": self.error,
            "duration_s": self.duration_s,
            "inspection": asdict(self.inspection) if self.inspection else None,
            "step_durations_sec": dict(self.step_durations_sec),
        }

    @property
    def success(self) -> bool:
        return self.status in {"success", "cached"}


def inspect_scenario(yaml_text: str) -> Inspection:
    """Validate and expand YAML, returning model facts without console parsing."""
    return Inspection.from_scenario(Scenario.from_yaml(yaml_text))


def simulate(yaml_text: str, require_traffic: bool = False) -> SimulationResult:
    """Load once and execute in the calling process. Exceptions become outcomes.

    For deadlines and isolation use ``run_simulation`` or ``SimulationBatch``.
    """
    start = perf_counter()
    inspection = None
    durations: dict[str, float] = {}

    @contextmanager
    def measure_step(step):
        start_step = perf_counter()
        try:
            yield
        finally:
            durations[step.name] = perf_counter() - start_step

    try:
        scenario = Scenario.from_yaml(yaml_text)
        inspection = Inspection.from_scenario(scenario)
        if require_traffic and (
            inspection.node_count < 2
            or inspection.link_count < 1
            or inspection.demand_count < 1
            or inspection.workflow_steps < 1
        ):
            return SimulationResult(
                "invalid",
                inspection,
                error="A research scenario needs at least two nodes, one link, one demand and one workflow step",
            )
        result = run(prepare(scenario), step_hook=measure_step)
        buffer = io.StringIO()
        write_json(result, buffer)
        exported = json.loads(buffer.getvalue())
        for step in exported["steps"]:
            if step["step_type"] == "MaximumSupportedDemand":
                feasible_alpha(step)
        return SimulationResult(
            "success",
            inspection,
            exported,
            duration_s=perf_counter() - start,
            step_durations_sec=durations,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if (
            isinstance(exc, ValueError)
            and str(exc).startswith("alpha_from_step ")
            and str(exc).endswith(": no tested feasible alpha")
        ):
            reason = NO_FEASIBLE_ALPHA
        elif isinstance(exc, ValueError) and str(exc) == NO_FEASIBLE_ALPHA:
            reason = NO_FEASIBLE_ALPHA
        # This is the task boundary: errors must reach the caller, not kill a batch.
        return SimulationResult(
            "invalid" if inspection is None else "failed",
            inspection,
            error=reason,
            duration_s=perf_counter() - start,
            step_durations_sec=durations,
        )


def invalidate_results(path: Path) -> None:
    """Remove a result and its cache/outcome records before rebuilding its input."""
    for artifact in (
        path,
        cache_path(path),
        run_record_path(path),
        cache_path(run_record_path(path)),
    ):
        artifact.unlink(missing_ok=True)


@dataclass
class SimulationJob:
    scenario_path: Path
    results_path: Path
    cache_key: str
    future: Future[SimulationResult]

    def result(self) -> SimulationResult:
        try:
            outcome = self.future.result()
        except TimeoutError as exc:
            outcome = SimulationResult("timeout", error=f"Simulation timed out: {exc}")
        except Exception as exc:
            outcome = SimulationResult("failed", error=f"{type(exc).__name__}: {exc}")
        if outcome.status == "cached":
            return outcome
        record_path = run_record_path(self.results_path)
        try:
            if outcome.status == "success":
                record = outcome.run_record(self.scenario_path)
                validate_step_durations(outcome.results, record)
                # Publish the record before the results become discoverable by metrics.
                write_json_atomic(record_path, record)
                write_json_atomic(self.results_path, outcome.results)
                record_cache(record_path, self.cache_key)
                record_cache(self.results_path, self.cache_key)
            else:
                write_json_atomic(record_path, outcome.run_record(self.scenario_path))
        except Exception:
            invalidate_results(self.results_path)
            raise
        return outcome


class SimulationBatch:
    """Submit scenarios to a queue managed by the caller."""

    def __init__(self, queue: TaskQueue) -> None:
        self.queue = queue
        self.packages = package_versions()
        self._destinations: set[Path] = set()

    def submit(
        self,
        scenario_path: Path,
        *,
        results_path: Path | None = None,
        timeout: float | None = 600,
        force: bool = False,
        require_traffic: bool = False,
    ) -> SimulationJob:
        scenario_path = scenario_path.resolve()
        results_path = (
            results_path or scenario_path.with_suffix(".results.json")
        ).resolve()
        if scenario_path in {
            results_path,
            run_record_path(results_path),
            cache_path(results_path),
            cache_path(run_record_path(results_path)),
        }:
            raise ValueError("Scenario and result paths must differ")
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be finite and positive or None")
        destinations = {
            results_path,
            run_record_path(results_path),
            cache_path(results_path),
            cache_path(run_record_path(results_path)),
        }
        if destinations & self._destinations:
            raise ValueError(f"Duplicate simulation output: {results_path}")
        text = scenario_path.read_text(encoding="utf-8")
        key = fingerprint(
            {
                "schema": 2,
                "scenario": text,
                "packages": self.packages,
                "require_traffic": require_traffic,
            }
        )
        cached_results = None
        record = None
        if not force and cache_matches(results_path, key):
            cached_results = json.loads(results_path.read_text())
            record = load_run_record(results_path, cached_results)
        if record is not None and cache_matches(run_record_path(results_path), key):
            future: Future[SimulationResult] = Future()
            inspection = record.get("inspection")
            future.set_result(
                SimulationResult(
                    "cached",
                    inspection=Inspection(**inspection) if inspection else None,
                    results=cached_results if cached_results is not None else {},
                    duration_s=record["duration_s"],
                    step_durations_sec=dict(record["step_durations_sec"]),
                )
            )
        else:
            # A failed rerun must not leave an old result that metrics can discover.
            invalidate_results(results_path)
            future = self.queue.submit(simulate, text, require_traffic, timeout=timeout)
        self._destinations.update(destinations)
        return SimulationJob(scenario_path, results_path, key, future)


def run_simulation(
    scenario_path: Path,
    *,
    results_path: Path | None = None,
    timeout: float | None = 600,
    force: bool = False,
) -> SimulationResult:
    """Execute one scenario with a deadline and publish only successful results."""
    with TaskQueue() as queue:
        return (
            SimulationBatch(queue)
            .submit(
                scenario_path,
                results_path=results_path,
                timeout=timeout,
                force=force,
            )
            .result()
        )
