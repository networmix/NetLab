"""NetGraph's Python API, shared by batch experiments and research loops."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ngraph.scenario import Scenario

from .artifacts import (
    cache_matches,
    cache_path,
    fingerprint,
    package_versions,
    record_cache,
    write_json_atomic,
)
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
                len(demands) for demands in scenario.demand_set.sets.values()
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
    start = time.perf_counter()
    inspection = None
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
        scenario.run()
        return SimulationResult(
            "success",
            inspection,
            scenario.results.to_dict(),
            duration_s=time.perf_counter() - start,
        )
    except Exception as exc:
        # This is the task boundary: errors must reach the caller, not kill a batch.
        return SimulationResult(
            "invalid" if inspection is None else "failed",
            inspection,
            error=f"{type(exc).__name__}: {exc}",
            duration_s=time.perf_counter() - start,
        )


def invalidate_results(path: Path) -> None:
    """Remove a result and its cache/outcome records before rebuilding its input."""
    for artifact in (path, cache_path(path), path.with_suffix(".run.json")):
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
        if outcome.status == "success":
            write_json_atomic(self.results_path, outcome.results)
            record_cache(self.results_path, self.cache_key)
        write_json_atomic(
            self.results_path.with_suffix(".run.json"),
            {
                "scenario": str(self.scenario_path),
                "status": outcome.status,
                "error": outcome.error,
                "duration_s": outcome.duration_s,
                "inspection": asdict(outcome.inspection)
                if outcome.inspection
                else None,
            },
        )
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
        if results_path == scenario_path:
            raise ValueError("Scenario and result paths must differ")
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be finite and positive or None")
        if results_path in self._destinations:
            raise ValueError(f"Duplicate simulation output: {results_path}")
        text = scenario_path.read_text(encoding="utf-8")
        self._destinations.add(results_path)
        key = fingerprint(
            {
                "schema": 1,
                "scenario": text,
                "packages": self.packages,
                "require_traffic": require_traffic,
            }
        )
        if not force and cache_matches(results_path, key):
            future: Future[SimulationResult] = Future()
            future.set_result(
                SimulationResult("cached", results=json.loads(results_path.read_text()))
            )
        else:
            # A failed rerun must not leave an old result that metrics can discover.
            invalidate_results(results_path)
            future = self.queue.submit(simulate, text, require_traffic, timeout=timeout)
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
