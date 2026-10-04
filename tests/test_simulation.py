"""Tests for API execution, queue cancellation, and cache integrity."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from netlab.artifacts import cache_path
from netlab.simulation import (
    SimulationBatch,
    inspect_scenario,
    run_simulation,
    simulate,
)
from netlab.tasks import TaskQueue

SCENARIO = """
seed: 7
network:
  nodes: {A: {}, B: {}}
  links: [{source: A, target: B, capacity: 100, cost: 1}]
demands:
  tm: [{source: A, target: B, volume: 10, mode: combine, flow_policy: SHORTEST_PATHS_ECMP}]
workflow:
  - {type: MaximumSupportedDemand, name: msd_baseline, demand_set: tm, resolution: 0.1}
"""


def _late_write(path: Path) -> None:
    time.sleep(2)
    path.write_text("worker was not stopped")


def _crash() -> None:
    import os

    os._exit(3)


def test_python_api_does_not_invoke_executables(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Simulation attempted an external command")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    inspection = inspect_scenario(SCENARIO)
    assert (inspection.node_count, inspection.link_count, inspection.demand_count) == (
        2,
        1,
        1,
    )
    result = simulate(SCENARIO)
    assert result.success
    assert result.results["steps"]["msd_baseline"]["data"][
        "alpha_star"
    ] == pytest.approx(10, abs=0.2)


def test_timeout_stops_work_and_queue_recovers(tmp_path):
    marker = tmp_path / "late"
    with TaskQueue() as queue:
        timed = queue.submit(_late_write, marker, timeout=0.1)
        following = queue.submit(pow, 3, 4, timeout=3)
        with pytest.raises(TimeoutError):
            timed.result()
        assert following.result() == 81
    time.sleep(2.1)
    assert not marker.exists()


def test_crashed_worker_is_replaced():
    from pebble import ProcessExpired

    with TaskQueue() as queue:
        crashed = queue.submit(_crash, timeout=3)
        with pytest.raises(ProcessExpired):
            crashed.result()
        assert queue.submit(pow, 2, 5, timeout=3).result() == 32


def test_cache_checks_inputs_dependencies_and_output(tmp_path):
    scenario = tmp_path / "scenario.yml"
    results = scenario.with_suffix(".results.json")
    scenario.write_text(SCENARIO)
    assert run_simulation(scenario).status == "success"
    assert run_simulation(scenario).status == "cached"
    results.write_text("{}")
    assert run_simulation(scenario).status == "success"
    scenario.write_text(SCENARIO.replace("capacity: 100", "capacity: 200"))
    assert run_simulation(scenario).status == "success"
    assert (
        json.loads(results.read_text())["steps"]["msd_baseline"]["data"]["alpha_star"]
        > 19
    )
    with TaskQueue() as queue:
        batch = SimulationBatch(queue)
        batch.packages["ngraph"]["version"] = "changed"
        assert batch.submit(scenario).result().status == "success"
    assert run_simulation(scenario).status == "success"
    scenario.write_text("invalid: true")
    assert run_simulation(scenario).status == "invalid"
    assert not results.exists()
    assert not cache_path(results).exists()


def test_invalid_task_does_not_block_valid_task(tmp_path):
    bad, good = tmp_path / "bad.yml", tmp_path / "good.yml"
    bad.write_text("invalid: true")
    good.write_text(SCENARIO)
    with TaskQueue(2) as queue:
        batch = SimulationBatch(queue)
        first, second = batch.submit(bad), batch.submit(good)
        assert first.result().status == "invalid"
        assert second.result().success
        with pytest.raises(ValueError, match="Duplicate"):
            batch.submit(good)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_cannot_delete_existing_result(tmp_path, timeout):
    scenario = tmp_path / "scenario.yml"
    output = tmp_path / "result.json"
    scenario.write_text(SCENARIO)
    output.write_text("original")
    with TaskQueue() as queue:
        with pytest.raises(ValueError, match="timeout"):
            SimulationBatch(queue).submit(
                scenario, results_path=output, timeout=timeout
            )
    assert output.read_text() == "original"


def test_output_cannot_replace_input(tmp_path):
    scenario = tmp_path / "scenario.yml"
    scenario.write_text(SCENARIO)
    with TaskQueue() as queue:
        with pytest.raises(ValueError, match="paths must differ"):
            SimulationBatch(queue).submit(scenario, results_path=scenario)
    assert scenario.read_text() == SCENARIO
