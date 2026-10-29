"""Tests for API execution, queue cancellation, and cache integrity."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from netlab.artifacts import cache_path
from netlab.results import feasible_alpha, require_step
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
  tm: [{source: A, target: B, volume: 10, mode: combine, preset: SHORTEST_PATHS_ECMP}]
workflow:
  - {type: MaximumSupportedDemand, name: msd_baseline, demand_set: tm, alpha_tolerance: 0.1}
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
    assert feasible_alpha(
        require_step(result.results, "msd_baseline")
    ) == pytest.approx(10, abs=0.2)


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
        feasible_alpha(require_step(json.loads(results.read_text()), "msd_baseline"))
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


@pytest.mark.parametrize("placement", [False, True])
def test_none_feasible_is_a_failed_run(placement):
    import yaml

    doc = yaml.safe_load(SCENARIO)
    doc["workflow"][0]["alpha_min"] = 100
    doc["workflow"][0]["alpha_start"] = 100
    if placement:
        doc["workflow"].append(
            {
                "type": "TrafficMatrixPlacement",
                "name": "tm_placement",
                "demand_set": "tm",
                "alpha_from_step": "msd_baseline",
            }
        )
    result = simulate(yaml.safe_dump(doc))
    assert result.status == "failed"
    assert "none_feasible" in result.error
    assert result.results == {}
    doc["workflow"][0]["alpha_min"] = 0.001
    doc["workflow"][0]["alpha_start"] = 1
    assert simulate(yaml.safe_dump(doc)).success


def test_step_hook_excludes_preparation_and_native_json_stays_native(monkeypatch):
    from contextlib import contextmanager

    import netlab.simulation as simulation

    now = 0.0
    original_prepare, original_run = simulation.prepare, simulation.run

    def prepare(scenario):
        nonlocal now
        now += 100
        return original_prepare(scenario)

    def run(prepared, *, step_hook):
        @contextmanager
        def measured(step):
            nonlocal now
            with step_hook(step):
                yield
                now += {"msd_baseline": 3, "tm_placement": 7}[step.name]

        return original_run(prepared, step_hook=measured)

    monkeypatch.setattr(simulation, "perf_counter", lambda: now)
    monkeypatch.setattr(simulation, "prepare", prepare)
    monkeypatch.setattr(simulation, "run", run)
    outcome = simulate(
        SCENARIO
        + "  - {type: TrafficMatrixPlacement, name: tm_placement, demand_set: tm, alpha_from_step: msd_baseline}\n"
    )
    assert outcome.success, outcome.error
    assert outcome.step_durations_sec == {"msd_baseline": 3, "tm_placement": 7}
    assert outcome.duration_s == 110
    assert set(outcome.results) == {"metadata", "units_report", "steps"}
    assert all(
        "duration" not in key for step in outcome.results["steps"] for key in step
    )


def test_cache_preserves_original_timing_record_and_detects_tampering(tmp_path):
    from netlab.run_record import run_record_path

    scenario = tmp_path / "scenario.yml"
    scenario.write_text(SCENARIO)
    path = scenario.with_suffix(".results.json")
    first = run_simulation(scenario)
    record_path = run_record_path(path)
    original = record_path.read_bytes()
    cached = run_simulation(scenario)
    assert cached.status == "cached"
    assert first.step_durations_sec == cached.step_durations_sec
    assert first.duration_s == cached.duration_s
    assert first.inspection == cached.inspection
    assert first.step_durations_sec["msd_baseline"] > 0
    assert record_path.read_bytes() == original
    record = json.loads(original)
    record["step_durations_sec"]["msd_baseline"] = 1000000
    record_path.write_text(json.dumps(record))
    assert run_simulation(scenario).status == "success"
    assert run_simulation(scenario, force=True).status == "success"


@pytest.mark.parametrize("damage", ["missing_file", "missing_step", "bad_json"])
def test_cache_refuses_missing_timing_and_same_batch_can_retry(tmp_path, damage):
    from netlab.run_record import run_record_path

    scenario = tmp_path / "scenario.yml"
    scenario.write_text(SCENARIO)
    path = scenario.with_suffix(".results.json")
    assert run_simulation(scenario).success
    record = run_record_path(path)
    if damage == "missing_file":
        record.unlink()
    elif damage == "missing_step":
        data = json.loads(record.read_text())
        data["step_durations_sec"].clear()
        record.write_text(json.dumps(data))
    else:
        record.write_text("{")
    with TaskQueue() as queue:
        batch = SimulationBatch(queue)
        with pytest.raises(ValueError, match="run record|Run record"):
            batch.submit(scenario)
        assert batch.submit(scenario, force=True).result().success


@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_publication_failure_removes_both_artifacts_and_retry_succeeds(
    tmp_path, monkeypatch, fail_at
):
    import netlab.artifacts as artifacts
    import netlab.simulation as simulation
    from netlab.run_record import run_record_path

    scenario = tmp_path / "scenario.yml"
    scenario.write_text(SCENARIO)
    path = scenario.with_suffix(".results.json")
    original = artifacts.write_json_atomic
    calls = 0

    def fail_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("publication fault")
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation, "write_json_atomic", fail_write)
    monkeypatch.setattr(artifacts, "write_json_atomic", fail_write)
    with TaskQueue() as queue:
        job = SimulationBatch(queue).submit(scenario)
        with pytest.raises(OSError, match="publication fault"):
            job.result()
        for item in (
            path,
            run_record_path(path),
            cache_path(path),
            cache_path(run_record_path(path)),
        ):
            assert not item.exists()
        assert job.result().success


def test_sidecar_paths_cannot_overlap_another_job(tmp_path):
    scenario = tmp_path / "scenario.yml"
    scenario.write_text(SCENARIO)
    with TaskQueue() as queue:
        batch = SimulationBatch(queue)
        first = batch.submit(scenario, results_path=tmp_path / "output.json")
        with pytest.raises(ValueError, match="Duplicate simulation output"):
            batch.submit(scenario, results_path=tmp_path / "output.run.json")
        assert first.result().success


def test_run_record_owns_its_measurement_mapping():
    outcome = simulate(SCENARIO)
    assert outcome.success
    original = outcome.step_durations_sec["msd_baseline"]
    record = outcome.run_record()
    record["step_durations_sec"]["msd_baseline"] = original + 1
    assert outcome.step_durations_sec["msd_baseline"] == original
