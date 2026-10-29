"""Step measurements survive execution, reporting and cache reuse."""

import json
import os

import pandas as pd
import pytest
import yaml

from dev.verify_metrics import scenario_document
from netlab.metrics.batch import run_metrics
from netlab.run_record import load_run_record, run_record_path
from netlab.simulation import run_simulation


def measured_run(tmp_path):
    doc = scenario_document(1)
    doc["demands"]["traffic"][0]["preset"] = "SHORTEST_PATHS_ECMP"
    scenario = tmp_path / "scenario.yml"
    scenario.write_text(yaml.safe_dump(doc))
    root = tmp_path / "runs"
    result_path = root / "baseline__seed42_scenario.results.json"
    outcome = run_simulation(scenario, results_path=result_path)
    assert outcome.success, outcome.error
    return root, result_path, outcome


def test_real_step_timing_reaches_every_report_and_plot(tmp_path):
    root, result_path, outcome = measured_run(tmp_path)
    durations = outcome.step_durations_sec
    assert set(durations) == {step["name"] for step in outcome.results["steps"]}
    assert all(value > 0 for value in durations.values())
    total = durations["tm_placement"]
    per_iter = total / 2  # One failure draw plus one unsampled baseline reference.
    run_metrics(root, no_plots=False)
    report = tmp_path / "runs_metrics"
    seed = report / "baseline/seed42"
    metrics = json.loads((seed / "iterops.json").read_text())
    assert metrics["tm_duration_total_sec"] == total
    assert metrics["tm_duration_per_iter_sec"] == per_iter
    for relative in (
        "baseline/iterops_summary.csv",
        "project.csv",
    ):
        table = pd.read_csv(report / relative)
        assert table["tm_duration_total_sec"].tolist() == pytest.approx([total])
        assert table["tm_duration_per_iter_sec"].tolist() == pytest.approx([per_iter])
    assert (report / "baseline/IterationOps.png").stat().st_size > 0
    provenance = json.loads((report / "provenance.json").read_text())
    assert set(provenance["source_files"]) == {
        os.path.relpath(result_path),
        os.path.relpath(run_record_path(result_path)),
    }
    original = run_record_path(result_path).read_bytes()
    cached = run_simulation(tmp_path / "scenario.yml", results_path=result_path)
    assert cached.status == "cached"
    assert cached.step_durations_sec == durations
    assert run_record_path(result_path).read_bytes() == original
    run_metrics(root, no_plots=True)
    assert json.loads((seed / "iterops.json").read_text()) == metrics


@pytest.mark.parametrize("damage", ["missing_file", "missing_step"])
def test_missing_timing_refuses_metrics_and_preserves_previous_report(tmp_path, damage):
    root, path, outcome = measured_run(tmp_path)
    run_metrics(root, no_plots=True)
    report = tmp_path / "runs_metrics"
    before = {
        str(p.relative_to(report)): p.read_bytes()
        for p in report.rglob("*")
        if p.is_file()
    }
    record_path = run_record_path(path)
    if damage == "missing_file":
        record_path.unlink()
    else:
        record = json.loads(record_path.read_text())
        del record["step_durations_sec"]["msd_baseline"]
        record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="run record|Run record"):
        run_metrics(root, no_plots=True)
    assert before == {
        str(p.relative_to(report)): p.read_bytes()
        for p in report.rglob("*")
        if p.is_file()
    }
    assert outcome.success


def test_timing_record_mutation_during_analysis_refuses_publication(
    tmp_path, monkeypatch
):
    import netlab.metrics.batch as batch

    root, path, _ = measured_run(tmp_path)
    original = batch.analyze_one_seed

    def analyze_and_mutate(*args, **kwargs):
        value = original(*args, **kwargs)
        record = run_record_path(path)
        record.write_text(record.read_text() + "\n")
        return value

    monkeypatch.setattr(batch, "analyze_one_seed", analyze_and_mutate)
    with pytest.raises(ValueError, match="changed during metric analysis"):
        run_metrics(root, no_plots=True)
    assert not (tmp_path / "runs_metrics").exists()


def test_mini_dcbb_named_steps_have_measurements(tmp_path):
    from pathlib import Path

    scenario = Path(__file__).parents[1] / "data/mini_dcbb.yaml"
    path = tmp_path / "mini.results.json"
    outcome = run_simulation(scenario, results_path=path)
    assert outcome.success, outcome.error
    record = load_run_record(path, outcome.results)
    assert set(record["step_durations_sec"]) == {
        "msd_baseline",
        "tm_lh_path",
        "tm_1x_bb",
    }
    assert all(value > 0 for value in record["step_durations_sec"].values())
    cached = run_simulation(scenario, results_path=path)
    assert cached.status == "cached"
    assert cached.step_durations_sec == outcome.step_durations_sec
