"""Exercise NetLab metrics against results produced by the installed NetGraph."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from netlab.metrics.analysis import analyze_one_seed
from netlab.results import require_step
from netlab.simulation import run_simulation


@pytest.fixture(scope="module")
def current_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("current-ngraph")
    scenario = yaml.safe_load(
        (Path(__file__).parent / "data" / "mini_dcbb.yaml").read_text()
    )
    scenario["workflow"] = scenario["workflow"][:2]
    scenario["workflow"][1]["name"] = "tm_placement"
    scenario["workflow"].append({"type": "NetworkStats", "name": "network_statistics"})
    scenario_path = root / "mini.yml"
    scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False))
    output = root / "scenarios" / "mini" / "mini__seed42"
    inspection = subprocess.run(
        [sys.executable, "-m", "ngraph", "inspect", str(scenario_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert inspection.returncode == 0, inspection.stdout + inspection.stderr
    outcome = run_simulation(
        scenario_path, results_path=output / "mini.results.json", timeout=30
    )
    assert outcome.success, outcome.error
    return outcome.results, root / "scenarios", outcome.run_record()


def test_analyze_fresh_results(current_results, tmp_path, monkeypatch):
    results, _, run_record = current_results
    results = copy.deepcopy(results)
    monkeypatch.delenv("NGRAPH_ENABLE_MAXFLOW", raising=False)
    alpha, bac, _, latency, _, _, _ = analyze_one_seed(
        results, tmp_path, False, run_record=current_results[2]
    )
    assert alpha.alpha_star == pytest.approx(3.0)
    assert alpha.base_total_demand == pytest.approx(200.0)
    assert bac.offered == pytest.approx(600.0)
    assert len(bac.series) == 11
    assert bac.auc_normalized == pytest.approx(6 / 11)
    assert len(bac.per_flow) == 2
    assert latency.baseline["WES"] == pytest.approx(100 * (212 / 112 - 1) / 300)


def test_metrics_cli_on_fresh_results(current_results):
    _, root, _ = current_results
    result = subprocess.run(
        [sys.executable, "-m", "netlab.cli", "metrics", str(root), "--no-plots"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metrics_root = root.with_name("scenarios_metrics")
    alpha = json.loads((metrics_root / "mini/seed42/alpha.json").read_text())
    bac = json.loads((metrics_root / "mini/seed42/bac.json").read_text())
    assert alpha["alpha_star"] == pytest.approx(3.0)
    assert bac["auc_normalized"] == pytest.approx(6 / 11)
    summary = pd.read_csv(metrics_root / "project.csv")
    assert summary["scenario"].tolist() == ["mini"]
    stats = pd.read_csv(metrics_root / "mini/network_stats_summary.csv")
    assert stats["node_count"].tolist() == [10]
    assert stats["link_count"].tolist() == [12]


@pytest.mark.parametrize(
    "volume, message",
    [
        (0, "zero-demand"),
        (-1, "MSD physical_volume must be a finite nonnegative number"),
        (float("nan"), "MSD physical_volume must be a finite nonnegative number"),
    ],
)
def test_current_demands_still_validated(current_results, tmp_path, volume, message):
    results = copy.deepcopy(current_results[0])
    require_step(results, "msd_baseline")["question"]["demands"][0][
        "physical_volume"
    ] = volume
    with pytest.raises(ValueError, match=message):
        analyze_one_seed(results, tmp_path, False, run_record=current_results[2])


def test_rounded_expansion_accounting_must_match(current_results, tmp_path):
    results = copy.deepcopy(current_results[0])
    require_step(results, "tm_placement")["result"]["accounting"][0][
        "resolved_units"
    ] += 1
    with pytest.raises(ValueError, match="expansion accounting"):
        analyze_one_seed(results, tmp_path, False, run_record=current_results[2])


@pytest.mark.parametrize("field", ["source", "target", "physical_volume"])
def test_current_demand_fields_required(current_results, tmp_path, field):
    results = copy.deepcopy(current_results[0])
    del require_step(results, "msd_baseline")["question"]["demands"][0][field]
    with pytest.raises(
        ValueError, match="source/target selector|MSD physical_volume must"
    ):
        analyze_one_seed(results, tmp_path, False, run_record=current_results[2])


def test_pairwise_maxflow_pipeline(tmp_path, monkeypatch):
    scenario = yaml.safe_load(
        (Path(__file__).parent / "autoresearch/data/square_mesh.yaml").read_text()
    )
    for step in scenario["workflow"]:
        if "iterations" in step:
            step["iterations"] = 10
            step["workers"] = 1
    scenario["workflow"].append({"type": "NetworkStats", "name": "network_statistics"})
    scenario_path = tmp_path / "square.yml"
    scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False))
    root = tmp_path / "scenarios"
    output = root / "square" / "square__seed42"
    outcome = run_simulation(
        scenario_path, results_path=output / "square.results.json", timeout=30
    )
    assert outcome.success, outcome.error
    results = outcome.results
    alpha, bac, maxflow, _, _, _, sps = analyze_one_seed(
        results, tmp_path, False, enable_maxflow=True, run_record=outcome.run_record()
    )
    assert alpha.alpha_star == pytest.approx(1.0)
    assert alpha.base_total_demand == pytest.approx(12.0)
    assert len(bac.per_flow) == 12
    assert len(bac.series) == 11
    assert maxflow is not None and len(maxflow.series) == 11
    assert sps is not None and sps.series.tolist() == pytest.approx([1.0] * 10)
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "netlab.cli",
            "metrics",
            str(root),
            "--no-plots",
            "--enable-maxflow",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    metrics = tmp_path / "scenarios_metrics/square/seed42"
    assert (metrics / "sps.json").is_file()
    assert not pd.read_csv(metrics / "pairs_tm_norm.csv").empty
