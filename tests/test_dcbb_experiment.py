"""Keep the experiment workflow, seed summary and comparison consistent."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml
from ngraph.scenario import Scenario

from netlab.simulation import simulate
from tests.result_samples import exported, msd

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "dc-bb-interconnect"
TOPOLOGY = "dc4x9_bb4x4_bb4x4_dc4x9_one_to_one"


@pytest.fixture
def experiment_module(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENT))
    spec = importlib.util.spec_from_file_location(
        "dcbb_experiment_runner", EXPERIMENT / "run.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_result(runner, scenario, seed, results):
    directory = runner.results_dir / scenario / f"{scenario}__seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scenario}__seed{seed}_scenario.results.json"
    path.write_text(json.dumps(results))
    return path


def read_summary(runner, scenario):
    return json.loads((runner.results_dir / scenario / "summary.json").read_text())


def test_included_workflow_preserves_msd_in_summary_and_comparison(
    experiment_module, tmp_path, capsys
):
    runner = experiment_module.DcBbExperimentRunner(EXPERIMENT, results_dir=tmp_path)
    document = runner.merger.merge(
        EXPERIMENT / "topologies" / TOPOLOGY / "scenario.yml", seed=42
    )
    for step in document["workflow"]:
        if "iterations" in step:
            step.update(iterations=1, workers=1)
    outcome = simulate(yaml.safe_dump(document, sort_keys=False))
    assert outcome.success, outcome.error
    msd_name = next(
        step["name"]
        for step in document["workflow"]
        if step["type"] == "MaximumSupportedDemand"
    )
    from netlab.results import feasible_alpha, require_step

    alpha = feasible_alpha(require_step(outcome.results, msd_name))
    assert alpha > 0
    save_result(runner, TOPOLOGY, 42, outcome.results)
    runner.compute_metrics([TOPOLOGY])
    assert read_summary(runner, TOPOLOGY)["alpha_star"] == alpha
    runner.generate_comparison([TOPOLOGY])
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["data"][TOPOLOGY]["alpha_star"] == alpha
    assert f"{alpha:.2f}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "alphas, expected", [([1, 3, 9], 3), ([0, 0, 9], 0), ([0], 0), ([None], None)]
)
def test_summary_aggregates_seed_alphas_and_keeps_zero(
    experiment_module, tmp_path, alphas, expected
):
    runner = experiment_module.DcBbExperimentRunner(EXPERIMENT, results_dir=tmp_path)
    for seed, alpha in enumerate(alphas, 1):
        result = (
            exported(msd(alpha), seed=seed)
            if alpha is not None
            else exported(seed=seed)
        )
        save_result(runner, "example", seed, result)
    runner.compute_metrics(["example"])
    summary = read_summary(runner, "example")
    assert summary["alpha_star"] == expected
    assert summary["alpha_star_by_seed"] == {
        str(seed): alpha for seed, alpha in enumerate(alphas, 1)
    }


def test_two_to_one_name_matches_expanded_backbone(experiment_module, tmp_path):
    runner = experiment_module.DcBbExperimentRunner(EXPERIMENT, results_dir=tmp_path)
    name = next(n for n in runner.discover_scenarios() if n.endswith("_two_to_one"))
    dimensions = [
        tuple(map(int, shape)) for shape in re.findall(r"bb(\d+)x(\d+)", name)
    ]
    document = runner.merger.merge(EXPERIMENT / "topologies" / name / "scenario.yml")
    network = Scenario.from_yaml(yaml.safe_dump(document)).network
    for site, (planes, devices) in zip(["SiteA", "SiteB"], dimensions, strict=True):
        nodes = [n for n in network.nodes if n.startswith(f"{site}/BB-")]
        assert len(nodes) == planes * devices
        assert len({n.split("/")[1] for n in nodes}) == planes


@pytest.mark.parametrize("other_alpha", [None, 2])
def test_comparison_displays_zero_and_handles_zero_maximum(
    experiment_module, tmp_path, capsys, other_alpha
):
    runner = experiment_module.DcBbExperimentRunner(EXPERIMENT, results_dir=tmp_path)
    for scenario, alpha in [("zero", 0), ("other", other_alpha)]:
        directory = tmp_path / scenario
        directory.mkdir()
        (directory / "summary.json").write_text(
            json.dumps({"alpha_star": alpha, "network": {}})
        )
    runner.generate_comparison(["zero", "other"])
    row = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("│ zero ")
    ).split()
    assert row[3] == "0.00"
    assert row[4] == ("0.0%" if other_alpha else "?")


def test_summary_refuses_none_feasible(experiment_module, tmp_path):
    runner = experiment_module.DcBbExperimentRunner(EXPERIMENT, results_dir=tmp_path)
    save_result(
        runner, "example", 1, exported(msd(None, status="none_feasible"), seed=1)
    )
    with pytest.raises(ValueError, match="none_feasible"):
        runner.compute_metrics(["example"])
