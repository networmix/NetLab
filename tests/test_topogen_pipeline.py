"""NetLab CLI -> TopoGen build -> NetGraph run -> metrics.

The geographic input is a tiny pre-generated graph, so this exercises the
build/simulation boundary without Census datasets or research backends.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import yaml
from netlab_topogen.config import TopologyConfig
from netlab_topogen.corridors import corridor_risk_name
from netlab_topogen.integrated_graph import save_to_json

from netlab.results import baseline_case, feasible_alpha, require_step, rows

pytestmark = [pytest.mark.slow, pytest.mark.timeout(300)]


@pytest.mark.parametrize(
    "template, policy",
    [
        ("small_baseline", "mc_baseline"),
        ("small_clos", "mc_baseline"),
        ("small_dragonfly", "mc_baseline"),
        ("small_dragonfly_custom", "mc_baseline"),
        ("small_baseline", "mc_hard"),
    ],
)
def test_topogen_build_run_metrics(tmp_path: Path, template: str, policy: str):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / f"topogen_configs_small/{template}.yml").read_text()
    )
    config["visualization"] = {}
    config["failure_policies"]["assignments"]["default"] = policy
    master = tmp_path / "tiny.yml"
    master.write_text(yaml.safe_dump(config))
    workflows = yaml.safe_load((root / "lib/workflows.yml").read_text())
    for step in workflows["design_analysis_brief"]:
        if "iterations" in step:
            step["iterations"] = 10
            step["workers"] = 1
            step["failure_policy"] = policy
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/workflows.yml").write_text(yaml.safe_dump(workflows))
    shutil.copy2(
        root / "lib/failure_policies.yml", tmp_path / "lib/failure_policies.yml"
    )
    graph = nx.MultiGraph()
    coords = [(0.0, 0.0), (100000.0, 0.0), (50000.0, 100000.0)]
    names = ["new-york-jersey-city-newark", "columbus", "washington-arlington"]
    for idx, coord in enumerate(coords):
        graph.add_node(
            coord,
            node_type="metro",
            name=names[idx],
            name_orig=names[idx],
            metro_id=f"metro_{idx}",
            x=coord[0],
            y=coord[1],
            radius_km=10.0,
        )
    for a, b in ((0, 1), (1, 2), (0, 2)):
        graph.add_edge(
            coords[a],
            coords[b],
            length_km=100.0,
            capacity=400,
            edge_type="corridor",
            geometry=[coords[a], coords[b]],
            risk_groups=[corridor_risk_name("corridor_risk", names[a], names[b], 0)],
        )
    scenarios = tmp_path / "scenarios"
    generated = tmp_path / "graphs"
    generated.mkdir(parents=True)
    save_to_json(
        graph,
        generated / "tiny_integrated_graph.json",
        "EPSG:5070",
        TopologyConfig().output.formatting,
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "netlab.cli",
            "run",
            str(master),
            "--seeds",
            "7",
            "--scenarios-dir",
            str(scenarios),
            "--graphs-dir",
            str(generated),
            "--build-jobs",
            "1",
            "--run-jobs",
            "1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    logs = "\n".join(f"{p.name}:\n{p.read_text()}" for p in scenarios.rglob("*.log"))
    assert run.returncode == 0, run.stdout + run.stderr + logs
    output = scenarios / "tiny/tiny__seed7/tiny__seed7_scenario.results.json"
    assert output.is_file(), logs
    results = json.loads(output.read_text())
    assert require_step(results, "tm_placement")["scenario_seed"] == 7
    assert require_step(results, "tm_placement")["seed_source"] != "explicit-step"
    assert feasible_alpha(require_step(results, "msd_baseline")) > 0
    assert len(rows(baseline_case(require_step(results, "tm_placement")))) == 6
    run = subprocess.run(
        [sys.executable, "-m", "netlab.cli", "metrics", str(scenarios), "--no-plots"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    summary = pd.read_csv(tmp_path / "scenarios_metrics/project.csv")
    assert summary["scenario"].tolist() == ["tiny"]


def test_missing_graph_invalidates_all_planned_seed_artifacts(tmp_path):
    from netlab.pipeline import PipelineConfig, run_pipeline

    master = tmp_path / "tiny.yml"
    master.write_text("{}")
    output = tmp_path / "out"
    scenario = output / "tiny/tiny__seed1/tiny__seed1_scenario.yml"
    scenario.parent.mkdir(parents=True)
    scenario.write_text("stale input")
    result = scenario.with_suffix(".results.json")
    result.write_text('{"stale": true}')
    outcome = run_pipeline(
        PipelineConfig([master], [1], output, graphs_dir=tmp_path / "missing")
    )
    assert not outcome.success
    assert "tiny/graph" in outcome.errors
    assert not result.exists()
    assert not scenario.exists()


@pytest.mark.parametrize("invalid", [True, False])
def test_topogen_validates_before_publishing(tmp_path, monkeypatch, invalid):
    from types import SimpleNamespace

    import netlab.topology as topology

    config_path = (
        Path(__file__).resolve().parents[1] / "topogen_configs_small/small_baseline.yml"
    )
    config = TopologyConfig.from_yaml(config_path)
    build = SimpleNamespace(yaml="seed: 7\n")
    validation = SimpleNamespace(
        issues=["bad scenario"] if invalid else [], prepared=object()
    )
    calls = []
    monkeypatch.setattr(
        topology.TopologyConfig, "from_yaml", staticmethod(lambda _: config)
    )
    monkeypatch.setattr(
        topology, "load_from_json", lambda _: (object(), config.projection.target_crs)
    )
    monkeypatch.setattr(
        topology,
        "load_library",
        lambda directory: calls.append(("library", directory)) or object(),
    )
    monkeypatch.setattr(
        topology,
        "build_scenario",
        lambda graph, cfg, library: calls.append(("build", cfg)) or build,
    )
    monkeypatch.setattr(
        topology,
        "validate_scenario",
        lambda text, components: calls.append(("validate", text, components))
        or validation,
    )

    def publish(path, text):
        calls.append(("scenario", path))
        path.write_text(text)

    monkeypatch.setattr(topology, "write_scenario", publish)
    monkeypatch.setattr(
        topology,
        "write_artifacts",
        lambda b, v, c, context: calls.append(("artifacts", b, v, c, context)),
    )
    scenario = tmp_path / "scenario.yml"
    scenario.write_text("previous")
    if invalid:
        with pytest.raises(
            ValueError, match="Scenario validation failed:\nbad scenario"
        ):
            topology.build_seed_scenario(
                config_path, tmp_path / "graph.json", scenario, 7
            )
        assert scenario.read_text() == "previous"
        assert [call[0] for call in calls] == ["library", "build", "validate"]
    else:
        assert (
            topology.build_seed_scenario(
                config_path, tmp_path / "graph.json", scenario, 7
            )
            == scenario
        )
        assert scenario.read_text() == build.yaml
        assert [call[0] for call in calls] == [
            "library",
            "build",
            "validate",
            "scenario",
            "artifacts",
        ]
        assert calls[-1][1:4] == (build, validation, config)
        assert calls[-1][4].output_dir == scenario.parent
    assert config.output.scenario_seed == 7
