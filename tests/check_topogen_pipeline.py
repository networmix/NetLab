"""Local-source gate: NetLab CLI -> TopoGen build -> NetGraph run -> metrics.

Run explicitly from check_ngraph_integration.sh. The geographic input is a tiny
pre-generated graph, so this verifies the build/simulation boundary without
requiring Census datasets or invoking external research backends.
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
from topogen.config import TopologyConfig
from topogen.integrated_graph import save_to_json


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
            step["parallelism"] = 1
            step["failure_policy"] = policy
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/workflows.yml").write_text(yaml.safe_dump(workflows))
    shutil.copy2(
        root / "lib/failure_policies.yml", tmp_path / "lib/failure_policies.yml"
    )
    graph = nx.Graph()
    coords = [(0.0, 0.0), (100000.0, 0.0), (50000.0, 100000.0)]
    names = ["new-york-jersey-city-newark", "columbus", "washington-arlington"]
    for idx, coord in enumerate(coords):
        graph.add_node(
            coord,
            node_type="metro",
            name=names[idx],
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
            risk_groups=[f"corridor_risk_{a}_{b}"],
        )
    scenarios = tmp_path / "scenarios"
    generated = scenarios / "tiny/tiny"
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
    assert results["workflow"]["tm_placement"]["scenario_seed"] == 7
    assert results["workflow"]["tm_placement"]["seed_source"] != "explicit-step"
    assert results["steps"]["msd_baseline"]["data"]["alpha_star"] > 0
    assert len(results["steps"]["tm_placement"]["data"]["baseline"]["flows"]) == 6
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
