"""Rebuild fixtures from cached geography and the installed dependencies.

Run: venv/bin/python dev/regenerate_fixtures.py
Require released packages or clean Git revisions. Replace checked-in fixtures
only after all simulations and analyses succeed; print the provenance path.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from netlab.metrics.batch import run_metrics
from netlab.pipeline import PipelineConfig, discover_configs, run_pipeline
from netlab.simulation import run_simulation

SOURCES = ("ngraph", "netgraph-core", "topogen")


def source_provenance(name: str) -> dict:
    dist = importlib.metadata.distribution(name)
    info: dict = {"version": dist.version}
    raw = dist.read_text("direct_url.json")
    if raw is None:
        return info
    direct = json.loads(raw)
    info["url"] = direct["url"]
    if "vcs_info" in direct:
        info["commit"] = direct["vcs_info"]["commit_id"]
    elif direct.get("dir_info", {}).get("editable"):
        root = direct["url"].removeprefix("file://")
        info["commit"] = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True
        ).strip()
        if subprocess.check_output(["git", "-C", root, "status", "--porcelain"]):
            sys.exit(
                f"{name}: {root} has uncommitted changes; fixtures must be "
                "reproducible from a committed revision."
            )
    return info


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    provenance = {
        "python": sys.version.split()[0],
        "sources": {name: source_provenance(name) for name in SOURCES},
    }
    build = root / "build/fixture-regeneration"
    build.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=build))
    print(f"Regeneration artifacts: {work}", flush=True)
    scenarios = work / "scenarios"
    graphs = work / "graphs"
    graphs.mkdir()
    for graph in (root / "tests/data/scenarios").rglob("*_integrated_graph.json"):
        shutil.copy2(graph, graphs / graph.name)
    outcome = run_pipeline(
        PipelineConfig(
            masters=discover_configs(root / "topogen_configs_small"),
            seeds=[11, 12],
            output_dir=scenarios,
            graphs_dir=graphs,
            build_jobs=2,
            run_jobs=2,
            force=True,
        )
    )
    if not outcome.success:
        raise RuntimeError(outcome.errors)
    run_metrics(scenarios, no_plots=True)
    for source, output in (
        (
            root / "tests/autoresearch/data/square_mesh.yaml",
            work / "square/square_mesh.results.json",
        ),
        (root / "tests/data/mini_dcbb.yaml", work / "mini/mini_dcbb.results.json"),
    ):
        result = run_simulation(source, results_path=output, force=True)
        if not result.success:
            raise RuntimeError(result.error)

    # Save the previous fixtures alongside the generated output for review.
    shutil.copytree(root / "tests/data/scenarios_metrics", work / "inputs/metrics")
    shutil.copytree(root / "tests/data/scenarios", work / "inputs/scenarios")
    fixture_root = root / "tests/data/scenarios"
    for path in fixture_root.rglob("*"):
        if path.is_file() and not path.name.endswith("_integrated_graph.json"):
            path.unlink()
    for path in scenarios.rglob("*"):
        if path.is_file() and (
            path.name.endswith("_scenario.yml") or path.name.endswith(".results.json")
        ):
            target = fixture_root / path.relative_to(scenarios)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    shutil.rmtree(root / "tests/data/scenarios_metrics")
    shutil.copytree(
        work / "scenarios_metrics",
        root / "tests/data/scenarios_metrics",
        ignore=shutil.ignore_patterns("summary.txt"),
    )
    shutil.copy2(
        work / "square/square_mesh.results.json",
        root / "tests/autoresearch/data/square_mesh_results.json",
    )
    shutil.copy2(
        work / "mini/mini_dcbb.results.json",
        root / "tests/data/mini_dcbb_output/mini_dcbb.results.json",
    )
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
