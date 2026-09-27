"""Rebuild simulation fixtures from cached geography and the installed tools.

Run with the local-source environment used by check_ngraph_integration.sh:
    python dev/regenerate_fixtures.py --topogen-commit <verified-commit>
All commands must succeed before any checked-in fixture is replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topogen-commit", required=True)
    args = parser.parse_args()
    ngraph = import_module("ngraph")
    topogen = import_module("topogen")
    root = Path(__file__).resolve().parents[1]
    topogen_file, ngraph_file = topogen.__file__, ngraph.__file__
    assert topogen_file is not None and ngraph_file is not None
    topogen_root = Path(topogen_file).resolve().parents[1]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(topogen_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.topogen_commit:
        parser.error(f"TopoGen is at {actual_commit}, expected {args.topogen_commit}")
    if subprocess.check_output(
        ["git", "-C", str(topogen_root), "status", "--porcelain"]
    ):
        parser.error("TopoGen source checkout must be clean")
    build = root / "build/fixture-regeneration"
    build.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=build))
    print(f"Regeneration artifacts: {work}", flush=True)
    scenarios = work / "scenarios"
    for graph in (root / "tests/data/scenarios").rglob("*_integrated_graph.json"):
        target = scenarios / graph.relative_to(root / "tests/data/scenarios")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(graph, target)

    def run(module: str, *command: str) -> None:
        subprocess.run([sys.executable, "-m", module, *command], cwd=root, check=True)

    run(
        "netlab.cli",
        "run",
        "topogen_configs_small",
        "--seeds",
        "11",
        "12",
        "--scenarios-dir",
        str(scenarios),
        "--force-run",
        "--build-jobs",
        "1",
        "--run-jobs",
        "1",
    )
    run("netlab.cli", "metrics", str(scenarios), "--no-plots")
    run(
        "ngraph",
        "run",
        "tests/autoresearch/data/square_mesh.yaml",
        "-o",
        str(work / "square"),
    )
    run("ngraph", "run", "tests/data/mini_dcbb.yaml", "-o", str(work / "mini"))

    # Save input fixtures alongside generated output for review.
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
    provenance = {
        "python": sys.version,
        "topogen_commit": actual_commit,
        "topogen_source": str(topogen_root),
        "ngraph_source": str(Path(ngraph_file).resolve()),
    }
    ngraph_root = Path(ngraph_file).resolve().parents[1]
    provenance["ngraph_head"] = subprocess.check_output(
        ["git", "-C", str(ngraph_root), "rev-parse", "HEAD"], text=True
    ).strip()
    patch = subprocess.check_output(
        ["git", "-C", str(ngraph_root), "diff", "HEAD", "--binary"]
    )
    (work / "ngraph.patch").write_bytes(patch)
    provenance["ngraph_diff_sha256"] = hashlib.sha256(patch).hexdigest()
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
