"""Rebuild simulation fixtures from cached geography and the installed tools.

Run from an environment installed with the dependencies declared in
pyproject.toml, for example ``venv/bin/python dev/regenerate_fixtures.py``.
Every source distribution must be a released version or a clean Git revision so
the fixtures stay reproducible. All commands must succeed before any checked-in
fixture is replaced; the recorded provenance is printed at the end.
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
    provenance = {
        "python": sys.version.split()[0],
        "sources": {name: source_provenance(name) for name in SOURCES},
    }
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
