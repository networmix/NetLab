"""Native analysis contracts for exact MSD inputs and grouped demand labels."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dev.regenerate_fixtures import normalize_fixture_record
from netlab.artifacts import cache_matches, package_provenance, record_cache
from netlab.metrics.analysis import analyze_one_seed
from netlab.metrics.bac import compute_bac
from netlab.metrics.matrixdump import compute_pair_matrices
from netlab.metrics.sps import compute_sps
from netlab.results import (
    NO_FEASIBLE_ALPHA,
    baseline_case,
    delivery_ratio,
    require_step,
    rows,
)
from netlab.run_record import run_record_path
from netlab.simulation import run_simulation, simulate


def mini_document():
    path = Path(__file__).parents[1] / "data/mini_dcbb.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["workflow"] = [
        s for s in doc["workflow"] if s["name"] in {"msd_baseline", "tm_1x_bb"}
    ]
    doc["workflow"][1]["name"] = "tm_placement"
    return doc


@pytest.fixture(scope="module")
def mini_run():
    outcome = simulate(yaml.safe_dump(mini_document()))
    assert outcome.success, outcome.error
    return outcome


def test_fixed_alpha_is_checked_against_exact_msd_value(tmp_path):
    for alpha in (1, 3):
        doc = mini_document()
        doc["workflow"][1].pop("alpha_from_step")
        doc["workflow"][1]["alpha"] = alpha
        outcome = simulate(yaml.safe_dump(doc))
        assert outcome.success, outcome.error
        if alpha == 1:
            with pytest.raises(ValueError, match="placement alpha.*exact MSD"):
                analyze_one_seed(
                    outcome.results, tmp_path, False, run_record=outcome.run_record()
                )
        else:
            result = analyze_one_seed(
                outcome.results, tmp_path, False, run_record=outcome.run_record()
            )
            assert result[0].alpha_star == 3 and result[1].offered == 600


def test_declared_demand_mismatch_is_refused_before_rounding(mini_run, tmp_path):
    result = analyze_one_seed(
        mini_run.results, tmp_path, False, run_record=mini_run.run_record()
    )
    assert result[0].base_total_demand == 200
    changed = deepcopy(mini_run.results)
    require_step(changed, "msd_baseline")["question"]["demands"][0][
        "physical_volume"
    ] = 101
    with pytest.raises(ValueError, match="declared demands.*MSD"):
        analyze_one_seed(changed, tmp_path, False, run_record=mini_run.run_record())


def test_sub_ulp_alpha_difference_is_not_accepted_as_float_equality(mini_run, tmp_path):
    changed = deepcopy(mini_run.results)
    alpha = require_step(changed, "tm_placement")["result"]["alpha"]
    alpha.update(numerator=3 * 2**54 + 1, denominator=2**54, value=3.0)
    assert (
        alpha["value"]
        == require_step(changed, "msd_baseline")["result"]["alpha_feasible"]["value"]
    )
    with pytest.raises(ValueError, match="placement alpha.*exact MSD"):
        analyze_one_seed(changed, tmp_path, False, run_record=mini_run.run_record())


def combine_document(reverse=False, maxflow=False):
    doc = mini_document()
    doc["network"]["nodes"]["abc1/rsw2"] = {"attrs": {"role": "rsw", "site": "abc1"}}
    doc["network"]["links"] += [
        {
            "source": "abc1/rsw2",
            "target": f"abc1/fadu/{plane}",
            "capacity": 500,
            "cost": 1,
        }
        for plane in (0, 1)
    ]
    declaration = {
        "source": "^abc1/rsw2?$",
        "target": "^xyz1/rsw$",
        "volume": 100,
        "mode": "combine",
        "preset": "SHORTEST_PATHS_ECMP",
    }
    doc["demands"]["baseline_traffic_matrix"] = [declaration]
    if reverse:
        doc["demands"]["baseline_traffic_matrix"].append(
            dict(
                declaration, source=declaration["target"], target=declaration["source"]
            )
        )
    doc["workflow"][1].update(iterations=4, seed=42)
    doc["workflow"].append({"type": "NetworkStats", "name": "network_statistics"})
    if maxflow:
        doc["workflow"].append(
            {
                "type": "MaxFlow",
                "name": "node_to_node_capacity_matrix",
                "source": declaration["source"],
                "target": declaration["target"],
                "mode": "combine",
                "failure_policy": "fm_1x_bb",
                "iterations": 4,
                "seed": 42,
                "workers": 1,
            }
        )
    return doc


def test_grouped_bac_uses_declared_selector_labels():
    outcome = simulate(yaml.safe_dump(combine_document(reverse=True)))
    assert outcome.success, outcome.error
    bac = compute_bac(outcome.results, "tm_placement")
    assert set(bac.per_flow) == {"abc1/rsw2?>xyz1/rsw", "xyz1/rsw>abc1/rsw2?"}
    for part in bac.per_flow.values():
        assert part.offered == 200
        assert part.series.tolist() == [200, 100, 100, 100, 100]


def test_combine_metrics_command_keeps_one_group_row(tmp_path):
    source = tmp_path / "scenario.yml"
    source.write_text(yaml.safe_dump(combine_document()))
    path = tmp_path / "runs/combine__seed42_scenario.results.json"
    outcome = run_simulation(source, results_path=path)
    assert outcome.success, outcome.error
    analyze_one_seed(outcome.results, tmp_path, False, run_record=outcome.run_record())
    matrix, normalized, _, _ = compute_pair_matrices(outcome.results, False)
    assert matrix.index.tolist() == ["abc1/rsw2?→xyz1/rsw"]
    assert matrix.to_numpy().tolist() == [[100.0] * 5]
    assert normalized.to_numpy().tolist() == [[0.5] * 5]
    command = subprocess.run(
        [sys.executable, "-m", "netlab.cli", "metrics", str(path.parent), "--no-plots"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert command.returncode == 0, command.stdout + command.stderr
    report = tmp_path / "runs_metrics/combine/seed42"
    assert pd.read_csv(report / "pairs_tm_abs.csv").iloc[:, 0].tolist() == [
        "abc1/rsw2?→xyz1/rsw"
    ]
    assert json.loads((report / "bac.json").read_text())["auc_normalized"] == 3 / 5


def test_grouped_sps_keeps_one_aggregate_and_refuses_unrelated_capacity():
    outcome = simulate(yaml.safe_dump(combine_document(maxflow=True)))
    assert outcome.success, outcome.error
    sps = compute_sps(outcome.results)
    assert sps.series.tolist() == [0.75] * 4
    _, _, mf, norm = compute_pair_matrices(outcome.results, True)
    assert mf is not None and norm is not None
    assert mf.index.tolist() == ["abc1/rsw2?→xyz1/rsw"]
    assert mf.to_numpy().tolist() == [[150.0] * 5]
    assert norm.to_numpy().tolist() == [[0.75] * 5]
    changed = deepcopy(outcome.results)
    row = rows(baseline_case(require_step(changed, "node_to_node_capacity_matrix")))[0]
    row["endpoints"]["sources"] = ["abc1/rsw"]
    with pytest.raises(
        ValueError, match="MaxFlow baseline does not cover exact placement endpoints"
    ):
        compute_sps(changed)


def test_fixture_record_is_relative_and_retains_measurements(tmp_path):
    path = tmp_path / "build/run/scenario.results.json"
    record = run_record_path(path)
    record.parent.mkdir(parents=True)
    before = {
        "status": "success",
        "scenario": str(tmp_path / "tests/scenario.yml"),
        "step_durations_sec": {"tm_placement": 0.25},
        "duration_s": 1.5,
    }
    record.write_text(json.dumps(before))
    record_cache(record, "key")
    normalize_fixture_record(path, tmp_path)
    after = json.loads(record.read_text())
    assert after == dict(before, scenario="tests/scenario.yml")
    assert cache_matches(record, "key")
    before_bytes = record.read_bytes()
    after["scenario"] = str(tmp_path.parent / "outside.yml")
    record.write_text(json.dumps(after))
    invalid = record.read_bytes()
    with pytest.raises(ValueError):
        normalize_fixture_record(path, tmp_path)
    assert record.read_bytes() == invalid and invalid != before_bytes


def test_netlab_provenance_has_version_and_commit_without_local_url():
    record = package_provenance()["netlab"]
    assert record["version"] and "commit" in record and "dirty" in record
    if record["dirty"]:
        assert record["commit"] is None
    assert record["commit"] is None or len(record["commit"]) == 40
    assert "source" not in record and "file://" not in json.dumps(record)


def test_native_zero_request_has_ratio_one():
    outcome = simulate("""network:
  nodes: {A: {}, B: {}}
  links: [{source: A, target: B, capacity: 1}]
demands:
  tm: [{source: A, target: B, volume: 0}]
workflow:
  - {type: TrafficMatrixPlacement, name: tm_placement, demand_set: tm}
""")
    assert outcome.success, outcome.error
    case = baseline_case(require_step(outcome.results, "tm_placement"))
    assert case["states"]["converged"]["totals"]["delivery_ratio"] == {
        "numerator": 1,
        "denominator": 1,
    }
    assert delivery_ratio(case) == 1


def test_no_feasible_reason_is_the_same_with_or_without_dependent_placement():
    doc = mini_document()
    doc["network"]["nodes"]["island"] = {}
    doc["demands"]["baseline_traffic_matrix"] = [
        {
            "source": "^abc1/rsw$",
            "target": "^island$",
            "volume": 10,
            "mode": "combine",
            "preset": "TE_MAXFLOW_UNLIM",
        }
    ]
    errors = []
    for workflow in (doc["workflow"][:1], doc["workflow"]):
        case = deepcopy(doc)
        case["workflow"] = workflow
        outcome = simulate(yaml.safe_dump(case))
        assert outcome.status == "failed"
        errors.append(outcome.error)
    assert errors == [NO_FEASIBLE_ALPHA, NO_FEASIBLE_ALPHA]


def per_group_document(group_mode):
    doc = combine_document()
    for name in ("def1/rsw", "def1/rsw2"):
        doc["network"]["nodes"][name] = {"attrs": {"role": "rsw"}}
        doc["network"]["links"] += [
            {"source": name, "target": f"abc1/fadu/{p}", "capacity": 500, "cost": 1}
            for p in (0, 1)
        ]
    doc["demands"]["baseline_traffic_matrix"][0].update(
        source="^(abc1|def1)/rsw2?$", group_mode=group_mode
    )
    return doc


@pytest.mark.parametrize("group_mode", ["per_group", "group_pairwise"])
def test_combine_group_rows_keep_separate_series_and_pairs(group_mode, tmp_path):
    outcome = simulate(yaml.safe_dump(per_group_document(group_mode)))
    assert outcome.success, outcome.error
    analyze_one_seed(outcome.results, tmp_path, False, run_record=outcome.run_record())
    bac = compute_bac(outcome.results, "tm_placement")
    source = "(abc1|def1)/rsw2?"
    target = 'xyz1/rsw ["xyz1/rsw"]'
    abc = f'{source} ["abc1/rsw", "abc1/rsw2"]'
    def_ = f'{source} ["def1/rsw", "def1/rsw2"]'
    assert set(bac.per_flow) == {f"{abc}>{target}", f"{def_}>{target}"}
    assert bac.series.tolist() == [200, 100, 100, 100, 100]
    # Each group asks 100. The first group fills the surviving 100-capacity
    # shortest corridor; the second group receives none after a BB failure.
    assert bac.per_flow[f"{abc}>{target}"].series.tolist() == [100] * 5
    assert bac.per_flow[f"{def_}>{target}"].series.tolist() == [100, 0, 0, 0, 0]
    assert all(flow.offered == 100 for flow in bac.per_flow.values())
    absolute, normalized, _, _ = compute_pair_matrices(outcome.results, False)
    assert set(absolute.index) == {f"{abc}→{target}", f"{def_}→{target}"}
    assert absolute.loc[f"{abc}→{target}"].tolist() == [100] * 5
    assert absolute.loc[f"{def_}→{target}"].tolist() == [0] * 5
    assert normalized.loc[f"{abc}→{target}"].tolist() == [1] * 5
    assert normalized.loc[f"{def_}→{target}"].tolist() == [0] * 5


def test_matching_capacity_labels_still_require_matching_physical_endpoints():
    from netlab.metrics.common import baseline_demand_map, require_capacity_pairs
    from netlab.results import pair_totals

    outcome = simulate(yaml.safe_dump(combine_document(maxflow=True)))
    assert outcome.success, outcome.error
    require_capacity_pairs(outcome.results)
    changed = deepcopy(outcome.results)
    capacity = require_step(changed, "node_to_node_capacity_matrix")
    base = baseline_case(capacity)
    # Retain a grouped row and the same declared selectors, but report a
    # capacity for a different complete source roster.
    rows(base)[0]["endpoints"]["sources"] = ["abc1/rsw", "xyz1/rsw"]
    assert set(baseline_demand_map(changed)) == set(
        pair_totals(changed, capacity, base, "delivered_units")
    )
    with pytest.raises(
        ValueError, match="MaxFlow baseline does not cover exact placement endpoints"
    ):
        require_capacity_pairs(changed)


@pytest.mark.parametrize(
    "status",
    ["", " M tracked.py\n", "M  tracked.py\n", "?? new.py\n", " M .claude/skills\n"],
)
def test_netlab_git_provenance_names_only_a_clean_tree(status, tmp_path, monkeypatch):
    from netlab import artifacts

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(artifacts, "__file__", str(tmp_path / "netlab/artifacts.py"))
    monkeypatch.setattr(
        artifacts,
        "package_versions",
        lambda: {
            "netlab": {
                "version": "0.5.0",
                "content_sha256": "content",
                "source": {"vcs_info": {"commit_id": "b" * 40}},
            }
        },
    )

    def git(command, **options):
        assert options["cwd"] == tmp_path and options["check"]
        if "status" in command:
            assert "--untracked-files=normal" in command
            assert "--ignore-submodules=none" in command
            return subprocess.CompletedProcess(command, 0, stdout=status)
        assert command == ["git", "rev-parse", "HEAD"] and not status
        return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n")

    monkeypatch.setattr(artifacts.subprocess, "run", git)
    record = artifacts.package_provenance()["netlab"]
    assert record == {
        "version": "0.5.0",
        "content_sha256": "content",
        "commit": None if status else "a" * 40,
        "dirty": bool(status),
    }


def test_regenerator_refuses_dirty_netlab_before_creating_outputs(
    tmp_path, monkeypatch
):
    from dev import regenerate_fixtures as regenerate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        regenerate, "__file__", str(tmp_path / "dev/regenerate_fixtures.py")
    )
    monkeypatch.setattr(
        regenerate,
        "package_provenance",
        lambda: {"netlab": {"dirty": True, "commit": None}},
    )
    with pytest.raises(SystemExit, match="NetLab has uncommitted changes"):
        regenerate.main()
    assert not (tmp_path / "build").exists()
    monkeypatch.setattr(
        regenerate,
        "package_provenance",
        lambda: {"netlab": {"dirty": False, "commit": "a" * 40}},
    )
    (tmp_path / "topogen_configs_small").mkdir()
    (tmp_path / "topogen_configs_small/control.yml").write_text("{}\n")
    monkeypatch.setattr(
        regenerate, "source_provenance", lambda _: {"version": "released"}
    )

    def pipeline(config):
        assert config.output_dir.parent.is_dir()
        raise RuntimeError("clean tree reached generation")

    monkeypatch.setattr(regenerate, "run_pipeline", pipeline)
    with pytest.raises(RuntimeError, match="clean tree reached generation"):
        regenerate.main()


def test_group_pairwise_keeps_both_source_and_target_group_keys():
    sources = [f"{group}/rsw{i}" for group in ("a", "b") for i in (0, 1)]
    targets = [f"{group}/rsw{i}" for group in ("c", "d") for i in (0, 1)]
    doc = {
        "network": {
            "nodes": {name: {} for name in sources + targets},
            "links": [
                {"source": s, "target": t, "capacity": 100, "cost": 1}
                for s in sources
                for t in targets
            ],
        },
        "demands": {
            "tm": [
                {
                    "source": "^(a|b)/rsw[01]$",
                    "target": "^(c|d)/rsw[01]$",
                    "volume": 40,
                    "mode": "combine",
                    "group_mode": "group_pairwise",
                }
            ]
        },
        "workflow": [
            {
                "type": "TrafficMatrixPlacement",
                "name": "tm_placement",
                "demand_set": "tm",
            }
        ],
    }
    outcome = simulate(yaml.safe_dump(doc))
    assert outcome.success, outcome.error
    bac = compute_bac(outcome.results, "tm_placement")
    # Four disjoint group pairs share the declaration's 40 equally.
    expected = {
        f'(a|b)/rsw[01] ["{s}/rsw0", "{s}/rsw1"]>(c|d)/rsw[01] ["{t}/rsw0", "{t}/rsw1"]'
        for s in ("a", "b")
        for t in ("c", "d")
    }
    assert set(bac.per_flow) == expected
    assert bac.series.tolist() == [40]
    assert all(flow.series.tolist() == [10] for flow in bac.per_flow.values())
