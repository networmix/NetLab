"""Native results, validation mutations and artifact lifecycle experiments."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest

from dev.verify_metrics import (
    native,
    scenario_document,
    verify_exhaustive,
    verify_variants,
)
from netlab.metrics.analysis import analyze_one_seed
from netlab.metrics.batch import run_metrics
from netlab.metrics.costpower import compute_cost_power
from netlab.metrics.sps import compute_sps
from netlab.metrics.summary import build_baseline_normalized_table


@pytest.fixture(scope="module")
def native_results():
    return native(scenario_document(1))


def test_all_sixteen_native_failure_subsets(tmp_path):
    verify_exhaustive(tmp_path)
    verify_variants(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "placed",
        "dropped",
        "cost",
        "cost_volume",
        "summary",
        "count",
        "duration",
        "alpha",
        "maxflow_baseline",
    ],
)
def test_current_results_reject_corrupt_numbers(native_results, tmp_path, mutation):
    r = deepcopy(native_results)
    step = r["steps"]["tm_placement"]
    flow = step["data"]["flow_results"][0]["flows"][0]
    if mutation in ["placed", "dropped"]:
        flow[mutation] += 1
    elif mutation == "cost":
        flow["cost_distribution"] = {"NaN": flow["placed"]}
    elif mutation == "cost_volume":
        flow["cost_distribution"] = {"2": flow["placed"] + 1}
    elif mutation == "summary":
        step["data"]["baseline"]["summary"]["total_demand"] = math.inf
    elif mutation == "count":
        step["metadata"]["iterations"] += 1
    elif mutation == "duration":
        step["metadata"]["duration_sec"] = -1
    elif mutation == "alpha":
        r["steps"]["msd_baseline"]["data"]["alpha_star"] = math.nan
    elif mutation == "maxflow_baseline":
        r["steps"]["node_to_node_capacity_matrix"]["data"]["baseline"]["flows"][0][
            "placed"
        ] = -1
    with pytest.raises(ValueError):
        analyze_one_seed(r, tmp_path, False, enable_maxflow=True)


def test_maxflow_group_names_cannot_silently_become_outages(native_results):
    r = deepcopy(native_results)
    r["steps"]["node_to_node_capacity_matrix"]["data"]["baseline"]["flows"][0][
        "source"
    ] = "^a/dc/node$"
    with pytest.raises(ValueError, match="exact placement endpoints"):
        compute_sps(r)


def test_no_baseline_ratio_without_a_denominator(tmp_path):
    sdir = tmp_path / "baseline" / "seed1"
    sdir.mkdir(parents=True)
    (sdir / "bac.json").write_text(
        json.dumps({"auc_normalized": 0, "bw_at_probability_pct": {"99.0": 0}})
    )
    table = build_baseline_normalized_table(tmp_path)
    assert math.isnan(table.loc["baseline", "auc_norm_r"])
    assert math.isnan(table.loc["baseline", "bw_p99_pct_r"])
    assert math.isnan(table.loc["baseline", "lat_WES_delta_d"])
    assert math.isnan(table.loc["baseline", "node_count_r"])


def test_rerun_replaces_stale_seeds_and_optional_artifacts(native_results, tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    first = root / "baseline__seed1_scenario.results.json"
    second = root / "baseline__seed2_scenario.results.json"
    first.write_text(
        json.dumps(
            dict(native_results, scenario=dict(native_results["scenario"], seed=1))
        )
    )
    second.write_text(
        json.dumps(
            dict(native_results, scenario=dict(native_results["scenario"], seed=2))
        )
    )
    run_metrics(root, no_plots=True, enable_maxflow=True)
    out = tmp_path / "runs_metrics"
    assert (out / "baseline" / "seed2" / "bac.json").exists()
    assert (out / "baseline" / "seed1" / "sps.json").exists()
    second.unlink()
    run_metrics(root, no_plots=True, enable_maxflow=False)
    assert not (out / "baseline" / "seed2").exists()
    assert not (out / "baseline" / "seed1" / "sps.json").exists()
    assert not (out / "baseline" / "seed1" / "pairs_mf_abs.csv").exists()
    assert json.loads((out / "provenance.json").read_text())["seeds_analyzed"] == {
        "baseline": [1]
    }


def test_invalid_rerun_does_not_publish_partial_metrics(native_results, tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    first = root / "baseline__seed1_scenario.results.json"
    first.write_text(
        json.dumps(
            dict(native_results, scenario=dict(native_results["scenario"], seed=1))
        )
    )
    run_metrics(root, no_plots=True)
    out = tmp_path / "runs_metrics"
    before = {
        str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()
    }
    # A later seed fails after the first one was analyzed.
    bad = deepcopy(native_results)
    bad["scenario"]["seed"] = 2
    bad["steps"]["tm_placement"]["data"]["baseline"]["flows"][0]["placed"] = -1
    (root / "baseline__seed2_scenario.results.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        run_metrics(root, no_plots=True)
    after = {
        str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()
    }
    assert before == after
    assert not (out / "baseline" / "seed2").exists()


def test_cost_invalid_denominator_and_actual_zero():
    results = {
        "steps": {
            "cost_power": {
                "data": {"levels": {"0": [{"capex_total": 0, "power_total_watts": 0}]}}
            }
        }
    }
    assert compute_cost_power(results, 100, 100).per_offered_demand__usd_per_gbit == 0
    assert (
        compute_cost_power(results, math.inf, 0).per_offered_demand__usd_per_gbit
        is None
    )
    results["steps"]["cost_power"]["data"]["levels"]["0"][0]["capex_total"] = -1
    with pytest.raises(ValueError):
        compute_cost_power(results, 100, 100)


def test_multiseed_report_against_independent_reference_and_pattern_permutation(
    tmp_path,
):
    import pandas as pd

    from dev.verify_metrics import verify_multiseed

    root = verify_multiseed(tmp_path)
    before = pd.read_csv(tmp_path / "multiseed_metrics" / "project.csv")
    for path in root.glob("*.json"):
        result = json.loads(path.read_text())
        for name in ["tm_placement", "node_to_node_capacity_matrix"]:
            step = result["steps"][name]
            step["data"]["flow_results"].reverse()
            step["metadata"]["occurrence_counts"].reverse()
        path.write_text(json.dumps(result))
    run_metrics(root, no_plots=True, enable_maxflow=True)
    after = pd.read_csv(tmp_path / "multiseed_metrics" / "project.csv")
    pd.testing.assert_frame_equal(before, after)


def test_metrics_json_is_strict_and_seed_mismatch_is_rejected(native_results, tmp_path):
    from netlab.metrics.batch import group_by_scenario
    from netlab.metrics.common import write_metric_json

    path = tmp_path / "metrics.json"
    write_metric_json(
        path, {"missing": math.nan, "infinite": math.inf, "nested": [0, math.nan]}
    )

    def reject_constant(text):
        pytest.fail(f"Nonstandard JSON number: {text}")

    assert json.loads(path.read_text(), parse_constant=reject_constant) == {
        "missing": None,
        "infinite": None,
        "nested": [0, None],
    }
    wrong_seed = tmp_path / "baseline__seed999_scenario.results.json"
    wrong_seed.write_text(json.dumps(native_results))
    with pytest.raises(ValueError, match="seed disagrees"):
        group_by_scenario([wrong_seed])


@pytest.mark.parametrize("invalid", [-0.1, math.nan, math.inf, 1.1])
def test_failure_statistics_reject_invalid_ratios(invalid):
    from netlab.metrics_failure import compute_failure_stats

    with pytest.raises(ValueError):
        compute_failure_stats({"tm": [invalid]})


def test_native_without_path_details_reports_unavailable_latency(tmp_path):
    doc = scenario_document(1)
    doc["workflow"][1]["include_flow_details"] = False
    raw = native(doc)
    _, bac, _, latency, _, _, _ = analyze_one_seed(raw, tmp_path, True)
    assert bac.series.tolist() == [8, 3]
    assert latency.baseline["reference_coverage"] == 0
    assert latency.failures["reference_coverage"] == 0
    assert math.isnan(latency.failures["p99"])
    assert not (tmp_path / "latency.png").exists()


def test_changed_input_cannot_publish_metrics(native_results, tmp_path, monkeypatch):
    import netlab.metrics.batch as batch

    root = tmp_path / "runs"
    root.mkdir()
    path = root / "baseline__seed42_scenario.results.json"
    path.write_text(json.dumps(native_results))
    original = batch.analyze_one_seed

    def analyze_and_mutate(*args, **kwargs):
        output = original(*args, **kwargs)
        path.write_text(path.read_text() + "\n")
        return output

    monkeypatch.setattr(batch, "analyze_one_seed", analyze_and_mutate)
    with pytest.raises(ValueError, match="changed during metric analysis"):
        batch.run_metrics(root, no_plots=True)
    assert not (tmp_path / "runs_metrics").exists()
