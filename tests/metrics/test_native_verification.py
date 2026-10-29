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
from netlab.results import baseline_case, require_step, rows, sampled_cases
from netlab.run_record import run_record_path
from tests.result_samples import exported


@pytest.fixture(scope="module")
def native_run():
    return native(scenario_document(1))


@pytest.fixture(scope="module")
def native_results(native_run):
    return native_run[0]


@pytest.fixture(scope="module")
def native_record(native_run):
    return native_run[1]


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
        "quantum",
        "alpha",
        "maxflow_baseline",
    ],
)
def test_current_results_reject_corrupt_numbers(
    native_results, native_record, tmp_path, mutation
):
    r = deepcopy(native_results)
    step = require_step(r, "tm_placement")
    flow = rows(sampled_cases(step)[0])[0]
    if mutation == "placed":
        flow["result"]["delivered_units"] += 1
    elif mutation == "dropped":
        flow["reference_shortfall_units"] += 1
    elif mutation == "cost":
        flow["cost_distribution"]["bins"][0]["marginal_cost"] = math.nan
    elif mutation == "cost_volume":
        flow["cost_distribution"]["bins"][0]["units"] += 1
    elif mutation == "summary":
        baseline_case(step)["states"]["converged"]["totals"][
            "reference_request_units"
        ] = math.inf
    elif mutation == "count":
        step["result"]["metadata"]["sampling"]["iterations"] += 1
    elif mutation == "quantum":
        r["units_report"]["q"] = 0
    elif mutation == "alpha":
        require_step(r, "msd_baseline")["result"]["alpha_feasible"]["value"] = math.nan
    elif mutation == "maxflow_baseline":
        rows(baseline_case(require_step(r, "node_to_node_capacity_matrix")))[0][
            "result"
        ]["delivered_units"] = -1
    with pytest.raises(ValueError):
        analyze_one_seed(
            r, tmp_path, False, enable_maxflow=True, run_record=native_record
        )


def test_maxflow_group_names_cannot_silently_become_outages(
    native_results, native_record
):
    r = deepcopy(native_results)
    rows(baseline_case(require_step(r, "node_to_node_capacity_matrix")))[0][
        "endpoints"
    ]["sources"] = ["^a/dc/node$"]
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


def test_rerun_replaces_stale_seeds_and_optional_artifacts(
    native_results, native_record, tmp_path
):
    root = tmp_path / "runs"
    root.mkdir()
    first = root / "baseline__seed1_scenario.results.json"
    second = root / "baseline__seed2_scenario.results.json"
    first.write_text(
        json.dumps(
            dict(native_results, metadata=dict(native_results["metadata"], seed=1))
        )
    )
    second.write_text(
        json.dumps(
            dict(native_results, metadata=dict(native_results["metadata"], seed=2))
        )
    )
    for path in (first, second):
        run_record_path(path).write_text(json.dumps(native_record))
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


def test_invalid_rerun_does_not_publish_partial_metrics(
    native_results, native_record, tmp_path
):
    root = tmp_path / "runs"
    root.mkdir()
    first = root / "baseline__seed1_scenario.results.json"
    first.write_text(
        json.dumps(
            dict(native_results, metadata=dict(native_results["metadata"], seed=1))
        )
    )
    run_record_path(first).write_text(json.dumps(native_record))
    run_metrics(root, no_plots=True)
    out = tmp_path / "runs_metrics"
    before = {
        str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()
    }
    # A later seed fails after the first one was analyzed.
    bad = deepcopy(native_results)
    bad["metadata"]["seed"] = 2
    rows(baseline_case(require_step(bad, "tm_placement")))[0]["result"][
        "delivered_units"
    ] = -1
    (root / "baseline__seed2_scenario.results.json").write_text(json.dumps(bad))
    run_record_path(root / "baseline__seed2_scenario.results.json").write_text(
        json.dumps(native_record)
    )
    with pytest.raises(ValueError):
        run_metrics(root, no_plots=True)
    after = {
        str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()
    }
    assert before == after
    assert not (out / "baseline" / "seed2").exists()


def test_cost_invalid_denominator_and_actual_zero():
    results = exported(
        {
            "name": "cost_power",
            "step_type": "CostPower",
            "result": {
                "levels": [
                    {"level": 0, "rows": [{"capex_total": 0, "power_total_watts": 0}]}
                ]
            },
        }
    )
    assert compute_cost_power(results, 100, 100).per_offered_demand__usd_per_gbit == 0
    assert (
        compute_cost_power(results, math.inf, 0).per_offered_demand__usd_per_gbit
        is None
    )
    results["steps"][0]["result"]["levels"][0]["rows"][0]["capex_total"] = -1
    with pytest.raises(ValueError):
        compute_cost_power(results, 100, 100)


def test_multiseed_report_against_independent_reference_and_pattern_permutation(
    tmp_path,
):
    import pandas as pd

    from dev.verify_metrics import verify_multiseed

    root = verify_multiseed(tmp_path)
    before = pd.read_csv(tmp_path / "multiseed_metrics" / "project.csv")
    for path in root.glob("*.results.json"):
        result = json.loads(path.read_text())
        for name in ["tm_placement", "node_to_node_capacity_matrix"]:
            step = require_step(result, name)
            step["result"]["cases"].reverse()
        path.write_text(json.dumps(result))
    run_metrics(root, no_plots=True, enable_maxflow=True)
    after = pd.read_csv(tmp_path / "multiseed_metrics" / "project.csv")
    pd.testing.assert_frame_equal(before, after)


def test_metrics_json_is_strict_and_seed_mismatch_is_rejected(
    native_results, native_record, tmp_path
):
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
    doc["workflow"][1]["outputs"]["cost_distribution"] = False
    raw, run_record = native(doc)
    _, bac, _, latency, _, _, _ = analyze_one_seed(
        raw, tmp_path, True, run_record=run_record
    )
    assert bac.series.tolist() == [8, 3]
    assert latency.baseline["reference_coverage"] == 0
    assert latency.failures["reference_coverage"] == 0
    assert math.isnan(latency.failures["p99"])
    assert not (tmp_path / "latency.png").exists()


def test_changed_input_cannot_publish_metrics(
    native_results, native_record, tmp_path, monkeypatch
):
    import netlab.metrics.batch as batch

    root = tmp_path / "runs"
    root.mkdir()
    path = root / "baseline__seed42_scenario.results.json"
    path.write_text(json.dumps(native_results))
    run_record_path(path).write_text(json.dumps(native_record))
    original = batch.analyze_one_seed

    def analyze_and_mutate(*args, **kwargs):
        output = original(*args, **kwargs)
        path.write_text(path.read_text() + "\n")
        return output

    monkeypatch.setattr(batch, "analyze_one_seed", analyze_and_mutate)
    with pytest.raises(ValueError, match="changed during metric analysis"):
        batch.run_metrics(root, no_plots=True)
    assert not (tmp_path / "runs_metrics").exists()
