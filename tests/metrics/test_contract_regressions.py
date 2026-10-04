"""Regression tests for aggregation and empirical probabilities."""

import math

import numpy as np
import pytest

from netlab.metrics.bac import compute_bac
from netlab.metrics.common import baseline_demand_map
from netlab.metrics.distributions import availability_curve, curve_on_grid
from netlab.metrics.latency import compute_latency_stretch
from netlab.metrics.matrixdump import compute_pair_matrices
from netlab.metrics.sps import compute_sps
from netlab.metrics_failure import aggregate_failure_metrics, analyze_results


def flow(demand, placed, cost=1):
    return {
        "source": "A",
        "destination": "B",
        "demand": demand,
        "placed": placed,
        "cost_distribution": {str(cost): placed},
    }


def result_data():
    base = {"flows": [flow(40, 40), flow(60, 60, 2)]}
    failures = [
        {"occurrence_count": 2, "flows": [], "summary": {"overall_ratio": 0}},
        {
            "occurrence_count": 1,
            "flows": [flow(40, 10, 2), flow(60, 30, 3)],
            "summary": {"overall_ratio": 0.4},
        },
    ]
    return {
        "steps": {
            name: {"data": {"baseline": base, "flow_results": failures}}
            for name in ("tm_placement", "node_to_node_capacity_matrix")
        }
    }


def test_sum_classes_and_retain_total_outage_samples():
    data = result_data()
    assert baseline_demand_map(data) == {("A", "B"): 100}
    bac = compute_bac(data, step_name="tm_placement")
    assert bac.series.tolist() == [100, 0, 0, 40]
    assert bac.auc_normalized == pytest.approx(0.35)
    assert compute_sps(data).series.tolist() == [0, 0, 0.4]
    absolute, normalized, _, _ = compute_pair_matrices(data, include_maxflow=True)
    assert absolute.loc["A→B", "p50.0"] == 0
    assert normalized.loc["A→B", "p50.0"] == 0


def test_all_failures_disconnected_is_zero_not_missing():
    data = result_data()
    for step in data["steps"].values():
        step["data"]["flow_results"] = [{"occurrence_count": 3, "flows": []}]
    assert compute_sps(data).series.tolist() == [0, 0, 0]
    absolute, normalized, _, _ = compute_pair_matrices(data, include_maxflow=True)
    assert (absolute.loc["A→B"] == 0).all()
    assert (normalized.loc["A→B"] == 0).all()


def test_latency_uses_minimum_across_classes_and_preserves_outage_indices():
    latency = compute_latency_stretch(result_data())
    assert latency.baseline["p99"] == 2
    assert latency.per_iteration is not None
    values = latency.per_iteration["p99"]
    assert len(values) == 3
    assert math.isnan(values[0]) and math.isnan(values[1])
    assert values[2] == 3


def test_failure_summary_weights_patterns_and_counts_present_seeds():
    stats = analyze_results(result_data()).failure_stats
    assert stats["tm_placement"].iterations == 3
    assert stats["tm_placement"].avg_ratio == pytest.approx(0.4 / 3)
    combined = aggregate_failure_metrics([stats, {}])["tm_placement"]
    assert combined.seeds == 1
    assert combined.total_iterations == 3


def test_survival_curve_ties_and_between_sample_thresholds():
    samples = np.array([0, 1, 1, 4, 4, 4], dtype=float)
    xs, probabilities = availability_curve(samples)
    grid = np.array([-1, 0, 0.5, 1, 1.1, 3, 4, 4.01])
    expected = [(samples >= threshold).mean() for threshold in grid]
    np.testing.assert_allclose(curve_on_grid(xs, probabilities, grid), expected)
    np.testing.assert_allclose(probabilities, [1, 5 / 6, 0.5])


def test_comparisons_read_current_latency_and_availability_tail(tmp_path):
    import json

    from netlab.metrics.comparisons import compare_scenarios

    for scenario, values in {"A": [0.7, 0.8, 0.95], "B": [0.6, 0.6, 0.7]}.items():
        for seed, value in enumerate(values):
            directory = tmp_path / scenario / f"seed{seed}"
            directory.mkdir(parents=True)
            (directory / "bac.json").write_text(
                json.dumps(
                    {
                        "quantiles_pct": {"0.999": 1.0},
                        "bw_at_probability_pct": {"99.9": value},
                    }
                )
            )
            (directory / "latency.json").write_text(
                json.dumps({"failures": {"p99": 2 * value}})
            )
    results = {
        r["metric"]: r for r in compare_scenarios(tmp_path, scenarios=("A", "B"))
    }
    assert results["bw_p999_pct"]["mean_diff"] == pytest.approx((0.1 + 0.2 + 0.25) / 3)
    assert results["lat_fail_p99"]["mean_diff"] == pytest.approx(
        2 * (0.1 + 0.2 + 0.25) / 3
    )
    narrow = {r["metric"]: r for r in compare_scenarios(tmp_path, alpha=0.2)}
    assert narrow["lat_fail_p99"]["ci_high"] < results["lat_fail_p99"]["ci_high"]
