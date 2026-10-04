"""Check metrics against independent calculations and scaling invariants."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from netlab.metrics.bac import _compute_bac_stats, compute_bac
from netlab.metrics.costpower import compute_cost_power
from netlab.metrics.distributions import availability_curve, curve_on_grid
from netlab.metrics.latency import compute_latency_stretch
from netlab.metrics.matrixdump import compute_pair_matrices
from netlab.metrics.msd import compute_alpha_star
from netlab.metrics.paired import holm_adjust, paired_t
from netlab.metrics.sps import compute_sps


def record(source, placed, cost=1, demand=None):
    return dict(
        source=source,
        destination="b/dc/node",
        priority=0,
        demand=placed if demand is None else demand,
        placed=placed,
        dropped=0 if demand is None else demand - placed,
        cost_distribution={str(cost): placed} if placed else {},
    )


def payload(samples, baseline=9):
    return {
        "steps": {
            name: {
                "data": {
                    "baseline": {"flows": [record("a/dc/node", baseline)]},
                    "flow_results": [
                        dict(occurrence_count=1, flows=[record("a/dc/node", x)])
                        for x in samples
                    ],
                }
            }
            for name in ("tm_placement", "node_to_node_capacity_matrix")
        }
    }


def inverse_survival_oracle(values, percent):
    p = Fraction(str(percent)) / 100
    return max(
        x
        for x in set(values)
        if Fraction(sum(y >= x for y in values), len(values)) >= p
    )


@pytest.mark.parametrize(
    "n", [1, 2, 3, 10, 11, 20, 21, 100, 101, 1000, 1001, 10000, 10001]
)
def test_bandwidth_probability_matches_counting_oracle(n):
    samples = list(range(n))
    actual = _compute_bac_stats(pd.Series(samples, dtype=float), max(n - 1, 1))[4]
    for percent, bandwidth in actual.items():
        # Independent rational integer-count oracle, restricted to candidate tail ranks.
        candidates = samples[: max(2, n // 10 + 1)]
        p = Fraction(str(percent)) / 100
        expected = max(x for x in candidates if Fraction(n - x, n) >= p)
        assert bandwidth == expected, (n, percent, bandwidth, expected)


def test_bandwidth_replicating_empirical_distribution_is_invariant():
    values = list(range(20))
    one = _compute_bac_stats(pd.Series(values), 19)[4]
    repeat = _compute_bac_stats(pd.Series(values * 3), 19)[4]
    assert one == repeat


def test_sps_probability_matches_counting_oracle():
    result = compute_sps(payload(range(10)))
    assert result.sps_at_probability[90] == pytest.approx(1 / 9)


def test_distinct_physical_endpoints_do_not_share_latency_reference_or_matrix_row():
    baseline = {"flows": [record("a/dc/node1", 1, 1), record("a/dc/node2", 1, 10)]}
    results = payload([])
    data = results["steps"]["tm_placement"]["data"]
    data["baseline"] = baseline
    data["flow_results"] = [dict(baseline, occurrence_count=1)]
    latency = compute_latency_stretch(results)
    assert latency.baseline["p99"] == 1
    assert latency.baseline["WES"] == 0
    matrix = compute_pair_matrices(results, False)[0]
    assert len(matrix.index) == 2


def test_missing_cost_is_unavailable_not_free():
    cost = compute_cost_power({}, 100, 100)
    assert math.isnan(cost.capex_total)
    assert cost.per_offered_demand__usd_per_gbit is None


@pytest.mark.parametrize("bad", [-1, math.nan, math.inf])
def test_alpha_rejects_invalid_numeric_values(bad):
    with pytest.raises(ValueError):
        compute_alpha_star(
            {
                "steps": {
                    "msd_baseline": {
                        "data": {"alpha_star": bad, "base_demands": [{"volume": 1}]}
                    }
                }
            }
        )


def test_zero_baseline_preserves_absolute_outage_not_fictitious_availability():
    result = compute_bac(payload([0, 0], baseline=0), "tm_placement")
    assert result.series.tolist() == [0, 0, 0]
    assert result.bw_at_probability_abs[99] == 0
    assert math.isnan(result.auc_normalized)
    assert math.isnan(result.bw_at_probability_pct[99])


def test_baseline_only_analysis_has_one_sample():
    result = compute_bac(payload([], baseline=10), "tm_placement")
    assert result.series.tolist() == [10]
    assert result.bw_at_probability_pct[99] == 1


def test_empirical_survival_on_random_grids():
    rng = np.random.default_rng(744)
    for _ in range(100):
        samples = rng.integers(0, 30, size=int(rng.integers(1, 100)))
        grid = np.arange(-0.5, 31, 0.5)
        expected = [(samples >= threshold).mean() for threshold in grid]
        np.testing.assert_allclose(
            curve_on_grid(*availability_curve(samples), grid), expected
        )


def test_paired_statistics_against_scipy_and_scale_invariance():
    rng = np.random.default_rng(745)
    for n in [3, 5, 10, 50]:
        for scale in [1e-150, 1e-15, 1, 1e15, 1e150]:
            a, b = rng.normal(size=(2, n)) * scale
            actual = paired_t(a, b)
            expected = stats.ttest_rel(a, b)
            np.testing.assert_allclose(actual["t_stat"], expected.statistic, rtol=1e-12)
            np.testing.assert_allclose(actual["p"], expected.pvalue, rtol=1e-12)
            ci = expected.confidence_interval()
            np.testing.assert_allclose(
                [actual["ci_low"] / scale, actual["ci_high"] / scale],
                [ci.low / scale, ci.high / scale],
                rtol=1e-12,
            )


def test_holm_matches_hand_calculation():
    # Sorted .01, .03, .04: maxima of .03, .06, .04 -> .03, .06, .06.
    result = holm_adjust([("c", 0.04), ("a", 0.01), ("b", 0.03), ("missing", math.nan)])
    assert result["a"] == pytest.approx(0.03)
    assert result["b"] == result["c"] == pytest.approx(0.06)
    assert math.isnan(result["missing"])


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_inference_remains_valid_at_extreme_measurement_scales(scale):
    sample = np.array([1.0, 2.0, 4.0, 8.0])
    original = paired_t(sample, np.zeros(4))
    scaled = paired_t(sample * scale, np.zeros(4))
    assert scaled["p"] == pytest.approx(original["p"])
    assert scaled["t_stat"] == pytest.approx(original["t_stat"])
    assert not scaled["deterministic"]


def test_constant_decimal_difference_is_detected_exactly():
    assert paired_t(np.full(3, 0.1), np.zeros(3))["deterministic"]


@pytest.mark.parametrize(
    "values", [[("x", 0.1), ("x", 0.2)], [("x", math.inf)], [("x", -0.1)]]
)
def test_holm_rejects_ambiguous_or_invalid_family(values):
    with pytest.raises(ValueError):
        holm_adjust(values)


def test_latency_metrics_ignore_cost_measurement_units():
    def evaluate(scale):
        data = payload([])
        step = data["steps"]["tm_placement"]["data"]
        step["baseline"] = {"flows": [record("A", 10, 2 * scale)]}
        step["flow_results"] = [
            dict(occurrence_count=1, flows=[record("A", 10, 4 * scale)])
        ]
        return compute_latency_stretch(data).failures

    assert evaluate(1e-12) == evaluate(1)


def test_latency_weighted_tails_against_explicit_volume_replication():
    rng = np.random.default_rng(746)
    for _ in range(100):
        costs = rng.integers(1, 20, 5)
        weights = rng.integers(1, 30, 5)
        data = payload([])
        step = data["steps"]["tm_placement"]["data"]
        step["baseline"] = {"flows": [record("A", 1, 1)]}
        flows = [
            record("A", int(w), int(c)) for c, w in zip(costs, weights, strict=True)
        ]
        step["flow_results"] = [dict(occurrence_count=1, flows=flows)]
        values = sorted(
            int(c) for c, w in zip(costs, weights, strict=True) for _ in range(int(w))
        )
        result = compute_latency_stretch(data).failures
        for key, q in [
            ("p50", Fraction(1, 2)),
            ("p95", Fraction(95, 100)),
            ("p99", Fraction(99, 100)),
        ]:
            expected = values[math.ceil(len(values) * q) - 1]
            assert result[key] == expected
        assert result["WES"] == pytest.approx(sum(v - 1 for v in values) / len(values))
        assert result["best_path_share"] == pytest.approx(
            sum(v == 1 for v in values) / len(values)
        )


def test_common_seed_pairing_precedes_normalization(tmp_path):
    import json

    from netlab.metrics.comparisons import compare_scenarios
    from netlab.metrics.summary import build_baseline_normalized_table

    # Seed 1 and 5 must not participate; median(A/B) differs from median(A)/median(B).
    samples = {
        "baseline": {1: 999, 2: 1, 3: 2, 4: 100},
        "candidate": {2: 2, 3: 6, 4: 400, 5: -999},
    }
    for name, rows in samples.items():
        for seed, value in rows.items():
            directory = tmp_path / name / f"seed{seed}"
            directory.mkdir(parents=True)
            (directory / "bac.json").write_text(
                json.dumps(
                    {"auc_normalized": value, "bw_at_probability_pct": {"99.9": value}}
                )
            )
            (directory / "alpha.json").write_text(json.dumps({"alpha_star": value}))
    table = build_baseline_normalized_table(tmp_path)
    assert table.loc["candidate", "auc_norm_r"] == 3
    record = next(
        r
        for r in compare_scenarios(tmp_path, scenarios=("candidate", "baseline"))
        if r["metric"] == "alpha_star"
    )
    assert record["n"] == 3
    assert record["mean_diff"] == pytest.approx((1 + 4 + 300) / 3)


def test_latency_partial_reference_coverage_is_visible():
    data = payload([])
    step = data["steps"]["tm_placement"]["data"]
    step["baseline"] = {"flows": [record("A", 10, 1), record("unreachable", 0)]}
    step["flow_results"] = [
        dict(occurrence_count=1, flows=[record("A", 5, 2), record("unreachable", 5, 3)])
    ]
    result = compute_latency_stretch(data)
    assert result.failures["p99"] == 2
    assert result.failures["reference_coverage"] == 0.5
