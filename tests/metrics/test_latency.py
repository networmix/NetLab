from __future__ import annotations

import numpy as np
import pytest

from netlab.metrics.latency import LatencyResult, compute_latency_stretch
from tests.result_samples import case, exported, placement, row


def _latency_results() -> dict:
    def records(a, b):
        return [
            row("metro1/dc1/r1", "metro1/dc2/r2", 10, 10, costs=a),
            row("metro1/dc1/r1", "metro2/dc3/r3", 4, 4, costs=b),
        ]

    return exported(
        placement(
            records({10: 10, 15: 0}, {20: 4, 25: 0}),
            [
                case(records({10: 5, 12: 5}, {20: 2, 30: 2})),
                case(records({10: 10}, {20: 1, 40: 3}), name="f2"),
            ],
        ),
        q=0.5,
    )


def test_compute_latency_stretch_correctness() -> None:
    res = _latency_results()
    out: LatencyResult = compute_latency_stretch(res)
    # Baseline per-pair minimum costs: (A,B)=10, (A,C)=20
    # Baseline distribution equals min cost only (stretch 1 everywhere)
    # Baseline weighted percentiles → 1.0
    assert np.isclose(out.baseline["p50"], 1.0)
    assert np.isclose(out.baseline["p95"], 1.0)
    assert np.isclose(out.baseline["p99"], 1.0)
    assert np.isclose(out.baseline["SLO_1_2"], 1.0)
    assert np.isclose(out.baseline["SLO_1_5"], 1.0)
    assert np.isclose(out.baseline["best_path_share"], 1.0)
    assert np.isclose(out.baseline["WES"], 0.0)

    # Failures: compute weighted quantiles by hand
    # Failure 1 pairs & stretches:
    #  (A,B): costs {10:2.5, 12:2.5} over denom 10 → stretches {1.0:2.5, 1.2:2.5}
    #  (A,C): costs {20:1.0, 30:1.0} over denom 20 → stretches {1.0:1.0, 1.5:1.0}
    # Combined weighted values (weights sum 7.0): [1.0 x 3.5, 1.2 x 2.5, 1.5 x 1.0]
    # p50 = first index where cumweight >= 3.5/7=0.5 → 1.0
    # p95 = 0.95*7=6.65 -> in last bucket 1.5 → 1.5
    # best_path_share = weight at stretch==1 / total = 3.5/7 = 0.5
    # WES = E[(stretch-1)+] = (0*3.5 + 0.2*2.5 + 0.5*1)/7 = (0 + 0.5 + 0.5)/7 = 1.0/7 ≈ 0.142857
    # Failure 2 pairs & stretches:
    #  (A,B): {10:5} -> stretch 1.0 x 5
    #  (A,C): {20:0.5, 40:1.5} -> stretches {1.0:0.5, 2.0:1.5}
    # Combined sum weights = 7.0; weighted values [1.0 x 5.5, 2.0 x 1.5]
    # p50=1.0; p95=2.0; best_path_share=5.5/7≈0.785714; WES=(0*5.5 + 1.0*1.5)/7=1.5/7≈0.2142857
    # Failure medians: p50=1.0 and p95=p99=(1.5+2.0)/2=1.75.
    assert np.isclose(out.failures["p50"], 1.0)
    # Across failures: medians use arithmetic median for even counts
    assert np.isclose(out.failures["p95"], 1.75)
    assert np.isclose(out.failures["p99"], 1.75)
    # SLO composition at thresholds across failures (median of shares)
    # f1: <=1.2 share = 6/7 ≈ 0.857142; <=1.5 share = 1.0
    # f2: <=1.2 share = 5.5/7 ≈ 0.785714; <=1.5 share = 5.5/7 ≈ 0.785714
    # medians (average of two): ~0.821428 and ~0.892857
    assert np.isclose(out.failures["SLO_1_2"], 0.8214285714)
    assert np.isclose(out.failures["SLO_1_5"], 0.8928571429)
    # best_path_share median: (0.5 + 0.785714)/2 ≈ 0.642857
    assert np.isclose(out.failures["best_path_share"], 0.6428571429)
    # WES median: ((1/7) + (1.5/7))/2 = (1.25/7)
    assert np.isclose(out.failures["WES"], 1.25 / 7.0)

    # Derived
    assert np.isclose(out.derived["TD99"], 1.75)
    assert np.isclose(
        out.derived["SLO_1_2_drop"], 1.0 - out.failures["SLO_1_2"]
    )  # baseline SLO=1
    assert np.isclose(
        out.derived["best_path_share_drop"], 1.0 - out.failures["best_path_share"]
    )
    assert np.isclose(out.derived["WES_delta"], out.failures["WES"] - 0.0)


def test_latency_requires_baseline_key() -> None:
    res = _latency_results()
    del res["steps"][0]["result"]["cases"][0]
    with pytest.raises(ValueError, match="baseline case required"):
        compute_latency_stretch(res)


def test_latency_occurrences_weighting() -> None:
    """Verify latency stretch weights by case occurrences.

    Setup:
    - Baseline: pair (m1/d1, m2/d2) with cost 10, volume 5.
    - Failure f1 (count=8): stretch 2.0 (cost 20, volume 5)
    - Failure f2 (count=2): stretch 1.0 (cost 10, volume 5)

    Without weighting: median of [2.0, 1.0] = 1.5
    With weighting: median of [2.0]*8 + [1.0]*2 = 2.0 (both middle values in the sorted sample)
    """
    baseline = [row("m1/d1/r", "m2/d2/r", 5, 5, costs={10: 5})]
    res = exported(
        placement(
            baseline,
            [
                case([row("m1/d1/r", "m2/d2/r", 5, 5, costs={20: 5})], 8),
                case(baseline, 2, name="f2"),
            ],
        )
    )
    out = compute_latency_stretch(res)
    assert np.isclose(out.failures["p50"], 2.0)
