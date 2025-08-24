from __future__ import annotations

import numpy as np
import pytest

from metrics.latency import LatencyResult, compute_latency_stretch


def _latency_results() -> dict:
    # Baseline: pair A/B has min cost 10 with positive volume; A/C min cost 20.
    base_flows = [
        {
            "source": "metro1/dc1/r1",
            "destination": "metro1/dc2/r2",
            "cost_distribution": {"10": 5.0, "15": 0.0},
        },
        {
            "source": "metro1/dc1/r1",
            "destination": "metro2/dc3/r3",
            "cost_distribution": {"20": 2.0, "25": 0.0},
        },
    ]
    # Failure 1 increases costs for both pairs; shares include best-path (equal to baseline min)
    f1_flows = [
        {
            "source": "metro1/dc1/node",
            "destination": "metro1/dc2/leaf",
            "cost_distribution": {"10": 2.5, "12": 2.5},
        },
        {
            "source": "metro1/dc1/x",
            "destination": "metro2/dc3/y",
            "cost_distribution": {"20": 1.0, "30": 1.0},
        },
    ]
    # Failure 2 worse for second pair only
    f2_flows = [
        {
            "source": "metro1/dc1",
            "destination": "metro1/dc2",
            "cost_distribution": {"10": 5.0},
        },
        {
            "source": "metro1/dc1",
            "destination": "metro2/dc3",
            "cost_distribution": {"20": 0.5, "40": 1.5},
        },
    ]
    return {
        "steps": {
            "tm_placement": {
                "metadata": {"baseline": True},
                "data": {
                    "flow_results": [
                        {"failure_id": "baseline", "flows": base_flows},
                        {"failure_id": "f1", "flows": f1_flows},
                        {"failure_id": "f2", "flows": f2_flows},
                    ]
                },
            }
        }
    }


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
    # Medians across failures (two values): p50=1.0, p95=1.75 (lower interpolation gives 1.5?), p99 similar.
    # Because implementation uses discrete order and left-search, medians across [1.0,1.0] for p50,
    # and [1.5,2.0] for p95/p99 → median=1.5.
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


def test_latency_requires_baseline_first() -> None:
    res = _latency_results()
    # Reorder to break the baseline-first rule
    fr = res["steps"]["tm_placement"]["data"]["flow_results"]
    fr[0], fr[1] = fr[1], fr[0]
    with pytest.raises(ValueError, match="baseline must be first"):
        compute_latency_stretch(res)
