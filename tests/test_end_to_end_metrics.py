from __future__ import annotations

import numpy as np

from metrics.bac import compute_bac
from metrics.latency import compute_latency_stretch
from metrics.matrixdump import compute_pair_matrices
from metrics.msd import compute_alpha_star
from metrics.sps import compute_sps


def _make_results_payload() -> dict:
    # tm_placement baseline and two failures
    base_tm = [
        {
            "source": "m1/d1/r",
            "destination": "m1/d2/x",
            "demand": 100.0,
            "placed": 100.0,
            "cost_distribution": {"10": 5.0},
        },
        {
            "source": "m1/d1/r",
            "destination": "m2/d3/x",
            "demand": 50.0,
            "placed": 50.0,
            "cost_distribution": {"20": 2.0},
        },
    ]
    tm_i1 = [
        {
            "source": "m1/d1/r",
            "destination": "m1/d2/x",
            "placed": 80.0,
            "cost_distribution": {"10": 2.5, "12": 2.5},
        },
        {
            "source": "m1/d1/r",
            "destination": "m2/d3/x",
            "placed": 40.0,
            "cost_distribution": {"20": 1.0, "30": 1.0},
        },
    ]
    tm_i2 = [
        {
            "source": "m1/d1/r",
            "destination": "m1/d2/x",
            "placed": 60.0,
            "cost_distribution": {"10": 5.0},
        },
        {
            "source": "m1/d1/r",
            "destination": "m2/d3/x",
            "placed": 50.0,
            "cost_distribution": {"20": 0.5, "40": 1.5},
        },
    ]
    # maxflow capacities per iteration
    mf_i0 = [
        {"source": "m1/d1/r", "destination": "m1/d2/x", "placed": 120.0},
        {"source": "m1/d1/r", "destination": "m2/d3/x", "placed": 45.0},
    ]
    mf_i1 = [
        {"source": "m1/d1/r", "destination": "m1/d2/x", "placed": 50.0},
        {"source": "m1/d1/r", "destination": "m2/d3/x", "placed": 55.0},
    ]

    results = {
        "workflow": {"tm_placement": {"step_type": "TrafficMatrixPlacement"}},
        "steps": {
            "msd_baseline": {
                "data": {
                    "alpha_star": 1.1,
                    "base_demands": [{"demand": 100.0}, {"demand": 50.0}],
                }
            },
            "tm_placement": {
                "metadata": {"baseline": True},
                "data": {
                    "flow_results": [
                        {"failure_id": "baseline", "flows": base_tm},
                        {"failure_id": "f1", "flows": tm_i1},
                        {"failure_id": "f2", "flows": tm_i2},
                    ]
                },
            },
            "node_to_node_capacity_matrix": {
                "data": {
                    "flow_results": [
                        {"failure_id": "baseline", "flows": mf_i0},
                        {"failure_id": "f1", "flows": mf_i1},
                    ]
                }
            },
        },
    }
    return results


def test_end_to_end_metrics_pipeline() -> None:
    results = _make_results_payload()

    # BAC end-to-end
    bac = compute_bac(results, step_name="tm_placement", mode="auto")
    assert bac.offered == 150.0  # should be baseline delivered sum (100+50)
    # Delivered series = [150, 120, 110] → normalized mean clip
    assert np.isclose(bac.auc_normalized, (1.0 + 120.0 / 150.0 + 110.0 / 150.0) / 3.0)
    assert bac.quantiles_abs[0.50] >= 100.0
    assert bac.bw_at_probability_pct[99.0] <= 1.0

    # Latency end-to-end (uses tm_placement baseline cost distributions)
    latency = compute_latency_stretch(results)
    assert "p95" in latency.baseline and "p99" in latency.failures
    assert latency.derived["TD99"] >= 1.0

    # Pair matrices across TM and MaxFlow
    tm_abs, tm_norm, mf_abs, mf_norm = compute_pair_matrices(
        results, include_maxflow=True
    )
    assert not tm_abs.empty and not tm_norm.empty
    assert mf_abs is not None and mf_norm is not None

    # SPS uses baseline demands and per-iteration maxflows
    sps = compute_sps(results)
    assert sps.series.size >= 1
    assert 0.0 <= sps.tails["p50"] <= 1.0

    # Alpha*
    alpha = compute_alpha_star(results)
    assert alpha.source == "msd_baseline" and np.isclose(alpha.alpha_star, 1.1)
