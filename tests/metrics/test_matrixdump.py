from __future__ import annotations

import numpy as np

from metrics.matrixdump import compute_pair_matrices


def _fixture() -> dict:
    # Baseline TM with demands
    base_tm = [
        {
            "source": "m1/d1/r",
            "destination": "m1/d2/x",
            "demand": 100.0,
            "placed": 100.0,
        },
        {"source": "m1/d1/r", "destination": "m2/d3/x", "demand": 50.0, "placed": 50.0},
    ]
    # Iterations with placed values for TM and MaxFlow
    tm_i1 = [
        {"source": "m1/d1/r", "destination": "m1/d2/x", "placed": 80.0},
        {"source": "m1/d1/r", "destination": "m2/d3/x", "placed": 40.0},
    ]
    tm_i2 = [
        {"source": "m1/d1/r", "destination": "m1/d2/x", "placed": 60.0},
        {"source": "m1/d1/r", "destination": "m2/d3/x", "placed": 50.0},
    ]
    mf_i1 = [
        {"source": "m1/d1/r", "destination": "m1/d2/x", "placed": 120.0},
        {"source": "m1/d1/r", "destination": "m2/d3/x", "placed": 45.0},
    ]
    mf_i2 = [
        {"source": "m1/d1/r", "destination": "m1/d2/x", "placed": 50.0},
        {"source": "m1/d1/r", "destination": "m2/d3/x", "placed": 55.0},
    ]
    return {
        "steps": {
            "tm_placement": {
                "data": {
                    "flow_results": [
                        {"failure_id": "baseline", "flows": base_tm},
                        {"failure_id": "f1", "flows": tm_i1},
                        {"failure_id": "f2", "flows": tm_i2},
                    ]
                }
            },
            "node_to_node_capacity_matrix": {
                "data": {
                    "flow_results": [
                        {"failure_id": "baseline", "flows": mf_i1},
                        {"failure_id": "f1", "flows": mf_i2},
                    ]
                }
            },
        }
    }


def test_compute_pair_matrices_correctness() -> None:
    res = _fixture()
    tm_abs, tm_norm, mf_abs, mf_norm = compute_pair_matrices(res, include_maxflow=True)
    # Expected columns (pairs) normalized to canonical DC path and arrow separator
    expected_cols = ["m1/d1→m1/d2", "m1/d1→m2/d3"]
    assert list(tm_abs.index) == expected_cols
    assert list(tm_norm.index) == expected_cols
    assert list(mf_abs.index) == expected_cols
    assert list(mf_norm.index) == expected_cols

    # For tm_abs at p50 including baseline [100,80,60] and [50,40,50] with 'lower'
    assert np.isclose(tm_abs.loc["m1/d1→m1/d2", "p50.0"], 80.0)
    assert np.isclose(tm_abs.loc["m1/d1→m2/d3", "p50.0"], 50.0)
    # Normalized by baseline demand (100, 50), clipped to 1.0
    assert np.isclose(tm_norm.loc["m1/d1→m1/d2", "p50.0"], 0.8)
    assert np.isclose(tm_norm.loc["m1/d1→m2/d3", "p50.0"], 1.0)

    # MaxFlow percentiles
    assert np.isclose(mf_abs.loc["m1/d1→m1/d2", "p50.0"], 50.0)
    assert np.isclose(mf_abs.loc["m1/d1→m2/d3", "p50.0"], 45.0)
    # Normalized by baseline demand
    assert np.isclose(mf_norm.loc["m1/d1→m1/d2", "p50.0"], 0.5)
    assert np.isclose(mf_norm.loc["m1/d1→m2/d3", "p50.0"], 0.9)
