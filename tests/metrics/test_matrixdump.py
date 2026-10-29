from __future__ import annotations

import numpy as np

from netlab.metrics.matrixdump import compute_pair_matrices
from tests.result_samples import case, exported, placement, row


def _fixture() -> dict:
    def records(a, b):
        return [row("m1/d1/r", "m1/d2/x", a, 100), row("m1/d1/r", "m2/d3/x", b, 50)]

    return exported(
        placement(
            records(100, 50), [case(records(80, 40)), case(records(60, 50), name="f2")]
        ),
        placement(
            records(120, 45),
            [case(records(50, 55), capacity=True)],
            name="node_to_node_capacity_matrix",
            capacity=True,
        ),
        q=0.25,
    )


def test_compute_pair_matrices_correctness() -> None:
    res = _fixture()
    tm_abs, tm_norm, mf_abs, mf_norm = compute_pair_matrices(res, include_maxflow=True)
    # Expected columns (pairs) preserving full endpoint paths and arrow separator
    expected_cols = ["m1/d1/r→m1/d2/x", "m1/d1/r→m2/d3/x"]
    assert list(tm_abs.index) == expected_cols
    assert list(tm_norm.index) == expected_cols
    assert list(mf_abs.index) == expected_cols
    assert list(mf_norm.index) == expected_cols

    # For tm_abs at p50 over failures only [80,60] and [40,50] with 'lower'
    assert np.isclose(tm_abs.loc["m1/d1/r→m1/d2/x", "p50.0"], 15.0)
    assert np.isclose(tm_abs.loc["m1/d1/r→m2/d3/x", "p50.0"], 10.0)
    # Normalized by baseline demand (100, 50), clipped to 1.0
    assert np.isclose(tm_norm.loc["m1/d1/r→m1/d2/x", "p50.0"], 0.6)
    assert np.isclose(tm_norm.loc["m1/d1/r→m2/d3/x", "p50.0"], 0.8)

    # MaxFlow percentiles (single failure iteration)
    assert np.isclose(mf_abs.loc["m1/d1/r→m1/d2/x", "p50.0"], 12.5)
    assert np.isclose(mf_abs.loc["m1/d1/r→m2/d3/x", "p50.0"], 13.75)
    # Normalized by baseline demand (clipped to 1.0)
    assert np.isclose(mf_norm.loc["m1/d1/r→m1/d2/x", "p50.0"], 0.5)
    assert np.isclose(mf_norm.loc["m1/d1/r→m2/d3/x", "p50.0"], 1.0)
