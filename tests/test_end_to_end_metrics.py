from __future__ import annotations

import numpy as np

from netlab.metrics.bac import compute_bac
from netlab.metrics.latency import compute_latency_stretch
from netlab.metrics.matrixdump import compute_pair_matrices
from netlab.metrics.msd import compute_alpha_star
from netlab.metrics.sps import compute_sps
from tests.result_samples import case, exported, msd, placement, row


def _make_results_payload() -> dict:
    def records(a, b, ca=None, cb=None):
        return [
            row("m1/d1/r", "m1/d2/x", a, 200, costs=ca),
            row("m1/d1/r", "m2/d3/x", b, 100, costs=cb),
        ]

    return exported(
        msd(1.1, (100, 50)),
        placement(
            records(200, 100, {10: 200}, {20: 100}),
            [
                case(records(160, 80, {10: 80, 12: 80}, {20: 40, 30: 40})),
                case(records(120, 100, {10: 120}, {20: 25, 40: 75}), name="f2"),
            ],
        ),
        placement(
            records(240, 90),
            [case(records(100, 110), capacity=True)],
            name="node_to_node_capacity_matrix",
            capacity=True,
        ),
        q=0.5,
    )


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
