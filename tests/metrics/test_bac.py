from __future__ import annotations

import numpy as np
import pytest

from metrics.bac import BacResult, compute_bac


def _build_bac_results() -> dict:
    # Baseline delivered = 200
    baseline_flows = [
        {"source": "A", "destination": "B", "placed": 100.0, "demand": 100.0},
        {"source": "A", "destination": "C", "placed": 50.0, "demand": 50.0},
        {"source": "B", "destination": "C", "placed": 50.0, "demand": 50.0},
    ]
    f1_flows = [
        {"source": "A", "destination": "B", "placed": 80.0, "demand": 100.0},
        {"source": "A", "destination": "C", "placed": 40.0, "demand": 50.0},
        {"source": "B", "destination": "C", "placed": 30.0, "demand": 50.0},
    ]
    f2_flows = [
        {"source": "A", "destination": "B", "placed": 100.0, "demand": 100.0},
        {"source": "A", "destination": "C", "placed": 40.0, "demand": 50.0},
        {"source": "B", "destination": "C", "placed": 40.0, "demand": 50.0},
    ]
    return {
        "workflow": {"tm_placement": {"step_type": "TrafficMatrixPlacement"}},
        "steps": {
            "tm_placement": {
                "metadata": {"baseline": True},
                "data": {
                    "flow_results": [
                        {"failure_id": "baseline", "flows": baseline_flows},
                        {"failure_id": "f1", "flows": f1_flows},
                        {"failure_id": "f2", "flows": f2_flows},
                    ]
                },
            }
        },
    }


def test_compute_bac_core_stats() -> None:
    res = _build_bac_results()
    bac: BacResult = compute_bac(res, step_name="tm_placement", mode="auto")
    assert bac.mode == "placement"
    # Delivered series expected: [200, 150, 180]
    expected = [200.0, 150.0, 180.0]
    assert list(bac.series.values) == expected
    assert bac.offered == 200.0
    # Quantiles with 'lower' interpolation
    for p in (0.50, 0.90, 0.95, 0.99, 0.999, 0.9999):
        assert bac.quantiles_abs[p] == 180.0
        assert np.isclose(bac.quantiles_pct[p], 0.9)
    # Availability at thresholds
    assert np.isclose(bac.availability_at_pct_of_offer[90.0], 2.0 / 3.0)
    for pct in (95.0, 99.0, 99.9, 99.99):
        assert np.isclose(bac.availability_at_pct_of_offer[pct], 1.0 / 3.0)
    # BW at probability (lower-tail)
    for pct in (90.0, 95.0, 99.0, 99.9, 99.99):
        assert bac.bw_at_probability_abs[pct] == 150.0
        assert np.isclose(bac.bw_at_probability_pct[pct], 0.75)
    # AUC normalized
    assert np.isclose(bac.auc_normalized, (1.0 + 0.75 + 0.9) / 3.0)


def test_bac_mode_detection_maxflow() -> None:
    res = _build_bac_results()
    # Change step_type to MaxFlow
    res["workflow"]["tm_placement"]["step_type"] = "MaxFlow"
    bac = compute_bac(res, step_name="tm_placement", mode="auto")
    assert bac.mode == "maxflow"


def test_bac_requires_baseline_first() -> None:
    res = _build_bac_results()
    # Swap iterations to violate baseline-first rule
    flow_results = res["steps"]["tm_placement"]["data"]["flow_results"]
    flow_results[0], flow_results[1] = flow_results[1], flow_results[0]
    with pytest.raises(ValueError, match="baseline must be first"):
        compute_bac(res, step_name="tm_placement", mode="auto")
