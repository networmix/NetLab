from __future__ import annotations

import numpy as np

from metrics.sps import SpsResult, compute_sps


def _sps_fixture() -> dict:
    # Baseline TM with non-zero demands for two pairs
    base_tm = [
        {"source": "A", "destination": "B", "demand": 100.0},
        {"source": "A", "destination": "C", "demand": 50.0},
    ]
    # node_to_node_capacity_matrix per-iteration capacities for exact pairs
    it0_caps = [
        {"source": "A", "destination": "B", "placed": 100.0},
        {"source": "A", "destination": "C", "placed": 50.0},
    ]  # SPS = 1.0
    it1_caps = [
        {"source": "A", "destination": "B", "placed": 50.0},
        {"source": "A", "destination": "C", "placed": 50.0},
    ]  # SPS = (min(0.5,1)*100 + min(1,1)*50) / 150 = (50 + 50)/150 = 2/6 ≈ 0.6667
    it2_caps = [
        {"source": "A", "destination": "B", "placed": 0.0},
        {"source": "A", "destination": "C", "placed": 25.0},
    ]  # SPS = (0 + 0.5*50)/150 = 25/150 = 1/6 ≈ 0.1667
    return {
        "steps": {
            "tm_placement": {
                "metadata": {"baseline": True},
                "data": {
                    "flow_results": [{"failure_id": "baseline", "flows": base_tm}]
                },
            },
            "node_to_node_capacity_matrix": {
                "data": {
                    "flow_results": [
                        {"failure_id": "baseline", "flows": it0_caps},
                        {"failure_id": "f1", "flows": it1_caps},
                        {"failure_id": "f2", "flows": it2_caps},
                    ]
                }
            },
        }
    }


def test_compute_sps_series_and_tails() -> None:
    res = _sps_fixture()
    sps: SpsResult = compute_sps(res)
    # Expected per-iteration SPS values
    expected = [1.0, (50.0 + 50.0) / 150.0, 25.0 / 150.0]
    assert np.allclose(list(sps.series.values), expected)
    # Tails (lower interpolation)
    assert np.isclose(sps.tails["p50"], expected[1])
    assert np.isclose(sps.sps_at_probability[90.0], expected[2])
