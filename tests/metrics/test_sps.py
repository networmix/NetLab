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
                "metadata": {"iterations": 1, "unique_patterns": 1},
                "data": {
                    "baseline": {"failure_id": "baseline", "flows": base_tm},
                    "flow_results": [],
                },
            },
            "node_to_node_capacity_matrix": {
                "data": {
                    "baseline": {"failure_id": "baseline", "flows": it0_caps},
                    "flow_results": [
                        {"failure_id": "f1", "flows": it1_caps},
                        {"failure_id": "f2", "flows": it2_caps},
                    ],
                }
            },
        }
    }


def test_compute_sps_series_and_tails() -> None:
    res = _sps_fixture()
    sps: SpsResult = compute_sps(res)
    # Expected per-iteration SPS values (failures only, baseline excluded)
    expected = [(50.0 + 50.0) / 150.0, 25.0 / 150.0]
    assert np.allclose(list(sps.series.values), expected)
    # Tails (lower interpolation over 2 values); p50 picks the lower of the two
    assert np.isclose(sps.tails["p50"], expected[1])
    assert np.isclose(sps.sps_at_probability[90.0], expected[1])


def test_sps_occurrence_count_weighting() -> None:
    """Verify SPS expands by occurrence_count.

    Setup: 2 unique patterns, one with occurrence_count=3.
    - f1 (count=3): SPS = 1.0 (full capacity)
    - f2 (count=1): SPS = 0.5
    Series should have 4 entries: [1.0, 1.0, 1.0, 0.5]
    """
    res = {
        "steps": {
            "tm_placement": {
                "data": {
                    "baseline": {
                        "flows": [
                            {"source": "A", "destination": "B", "demand": 100.0},
                        ]
                    },
                    "flow_results": [],
                }
            },
            "node_to_node_capacity_matrix": {
                "data": {
                    "baseline": {"flows": []},
                    "flow_results": [
                        {
                            "failure_id": "f1",
                            "occurrence_count": 3,
                            "flows": [
                                {"source": "A", "destination": "B", "placed": 100.0}
                            ],
                        },
                        {
                            "failure_id": "f2",
                            "occurrence_count": 1,
                            "flows": [
                                {"source": "A", "destination": "B", "placed": 50.0}
                            ],
                        },
                    ],
                }
            },
        }
    }
    sps: SpsResult = compute_sps(res)
    assert len(sps.series) == 4
    vals = sorted(sps.series.values)
    assert np.isclose(vals[0], 0.5)
    assert all(np.isclose(v, 1.0) for v in vals[1:])
