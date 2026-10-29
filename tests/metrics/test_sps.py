from __future__ import annotations

import numpy as np

from netlab.metrics.sps import SpsResult, compute_sps
from tests.result_samples import case, exported, placement, row


def _sps_fixture() -> dict:
    def records(a, b):
        return [row("A", "B", a, 100), row("A", "C", b, 50)]

    return exported(
        placement(records(100, 50)),
        placement(
            records(100, 50),
            [
                case(records(50, 50), capacity=True),
                case(records(0, 25), name="f2", capacity=True),
            ],
            name="node_to_node_capacity_matrix",
            capacity=True,
        ),
        q=0.25,
    )


def test_compute_sps_series_and_tails() -> None:
    res = _sps_fixture()
    sps: SpsResult = compute_sps(res)
    # Expected per-iteration SPS values (failures only, baseline excluded)
    expected = [(50.0 + 50.0) / 150.0, 25.0 / 150.0]
    assert np.allclose(list(sps.series.values), expected)
    # Tails (lower interpolation over 2 values); p50 picks the lower of the two
    assert np.isclose(sps.tails["p50"], expected[1])
    assert np.isclose(sps.sps_at_probability[90.0], expected[1])


def test_sps_case_occurrences_weighting() -> None:
    """Verify SPS expands by case occurrences.

    Setup: 2 unique patterns, one with case occurrences=3.
    - f1 (count=3): SPS = 1.0 (full capacity)
    - f2 (count=1): SPS = 0.5
    Series should have 4 entries: [1.0, 1.0, 1.0, 0.5]
    """
    res = exported(
        placement([row()]),
        placement(
            [row()],
            [
                case([row()], 3, capacity=True),
                case([row(delivered=50)], 1, name="f2", capacity=True),
            ],
            name="node_to_node_capacity_matrix",
            capacity=True,
        ),
    )
    sps: SpsResult = compute_sps(res)
    assert len(sps.series) == 4
    vals = sorted(sps.series.values)
    assert np.isclose(vals[0], 0.5)
    assert all(np.isclose(v, 1.0) for v in vals[1:])
