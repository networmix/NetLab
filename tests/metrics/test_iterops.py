from __future__ import annotations

import numpy as np
import pytest

from metrics.iterops import compute_iter_ops


def _iterops_fixture() -> dict:
    return {
        "steps": {
            "tm_placement": {
                "metadata": {
                    "duration_sec": 10.0,
                    "iterations": 5,
                    "unique_patterns": 2,
                },
                "data": {
                    "baseline": {"failure_id": "baseline", "flows": []},
                    "flow_results": [
                        {"failure_id": "f1", "occurrence_count": 3, "flows": []},
                        {"failure_id": "f2", "occurrence_count": 2, "flows": []},
                    ],
                },
            }
        }
    }


def test_compute_iter_ops_counts() -> None:
    res = _iterops_fixture()
    ops = compute_iter_ops(res)
    # 3 + 2 = 5 failure iterations
    assert ops.failures_count == 5
    assert ops.unique_patterns == 2
    assert ops.total_iterations_count == 6  # 1 baseline + 5 failures


def test_compute_iter_ops_timing() -> None:
    res = _iterops_fixture()
    ops = compute_iter_ops(res)
    assert np.isclose(ops.total_duration_sec, 10.0)
    assert np.isclose(ops.per_iter_duration_sec, 10.0 / 6.0)


def test_compute_iter_ops_no_occurrence_count() -> None:
    """Without occurrence_count, each pattern counts as 1."""
    res = {
        "steps": {
            "tm_placement": {
                "metadata": {"duration_sec": 5.0},
                "data": {
                    "baseline": {"failure_id": "baseline", "flows": []},
                    "flow_results": [
                        {"failure_id": "f1", "flows": []},
                        {"failure_id": "f2", "flows": []},
                    ],
                },
            }
        }
    }
    ops = compute_iter_ops(res)
    assert ops.failures_count == 2
    assert ops.unique_patterns == 2
    assert ops.total_iterations_count == 3


def test_compute_iter_ops_requires_baseline() -> None:
    res = {
        "steps": {
            "tm_placement": {
                "metadata": {},
                "data": {"flow_results": []},
            }
        }
    }
    with pytest.raises(ValueError, match="baseline dict required"):
        compute_iter_ops(res)


def test_iterops_serialization() -> None:
    res = _iterops_fixture()
    ops = compute_iter_ops(res)
    d = ops.to_jsonable()
    assert d["iters_fail"] == 5
    assert d["iters_total"] == 6
    assert d["unique_patterns"] == 2

    s = ops.flat_series()
    assert s["iters_fail"] == 5.0
    assert s["iters_total"] == 6.0
    assert s["unique_patterns"] == 2.0
