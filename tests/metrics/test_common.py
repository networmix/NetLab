from __future__ import annotations

import pytest

from netlab.metrics.common import baseline_demand_map, expand_flow_results


@pytest.mark.parametrize("count", [None, 0, -1, True, "1", 1.5])
def test_expand_flow_results_requires_positive_integer_count(count) -> None:
    iteration = {"failure_id": "f1", "flows": []}
    if count is not None:
        iteration["occurrence_count"] = count
    with pytest.raises(ValueError, match="occurrence_count must be a positive integer"):
        expand_flow_results([iteration])


def test_expand_flow_results_with_counts() -> None:
    fr = [
        {"failure_id": "f1", "occurrence_count": 3, "flows": []},
        {"failure_id": "f2", "occurrence_count": 2, "flows": []},
    ]
    expanded = expand_flow_results(fr)
    assert len(expanded) == 5
    # First 3 are f1, next 2 are f2
    assert all(e["failure_id"] == "f1" for e in expanded[:3])
    assert all(e["failure_id"] == "f2" for e in expanded[3:])


def test_expand_flow_results_count_one() -> None:
    fr = [
        {"failure_id": "f1", "occurrence_count": 1, "flows": []},
    ]
    expanded = expand_flow_results(fr)
    assert len(expanded) == 1


def test_expand_flow_results_empty() -> None:
    assert expand_flow_results([]) == []


def test_baseline_demand_map_basic() -> None:
    results = {
        "steps": {
            "tm_placement": {
                "data": {
                    "baseline": {
                        "flows": [
                            {
                                "source": "m1/d1/r1",
                                "destination": "m2/d2/r2",
                                "demand": 100.0,
                            },
                            {
                                "source": "m1/d1/r1",
                                "destination": "m1/d1/r1",
                                "demand": 50.0,
                            },  # self-loop, skip
                        ]
                    }
                }
            }
        }
    }
    dm = baseline_demand_map(results)
    assert dm == {("m1/d1/r1", "m2/d2/r2"): 100.0}
