from __future__ import annotations

from metrics.common import baseline_demand_map, canonical_dc, expand_flow_results


def test_expand_flow_results_default_count() -> None:
    fr = [{"failure_id": "f1", "flows": []}]
    expanded = expand_flow_results(fr)
    assert len(expanded) == 1
    assert expanded[0] is fr[0]


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


def test_canonical_dc_full_path() -> None:
    assert canonical_dc("metro1/dc1/rack/node") == "metro1/dc1"


def test_canonical_dc_already_canonical() -> None:
    assert canonical_dc("metro1/dc1") == "metro1/dc1"


def test_canonical_dc_single_component() -> None:
    assert canonical_dc("metro1") == "metro1"


def test_canonical_dc_empty() -> None:
    assert canonical_dc("") == ""


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
    assert dm == {("m1/d1", "m2/d2"): 100.0}
