from __future__ import annotations

import numpy as np
import pytest

from netlab.metrics.bac import BacResult, compute_bac


def _build_bac_results() -> dict:
    """Build a baseline and weighted failure results."""
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
                "metadata": {"iterations": 2, "unique_patterns": 2},
                "data": {
                    "baseline": {
                        "occurrence_count": 1,
                        "failure_id": "baseline",
                        "flows": baseline_flows,
                    },
                    "flow_results": [
                        {"occurrence_count": 1, "failure_id": "f1", "flows": f1_flows},
                        {"occurrence_count": 1, "failure_id": "f2", "flows": f2_flows},
                    ],
                },
            }
        },
    }


def test_compute_bac_core_stats() -> None:
    res = _build_bac_results()
    bac: BacResult = compute_bac(res, step_name="tm_placement", mode="auto")
    assert bac.mode == "placement"
    # Delivered series: baseline=200, f1=150, f2=180
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
    # AUC normalized: baseline=200/200=1.0, f1=150/200=0.75, f2=180/200=0.9
    assert np.isclose(bac.auc_normalized, (1.0 + 0.75 + 0.9) / 3.0)


def test_bac_mode_detection_maxflow() -> None:
    res = _build_bac_results()
    res["workflow"]["tm_placement"]["step_type"] = "MaxFlow"
    bac = compute_bac(res, step_name="tm_placement", mode="auto")
    assert bac.mode == "maxflow"


def test_bac_requires_baseline() -> None:
    res = _build_bac_results()
    del res["steps"]["tm_placement"]["data"]["baseline"]
    with pytest.raises(ValueError, match="data.baseline dict required"):
        compute_bac(res, step_name="tm_placement", mode="auto")


def test_bac_requires_flow_results_list() -> None:
    res = _build_bac_results()
    res["steps"]["tm_placement"]["data"]["flow_results"] = {}
    with pytest.raises(ValueError, match="flow_results must be a list"):
        compute_bac(res, step_name="tm_placement", mode="auto")


def test_bac_per_flow() -> None:
    """Per-flow BAC separates each demand direction.

    Two flows with different degradation under failure.
    Flow A→B: placed 80 under failure (baseline 100).
    Flow B→C: placed 30 under failure (baseline 50).
    Per-flow AUC should differ.
    """
    baseline_flows = [
        {"source": "A", "destination": "B", "placed": 100.0, "demand": 100.0},
        {"source": "B", "destination": "C", "placed": 50.0, "demand": 50.0},
    ]
    f1_flows = [
        {"source": "A", "destination": "B", "placed": 80.0, "demand": 100.0},
        {"source": "B", "destination": "C", "placed": 30.0, "demand": 50.0},
    ]
    res = {
        "steps": {
            "tm_placement": {
                "data": {
                    "baseline": {"flows": baseline_flows},
                    "flow_results": [
                        {"occurrence_count": 1, "failure_id": "f1", "flows": f1_flows}
                    ],
                }
            }
        },
    }
    bac = compute_bac(res, step_name="tm_placement")

    # Aggregate: offered=150, series=[150, 110], AUC=(1 + 110/150)/2
    assert bac.offered == 150.0
    assert np.isclose(bac.auc_normalized, (1.0 + 110.0 / 150.0) / 2.0)

    assert len(bac.per_flow) == 2

    # Flow A→B: offered=100, series=[100, 80], AUC=(1 + 0.8)/2 = 0.9
    pf_ab = bac.per_flow["A>B"]
    assert pf_ab.offered == 100.0
    assert np.isclose(pf_ab.auc_normalized, 0.9)

    # Flow B→C: offered=50, series=[50, 30], AUC=(1 + 0.6)/2 = 0.8
    pf_bc = bac.per_flow["B>C"]
    assert pf_bc.offered == 50.0
    assert np.isclose(pf_bc.auc_normalized, 0.8)

    assert pf_bc.auc_normalized < pf_ab.auc_normalized


def test_bac_shared_source_keeps_all_directions() -> None:
    bac = compute_bac(_build_bac_results(), step_name="tm_placement")
    assert set(bac.per_flow) == {"A>B", "A>C", "B>C"}
    assert bac.per_flow["A>B"].series.tolist() == [100.0, 80.0, 100.0]
    assert bac.per_flow["A>C"].series.tolist() == [50.0, 40.0, 40.0]
    assert bac.per_flow["B>C"].series.tolist() == [50.0, 30.0, 40.0]
    assert (
        sum(flow.series for flow in bac.per_flow.values()).tolist()
        == bac.series.tolist()
    )


def test_bac_sums_priorities_and_handles_missing_pair() -> None:
    results = _build_bac_results()
    data = results["steps"]["tm_placement"]["data"]
    data["baseline"]["flows"].append(
        {"source": "A", "destination": "B", "priority": 1, "placed": 20.0}
    )
    data["flow_results"][0]["flows"].append(
        {"source": "A", "destination": "B", "priority": 1, "placed": 10.0}
    )
    data["flow_results"][1]["flows"] = [
        rec for rec in data["flow_results"][1]["flows"] if rec["source"] != "A"
    ]
    bac = compute_bac(results, step_name="tm_placement")
    assert bac.per_flow["A>B"].series.tolist() == [120.0, 90.0, 0.0]
    assert bac.per_flow["A>C"].series.tolist() == [50.0, 40.0, 0.0]
    assert (
        sum(flow.series for flow in bac.per_flow.values()).tolist()
        == bac.series.tolist()
    )


@pytest.mark.parametrize(
    "pairs",
    [
        [("A>B", "C"), ("A", "B>C")],
        [("long-site-name-" * 4 + "1", "B"), ("long-site-name-" * 4 + "2", "B")],
        [("_src_A|B|1", "_snk_A|B|1"), ("_src_A|B|2", "_snk_A|B|2")],
        [("_src_(A|B)|C|1", "_snk_(A|B)|C|1"), ("A", "C")],
    ],
)
def test_bac_labels_do_not_drop_distinct_pairs(pairs) -> None:
    results = _build_bac_results()
    data = results["steps"]["tm_placement"]["data"]
    data["baseline"]["flows"] = [
        {"source": src, "destination": dst, "placed": volume}
        for (src, dst), volume in zip(pairs, [100.0, 50.0], strict=True)
    ]
    data["flow_results"] = [{"occurrence_count": 1, "failure_id": "lost", "flows": []}]
    bac = compute_bac(results, "tm_placement")
    assert len(bac.per_flow) == 2
    assert sorted(flow.series.tolist() for flow in bac.per_flow.values()) == [
        [50.0, 0.0],
        [100.0, 0.0],
    ]


def test_bac_single_flow_no_per_flow() -> None:
    """Per-flow BAC is empty when there's only one flow."""
    res = {
        "steps": {
            "tm_placement": {
                "data": {
                    "baseline": {
                        "flows": [
                            {
                                "source": "A",
                                "destination": "B",
                                "placed": 100.0,
                                "demand": 100.0,
                            }
                        ]
                    },
                    "flow_results": [
                        {
                            "occurrence_count": 1,
                            "failure_id": "f1",
                            "flows": [
                                {
                                    "source": "A",
                                    "destination": "B",
                                    "placed": 50.0,
                                    "demand": 100.0,
                                }
                            ],
                        }
                    ],
                }
            }
        },
    }
    bac = compute_bac(res, step_name="tm_placement")
    assert len(bac.per_flow) == 0


def test_bac_occurrence_count_weighting() -> None:
    """Verify BAC correctly weights deduplicated patterns by occurrence_count.

    Setup: 10 MC iterations total.
    Pattern f1 (delivered=150): occurrence_count=7 (70% of iterations)
    Pattern f2 (delivered=180): occurrence_count=3 (30% of iterations)
    Baseline delivered=200 (offered).

    Series should have 11 entries: 1 baseline + 10 failure iterations.
    AUC = (1.0 + 0.75*7 + 0.9*3) / 11
        = (1.0 + 5.25 + 2.7) / 11 = 8.95 / 11
    """
    baseline_flows = [
        {"source": "A", "destination": "B", "placed": 200.0, "demand": 200.0},
    ]
    f1_flows = [
        {"source": "A", "destination": "B", "placed": 150.0, "demand": 200.0},
    ]
    f2_flows = [
        {"source": "A", "destination": "B", "placed": 180.0, "demand": 200.0},
    ]
    res = {
        "workflow": {"tm_placement": {"step_type": "TrafficMatrixPlacement"}},
        "steps": {
            "tm_placement": {
                "metadata": {"iterations": 10, "unique_patterns": 2},
                "data": {
                    "baseline": {
                        "occurrence_count": 1,
                        "failure_id": "baseline",
                        "flows": baseline_flows,
                    },
                    "flow_results": [
                        {"failure_id": "f1", "occurrence_count": 7, "flows": f1_flows},
                        {"failure_id": "f2", "occurrence_count": 3, "flows": f2_flows},
                    ],
                },
            }
        },
    }
    bac = compute_bac(res, step_name="tm_placement")

    assert len(bac.series) == 11
    assert bac.offered == 200.0

    expected_auc = (1.0 + 7 * 0.75 + 3 * 0.9) / 11.0
    assert np.isclose(bac.auc_normalized, expected_auc)

    vals = list(bac.series.values)
    assert vals.count(200.0) == 1  # baseline
    assert vals.count(150.0) == 7  # f1 expanded
    assert vals.count(180.0) == 3  # f2 expanded
