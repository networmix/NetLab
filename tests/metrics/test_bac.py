from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from netlab.metrics.bac import compute_bac
from netlab.results import baseline_case, require_step, sampled_cases
from tests.result_samples import case, exported, placement, row


def _build_bac_results():
    baseline = [row("A", "B", 100, 100), row("A", "C", 50, 50), row("B", "C", 50, 50)]
    first = [row("A", "B", 80, 100), row("A", "C", 40, 50), row("B", "C", 30, 50)]
    second = [row("A", "B", 100, 100), row("A", "C", 40, 50), row("B", "C", 40, 50)]
    return exported(placement(baseline, [case(first), case(second, name="f2")]))


def test_compute_bac_core_stats():
    bac = compute_bac(_build_bac_results(), "tm_placement")
    assert bac.mode == "placement"
    assert bac.series.tolist() == [200, 150, 180]
    assert bac.offered == 200
    for p in (0.50, 0.90, 0.95, 0.99, 0.999, 0.9999):
        assert bac.quantiles_abs[p] == 180
        assert bac.quantiles_pct[p] == 0.9
    assert bac.availability_at_pct_of_offer[90] == 2 / 3
    for p in (95, 99, 99.9, 99.99):
        assert bac.availability_at_pct_of_offer[p] == 1 / 3
    for p in (90, 95, 99, 99.9, 99.99):
        assert bac.bw_at_probability_abs[p] == 150
        assert bac.bw_at_probability_pct[p] == 0.75
    assert bac.auc_normalized == pytest.approx((1 + 0.75 + 0.9) / 3)
    assert set(bac.per_flow) == {"A>B", "A>C", "B>C"}
    assert bac.per_flow["A>B"].series.tolist() == [100, 80, 100]
    assert bac.per_flow["A>C"].series.tolist() == [50, 40, 40]
    assert bac.per_flow["B>C"].series.tolist() == [50, 30, 40]
    assert (
        sum(flow.series for flow in bac.per_flow.values()).tolist()
        == bac.series.tolist()
    )
    assert bac.to_jsonable()["series"] == [200, 150, 180]


def test_bac_mode_detection_maxflow():
    results = _build_bac_results()
    results["steps"][0]["step_type"] = "MaxFlow"
    assert compute_bac(results, "tm_placement").mode == "maxflow"


def test_bac_requires_baseline_and_cases():
    results = _build_bac_results()
    step = results["steps"][0]
    del step["result"]["cases"][0]
    with pytest.raises(ValueError, match="exactly one baseline"):
        compute_bac(results, "tm_placement")
    step["result"]["cases"] = {}
    with pytest.raises(ValueError, match="cases must be a list"):
        compute_bac(results, "tm_placement")


def test_bac_per_flow_and_missing_pairs():
    results = _build_bac_results()
    step = require_step(results, "tm_placement")
    step["question"]["demands"].append(
        {
            "id": "tm|1",
            "source": "A",
            "target": "B",
            "mode": "pairwise",
            "group_mode": "flatten",
        }
    )
    baseline_case(step)["states"]["converged"]["rows"].append(
        row("A", "B", 20, 20, base_id="tm|1")
    )
    failures = sampled_cases(step)
    failures[0]["states"]["converged"]["rows"].append(
        row("A", "B", 10, 20, base_id="tm|1")
    )
    failures[1]["states"]["converged"]["rows"] = [row("B", "C", 40, 50)]
    bac = compute_bac(results, "tm_placement")
    assert bac.per_flow["A>B"].series.tolist() == [120, 90, 0]
    assert bac.per_flow["A>C"].series.tolist() == [50, 40, 0]
    assert (
        sum(flow.series for flow in bac.per_flow.values()).tolist()
        == bac.series.tolist()
    )


@pytest.mark.parametrize(
    "pairs",
    [
        [("A>B", "C"), ("A", "B>C")],
        [("long-site-name-" * 4 + "1", "B"), ("long-site-name-" * 4 + "2", "B")],
        [(["A", "B"], "C"), ("A", "C")],
    ],
)
def test_bac_labels_do_not_drop_distinct_endpoint_sets(pairs):
    records = [
        row(s, t, amount, amount, base_id=f"tm|{i}")
        for i, ((s, t), amount) in enumerate(zip(pairs, [100, 50], strict=True))
    ]
    r = exported(placement(records, [case([])]))
    for demand, (source, target) in zip(
        r["steps"][0]["question"]["demands"], pairs, strict=True
    ):
        if isinstance(source, list):
            demand.update(mode="combine", source="^(A|B)$", target=target)
    bac = compute_bac(r, "tm_placement")
    assert len(bac.per_flow) == 2
    assert sorted(flow.series.tolist() for flow in bac.per_flow.values()) == [
        [50, 0],
        [100, 0],
    ]


def test_single_flow_and_zero_baseline():
    result = exported(placement([row()], [case([row(delivered=50)])]))
    assert compute_bac(result, "tm_placement").per_flow == {}
    result = exported(placement([row(delivered=0)], [case([])]))
    bac = compute_bac(result, "tm_placement")
    assert bac.series.tolist() == [0, 0]
    assert np.isnan(bac.auc_normalized)


def test_occurrence_weighting_and_inert_node_rows():
    step = placement(
        [row(delivered=200, requested=200)],
        [
            case([row(delivered=150, requested=200)], 7),
            case([row(delivered=180, requested=200)], 3, name="f2"),
        ],
    )
    r = exported(step)
    bac = compute_bac(r, "tm_placement")
    assert bac.series.tolist() == [200] + [150] * 7 + [180] * 3
    assert bac.auc_normalized == pytest.approx(8.95 / 11)
    changed = deepcopy(r)
    for c in changed["steps"][0]["result"]["cases"]:
        c["states"]["converged"]["rows"].append(row("unused", "B", 0, 0))
    assert compute_bac(changed, "tm_placement").to_jsonable() == bac.to_jsonable()
