"""Reported TE tails and mean-cost excess against hand-derived networks."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from netlab.metrics.latency import compute_latency_stretch
from netlab.results import baseline_case, require_step, rows, sampled_cases
from netlab.simulation import simulate
from tests.result_samples import case, exported, placement, row


def marginal_row(source="S", target="T", *, bins, total_cost):
    amount = sum(bins.values())
    record = row(source, target, amount, amount)
    record["cost_distribution"] = {
        "kind": "marginal_curve",
        "cost_domain": "physical_network",
        "bins": [
            {"marginal_cost": cost, "units": units} for cost, units in bins.items()
        ],
    }
    record["result"]["total_cost"] = total_cost
    return record


def cancellation_document():
    # X-A-B-Y occupies A-B first. When X fails, the residual flow sends one
    # unit at marginal cost 3 then one at 5, leaving S-A-T and S-B-T at 4 each.
    links = [
        ("S", "A", 1),
        ("A", "B", 1),
        ("B", "T", 1),
        ("S", "B", 3),
        ("A", "T", 3),
        ("X", "A", 1),
        ("B", "Y", 1),
    ]
    doc = {
        "seed": 42,
        "network": {
            "nodes": {
                name: {"attrs": {"fail": name == "X"}}
                for name in ("S", "A", "B", "T", "X", "Y")
            },
            "links": [
                dict(source=s, target=t, cost=cost, capacity=1) for s, t, cost in links
            ],
        },
        "demands": {},
        "failures": {
            "fail_X": {
                "modes": [
                    {
                        "weight": 1,
                        "rules": [
                            {
                                "scope": "node",
                                "mode": "all",
                                "match": {
                                    "conditions": [
                                        {"attr": "fail", "op": "==", "value": True}
                                    ]
                                },
                            }
                        ],
                    }
                ]
            }
        },
        "workflow": [],
    }
    for name, routing in (
        ("optimized", {"preset": "TE_MAXFLOW_UNLIM"}),
        (
            "equal",
            {
                "model": {
                    "routing": {"type": "centralized_te", "source_allocation": "equal"}
                }
            },
        ),
    ):
        doc["demands"][name] = [
            dict(source="^X$", target="^Y$", volume=1, priority=0, **routing),
            dict(source="^S$", target="^T$", volume=2, priority=1, **routing),
        ]
        doc["workflow"].append(
            dict(
                type="TrafficMatrixPlacement",
                name=name,
                demand_set=name,
                workers=1,
                failure_policy="fail_X",
                iterations=1,
                outputs={"cost_distribution": True, "edge_flows": True},
            )
        )
    return doc


def test_real_two_path_cancellation_uses_mean_wes_and_curve_tails():
    outcome = simulate(yaml.safe_dump(cancellation_document()))
    assert outcome.success, outcome.error
    results = outcome.results
    optimized = require_step(results, "optimized")
    equal = require_step(results, "equal")
    q = results["units_report"]["q"]
    flows = []
    for step in (optimized, equal):
        failure = sampled_cases(step)[0]
        flow = next(r for r in rows(failure) if r["endpoints"]["sources"] == ["S"])
        assert flow["result"]["delivered_units"] * q == 2
        assert flow["result"]["total_cost"] * q == 8
        flows.append(flow)
    assert flows[0]["result"]["edge_flows"] == flows[1]["result"]["edge_flows"]
    assert [
        (b["marginal_cost"], b["units"] * q)
        for b in flows[0]["cost_distribution"]["bins"]
    ] == [(3, 1), (5, 1)]
    assert [
        (b["physical_path_cost"], b["delivered_units"] * q)
        for b in flows[1]["cost_distribution"]["bins"]
    ] == [(4, 2)]
    curve = compute_latency_stretch(results, "optimized")
    actual_paths = compute_latency_stretch(results, "equal")
    baseline = dict(
        p50=1,
        p95=1,
        p99=1,
        SLO_1_2=1,
        SLO_1_5=1,
        best_path_share=1,
        WES=0,
        reference_coverage=1,
    )
    failure = dict(
        p50=0.75,
        p95=1.25,
        p99=1.25,
        SLO_1_2=0.5,
        SLO_1_5=1,
        best_path_share=0,
        WES=0,
        reference_coverage=1,
    )
    assert curve.baseline == pytest.approx(baseline)
    assert actual_paths.baseline == pytest.approx(baseline)
    assert actual_paths.failures == pytest.approx(baseline)
    assert curve.failures == pytest.approx(failure)
    assert curve.derived == pytest.approx(
        dict(
            TD95=1.25,
            TD99=1.25,
            SLO_1_2_drop=0.5,
            SLO_1_5_drop=0,
            best_path_share_drop=1,
            WES_delta=0,
        )
    )
    assert curve.per_iteration is not None
    for key, value in failure.items():
        assert curve.per_iteration[key] == pytest.approx([value])


@pytest.mark.parametrize("second_kind", ["marginal_curve", "delivered_by_path_cost"])
def test_wes_clips_each_row_and_keeps_delivered_path_bin_excess(second_kind):
    baseline = [row("A", "B", 8, 8, costs={4: 8}), row("C", "D", 8, 8, costs={4: 8})]
    first = marginal_row(
        "A", "B", bins={3: 4, 7: 4}, total_cost=40
    )  # Mean5/ref4 -> 1/4.
    if second_kind == "marginal_curve":
        second = marginal_row(
            "C", "D", bins={1: 4, 5: 4}, total_cost=24
        )  # Mean3/ref4 ->0.
        expected = 1 / 8  # Equal row volumes; clipping the aggregate mean would give0.
    else:
        second = row("C", "D", 8, 8, costs={1: 4, 5: 4})  # Bin WES1/8 despite mean3.
        expected = 3 / 16  # (1/4+1/8)/2; this row must not use mean-cost clipping.
    results = exported(placement(baseline, [case([first, second])]), q=0.25)
    metrics = compute_latency_stretch(results)
    assert metrics.failures["WES"] == pytest.approx(expected)
    assert metrics.failures["reference_coverage"] == 1
    assert metrics.failures["p50"] == 0.75


def test_nonunit_q_total_cost_and_delivered_volume_are_scaled_together():
    baseline = marginal_row(bins={4: 8}, total_cost=32)
    failure = marginal_row(bins={3: 4, 5: 4}, total_cost=32)
    results = exported(placement([baseline], [case([failure])]), q=0.25)
    out = compute_latency_stretch(results)
    assert out.failures["WES"] == 0  # Physical total cost8 / delivery2 / reference4 -1.
    assert out.failures["p50"] == 0.75


@pytest.mark.parametrize(
    "bad", [None, -1, float("nan"), float("inf"), True, 32.0, "32", "missing"]
)
def test_marginal_total_cost_must_be_a_whole_nonnegative_quantity(bad):
    baseline = marginal_row(bins={4: 8}, total_cost=32)
    failure = marginal_row(bins={3: 4, 5: 4}, total_cost=32)
    if bad == "missing":
        del failure["result"]["total_cost"]
    else:
        failure["result"]["total_cost"] = bad
    results = exported(placement([baseline], [case([failure])]), q=0.25)
    with pytest.raises(
        ValueError, match="marginal_curve total_cost must be a nonnegative integer"
    ):
        compute_latency_stretch(results)
    failure["result"]["total_cost"] = 32
    assert compute_latency_stretch(results).failures["WES"] == 0


def test_empty_marginal_report_has_no_division_by_zero():
    baseline = marginal_row(bins={4: 8}, total_cost=32)
    results = exported(
        placement([baseline], [case([marginal_row(bins={}, total_cost=0)])])
    )
    out = compute_latency_stretch(results)
    assert out.failures["reference_coverage"] != out.failures["reference_coverage"]
    assert np.isnan(out.failures["WES"])


def test_mini_dcbb_matches_physical_corridor_costs_for_both_report_kinds():
    doc = yaml.safe_load(
        (Path(__file__).parents[1] / "data/mini_dcbb.yaml").read_text()
    )
    curve_outcome = simulate(yaml.safe_dump(doc))
    assert curve_outcome.success, curve_outcome.error
    equal_doc = deepcopy(doc)
    for demand_set in equal_doc["demands"].values():
        for demand in demand_set:
            del demand["preset"]
            demand["model"] = {
                "routing": {"type": "centralized_te", "source_allocation": "equal"}
            }
    paths_outcome = simulate(yaml.safe_dump(equal_doc))
    assert paths_outcome.success, paths_outcome.error
    # The two disjoint corridors per direction carry200@112 and100@212.
    baseline = dict(
        p50=1,
        p95=53 / 28,
        p99=53 / 28,
        SLO_1_2=2 / 3,
        SLO_1_5=2 / 3,
        best_path_share=2 / 3,
        WES=25 / 84,
        reference_coverage=1,
    )
    lh_failure = dict(
        p50=81 / 56,
        p95=81 / 56,
        p99=81 / 56,
        SLO_1_2=1 / 2,
        SLO_1_5=1 / 2,
        best_path_share=1 / 2,
        WES=25 / 56,
        reference_coverage=1,
    )
    lh_derived = dict(
        TD95=81 / 106,
        TD99=81 / 106,
        SLO_1_2_drop=1 / 6,
        SLO_1_5_drop=1 / 6,
        best_path_share_drop=1 / 6,
        WES_delta=25 / 168,
    )
    for outcome, marginal in ((curve_outcome, True), (paths_outcome, False)):
        for name in ("tm_lh_path", "tm_1x_bb"):
            metrics = compute_latency_stretch(outcome.results, name)
            assert metrics.baseline == pytest.approx(baseline)
            assert metrics.failures == pytest.approx(
                lh_failure if name == "tm_lh_path" else baseline
            )
            assert metrics.derived == pytest.approx(
                lh_derived
                if name == "tm_lh_path"
                else dict(
                    TD95=1,
                    TD99=1,
                    SLO_1_2_drop=0,
                    SLO_1_5_drop=0,
                    best_path_share_drop=0,
                    WES_delta=0,
                )
            )
            step = require_step(outcome.results, name)
            q = outcome.results["units_report"]["q"]
            for sample in [baseline_case(step), *sampled_cases(step)]:
                pattern = sample["case"]["pattern"]
                expected = {112: 200, 212: 100}
                if pattern["excluded_nodes"]:
                    expected = {112: 100, 212: 50}
                elif pattern["excluded_links"]:
                    suffix = {
                        link.rsplit("|", 1)[-1] for link in pattern["excluded_links"]
                    }
                    expected = {212: 100} if suffix == {"0"} else {112: 200}
                for flow in rows(sample):
                    report = flow["cost_distribution"]
                    assert report["kind"] == (
                        "marginal_curve" if marginal else "delivered_by_path_cost"
                    )
                    actual = {
                        b["marginal_cost" if marginal else "physical_path_cost"]: b[
                            "units" if marginal else "delivered_units"
                        ]
                        * q
                        for b in report["bins"]
                    }
                    assert actual == expected
            assert metrics.per_iteration is not None
            count = 10 if name == "tm_lh_path" else 20
            for values in metrics.per_iteration.values():
                assert len(values) == count
