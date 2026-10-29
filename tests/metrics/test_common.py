"""Absolute units, populations and structural boundaries shared by all metrics."""

from copy import deepcopy

import pytest

from netlab.metrics.bac import compute_bac
from netlab.metrics.common import baseline_demand_map
from netlab.metrics.iterops import compute_iter_ops
from netlab.metrics_failure import extract_failure_ratios
from netlab.results import (
    baseline_case,
    converged_state,
    expand_cases,
    feasible_alpha,
    physical_bandwidth,
    require_step,
    sampled_cases,
    total_bandwidth,
)
from tests.result_samples import case, exported, msd, placement, row, timing_record


@pytest.mark.parametrize("count", [None, -1, True, "1", 1.5])
def test_case_count_requires_nonnegative_integer(count):
    record = case(count=count)
    with pytest.raises(
        ValueError, match="case.occurrences must be a nonnegative integer"
    ):
        expand_cases([record])
    record["case"]["occurrences"] = 0
    assert expand_cases([record]) == []


def test_counts_repeat_identity_and_allow_empty():
    first, second = case(count=3), case(count=2, name="f2")
    assert expand_cases([first, second]) == [first] * 3 + [second] * 2
    assert expand_cases([]) == []


def test_absolute_units_and_wide_sum():
    r = exported(placement([row(delivered=9600, requested=9600)]), q=1 / 16)
    assert total_bandwidth(r, baseline_case(r["steps"][0])) == 600
    assert baseline_demand_map(r) == {("A", "B"): 600}
    large = 2**63 - 1
    r = exported(placement([row(delivered=large, requested=large)] * 2))
    assert total_bandwidth(r, baseline_case(r["steps"][0])) == 2**64 - 2


@pytest.mark.parametrize("q", [0, -1, True, float("nan"), float("inf")])
def test_q_is_positive_finite(q):
    with pytest.raises(ValueError, match="positive and finite"):
        physical_bandwidth(exported(q=q), 1)
    assert physical_bandwidth(exported(q=0.25), 16) == 4


def test_sampled_baseline_is_not_the_extra_reference():
    # 71 no-failure draws, 25 at half capacity, 4 at zero: sum=83.5.
    step = placement(
        [row(delivered=9600, requested=9600)],
        [
            case([row(delivered=4800, requested=9600)], 25),
            case([row(delivered=0, requested=9600)], 4, name="outage"),
        ],
        baseline_count=71,
    )
    r = exported(step, q=1 / 16)
    bac = compute_bac(r, "tm_placement")
    assert bac.offered == 600
    assert len(bac.series) == 101 and list(bac.series).count(600) == 72
    assert bac.auc_normalized == pytest.approx(169 / 202)
    assert extract_failure_ratios(r)["tm_placement"] == [1] * 71 + [0.5] * 25 + [0] * 4
    ops = compute_iter_ops(r, timing_record(r))
    assert (ops.failures_count, ops.total_iterations_count, ops.unique_patterns) == (
        100,
        101,
        3,
    )


def test_no_policy_has_one_reference_and_no_failure_samples():
    step = placement([row()], baseline_count=5, policy=None)
    r = exported(step)
    assert sampled_cases(step) == []
    assert compute_bac(r, "tm_placement").series.tolist() == [100]
    ops = compute_iter_ops(r, timing_record(r))
    assert (ops.failures_count, ops.total_iterations_count, ops.unique_patterns) == (
        0,
        1,
        0,
    )
    assert extract_failure_ratios(r) == {}


def test_inline_policy_and_all_empty_samples():
    step = placement([row()], baseline_count=5, policy=None)
    step["question"]["failures"]["inline_policy"] = {"modes": []}
    assert len(expand_cases(sampled_cases(step))) == 5
    assert compute_bac(exported(step), "tm_placement").series.tolist() == [100] * 6


def test_named_steps_and_converged_state():
    step = placement([row(delivered=4, requested=4)])
    base = baseline_case(step)
    base["states"]["stale"] = {"rows": [row(delivered=0, requested=4)]}
    r = exported(msd(), step)
    assert total_bandwidth(r, base) == 4
    r["steps"].reverse()
    assert require_step(r, "msd_baseline")["name"] == "msd_baseline"
    with pytest.raises(ValueError, match="Missing required step"):
        require_step(r, "absent")
    r["steps"].append(deepcopy(step))
    with pytest.raises(ValueError, match="unique names"):
        require_step(r, "tm_placement")
    del base["states"]["converged"]
    with pytest.raises(ValueError, match="converged state required"):
        converged_state(base)


def test_failure_is_not_zero_and_capacity_totals_are_null():
    c = case(capacity=True)
    assert converged_state(c)["totals"] is None
    c["states"]["converged"]["failure"] = {"reason": "no_solution"}
    with pytest.raises(ValueError, match="no_solution"):
        total_bandwidth(exported(), c)


def test_alpha_none_is_not_zero():
    assert feasible_alpha(msd(0)) == 0
    with pytest.raises(ValueError, match="none_feasible"):
        feasible_alpha(msd(None, status="none_feasible"))


def test_self_capacity_rows_are_inert():
    from netlab.results import pair_totals

    c = case([row(delivered=160, requested=160)], capacity=True)
    r = exported(placement([], capacity=True), q=1 / 16)
    expected = total_bandwidth(r, c)
    c["states"]["converged"]["rows"].append(
        {"kind": "self", "id": "self-0", "node": "A", "capacity_units": 0}
    )
    assert total_bandwidth(r, c) == expected == 10
    assert pair_totals(r, r["steps"][0], c, "delivered_units") == {("A", "B"): 10}
    c["states"]["converged"]["rows"][-1]["capacity_units"] = 1
    with pytest.raises(ValueError, match="self capacity must be zero"):
        total_bandwidth(r, c)
