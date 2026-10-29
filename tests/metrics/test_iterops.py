from __future__ import annotations

import pytest

from netlab.metrics.iterops import compute_iter_ops
from tests.result_samples import case, exported, placement, timing_record


def _iterops_fixture():
    return exported(placement([], [case(count=3), case(count=2, name="f2")]))


def test_compute_iter_ops_counts_and_serialization():
    ops = compute_iter_ops(_iterops_fixture(), timing_record(_iterops_fixture()))
    assert ops.failures_count == 5
    assert ops.unique_patterns == 2
    assert ops.total_iterations_count == 6
    assert {
        key: ops.to_jsonable()[key]
        for key in ("iters_fail", "iters_total", "unique_patterns")
    } == {"iters_fail": 5, "iters_total": 6, "unique_patterns": 2}
    assert ops.to_jsonable()["tm_duration_total_sec"] == 10
    assert ops.to_jsonable()["tm_duration_per_iter_sec"] == pytest.approx(10 / 6)
    assert ops.flat_series()["tm_duration_total_sec"] == 10
    assert ops.flat_series()["tm_duration_per_iter_sec"] == pytest.approx(10 / 6)
    assert ops.flat_series()["iters_fail"] == 5


def test_compute_iter_ops_requires_count():
    r = _iterops_fixture()
    del r["steps"][0]["result"]["cases"][1]["case"]["occurrences"]
    with pytest.raises(ValueError, match="case.occurrences"):
        compute_iter_ops(r, timing_record(r))


def test_compute_iter_ops_requires_baseline():
    r = _iterops_fixture()
    del r["steps"][0]["result"]["cases"][0]
    with pytest.raises(ValueError, match="baseline case required"):
        compute_iter_ops(r, timing_record(r))


def test_sample_count_mismatch_is_refused():
    r = _iterops_fixture()
    r["steps"][0]["result"]["metadata"]["sampling"]["iterations"] += 1
    with pytest.raises(ValueError, match="sampling.iterations"):
        compute_iter_ops(r, timing_record(r))


@pytest.mark.parametrize("value", [None, -1, float("nan"), float("inf"), True, "1"])
def test_missing_or_invalid_step_time_is_refused(value):
    results = _iterops_fixture()
    record = timing_record(results)
    record["step_durations_sec"]["tm_placement"] = value
    with pytest.raises(ValueError, match="duration for step 'tm_placement'"):
        compute_iter_ops(results, record)
    with pytest.raises(ValueError, match="duration for step 'tm_placement'"):
        compute_iter_ops(results, {})
    record["step_durations_sec"]["tm_placement"] = 0
    ops = compute_iter_ops(results, record)
    assert ops.total_duration_sec == ops.per_iter_duration_sec == 0
