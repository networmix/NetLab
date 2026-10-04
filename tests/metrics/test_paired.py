from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from netlab.metrics.paired import holm_adjust, paired_t


def test_matches_scipy_and_has_correct_sign():
    a = np.array([2.0, 3.0, 5.0, 4.0, 7.0])
    b = np.array([1.0, 3.0, 2.0, 5.0, 2.0])
    expected = stats.ttest_rel(a, b)
    actual = paired_t(a, b)
    assert actual["t_stat"] == pytest.approx(expected.statistic)
    assert actual["p"] == pytest.approx(expected.pvalue)
    assert actual["ci_low"] == pytest.approx(expected.confidence_interval().low)
    assert paired_t(np.zeros(5), np.ones(5))["t_stat"] == -float("inf")


def test_holm_excludes_missing_tests():
    adjusted = holm_adjust([("a", 0.01), ("b", 0.04), ("missing", float("nan"))])
    assert adjusted["a"] == 0.02
    assert adjusted["b"] == 0.04
    assert np.isnan(adjusted["missing"])


def test_rejects_unpaired_shapes():
    with pytest.raises(ValueError):
        paired_t(np.ones(3), np.ones(4))


def test_inference_does_not_change_with_measurement_units():
    a = np.array([1.0, 2.0, 4.0, 8.0])
    b = np.zeros(4)
    original = paired_t(a, b)
    scaled = paired_t(a * 1e-15, b)
    assert scaled["p"] == pytest.approx(original["p"])
    assert scaled["t_stat"] == pytest.approx(original["t_stat"])
    assert not scaled["deterministic"]
