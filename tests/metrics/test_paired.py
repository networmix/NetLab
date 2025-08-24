from __future__ import annotations

import math

import numpy as np

from metrics.paired import paired_normal_test_holm, paired_wilcoxon_sign


def test_paired_normal_test_holm_known_values() -> None:
    # Construct small dataset with known mean and std
    a = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Differences d = [1,1,1,1,1] => mean=1, sd=0; z=inf; p=0
    res = paired_normal_test_holm(a, b)
    assert math.isinf(res["t_stat"]) or res["t_stat"] > 1e6
    assert res["p"] == 0.0 and res["p_adj"] == 0.0
    assert res["n"] == 5


def test_paired_wilcoxon_sign_symmetric_case() -> None:
    # Differences: [1, -1, 1, -1, 1, -1, 1, -1, 1, -1] => equal positives and negatives
    a = np.array([2, 0, 2, 0, 2, 0, 2, 0, 2, 0], dtype=float)
    b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    res = paired_wilcoxon_sign(a, b)
    assert res["n"] == 10
    assert math.isclose(res["z"], 0.0, abs_tol=1e-9)
    assert math.isclose(res["p"], 1.0, rel_tol=0.0, abs_tol=1e-12)
