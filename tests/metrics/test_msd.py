from __future__ import annotations

import pytest

from netlab.metrics.msd import compute_alpha_star
from tests.result_samples import exported, msd


def test_alpha_exact_and_physical_base_total():
    out = compute_alpha_star(exported(msd(1.3, (10, 5.5)), q=0.25))
    assert out.source == "msd_baseline"
    assert out.alpha_star == 1.3
    assert out.base_total_demand == 15.5
    assert out.to_jsonable() == {
        "source": "msd_baseline",
        "alpha_star": 1.3,
        "base_total_demand": 15.5,
    }


def test_none_feasible_is_not_zero():
    with pytest.raises(ValueError, match="none_feasible"):
        compute_alpha_star(exported(msd(None, status="none_feasible")))
    assert compute_alpha_star(exported(msd(0))).alpha_star == 0
    assert compute_alpha_star(exported(msd(3, status="limit"))).alpha_star == 3


def test_alpha_requires_msd_and_declarations():
    with pytest.raises(ValueError, match="Missing required step"):
        compute_alpha_star(exported())
    step = msd()
    step["question"]["demands"] = []
    with pytest.raises(ValueError, match="nonempty list"):
        compute_alpha_star(exported(step))
