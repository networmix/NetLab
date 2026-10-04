from __future__ import annotations

import math

import pytest

from netlab.metrics.msd import AlphaResult, compute_alpha_star


def test_alpha_from_msd_baseline_with_base_total() -> None:
    res = {
        "steps": {
            "msd_baseline": {
                "data": {
                    "alpha_star": 1.3,
                    "base_demands": [
                        {"volume": 10.0},
                        {"volume": 5.5},
                    ],
                }
            }
        }
    }
    out: AlphaResult = compute_alpha_star(res)
    assert out.source == "msd_baseline"
    assert math.isclose(out.alpha_star, 1.3)
    assert math.isclose(out.base_total_demand, 15.5)


@pytest.mark.parametrize(
    "data", [{}, {"alpha_star": 1.0}, {"alpha_star": 1.0, "base_demands": [{}]}]
)
def test_alpha_requires_current_msd_data(data: dict) -> None:
    with pytest.raises(ValueError, match="requires alpha_star and base_demands"):
        compute_alpha_star({"steps": {"msd_baseline": {"data": data}}})


def test_alpha_requires_msd_baseline() -> None:
    with pytest.raises(ValueError, match="requires alpha_star and base_demands"):
        compute_alpha_star({"steps": {}})
