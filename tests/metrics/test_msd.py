from __future__ import annotations

import math

from metrics.msd import AlphaResult, compute_alpha_star


def test_alpha_from_msd_baseline_with_base_total() -> None:
    res = {
        "steps": {
            "msd_baseline": {
                "data": {
                    "alpha_star": 1.3,
                    "base_demands": [
                        {"demand": 10.0},
                        {"demand": 5.5},
                    ],
                }
            }
        }
    }
    out: AlphaResult = compute_alpha_star(res)
    assert out.source == "msd_baseline"
    assert math.isclose(out.alpha_star, 1.3)
    assert math.isclose(out.base_total_demand, 15.5)


def test_alpha_from_probes_when_no_msd_alpha() -> None:
    # No alpha_star in msd_baseline, but have probes with feasible entries
    res = {
        "steps": {
            "msd_baseline": {"data": {}},
            "tm_placement": {
                "metadata": {
                    "probes": [
                        {"alpha": 1.0, "feasible": True},
                        {"alpha": 1.2, "feasible": True},
                        {"alpha": 1.1, "feasible": False},
                    ]
                }
            },
        }
    }
    out = compute_alpha_star(res)
    assert out.source == "probes"
    assert math.isclose(out.alpha_star, 1.2)


def test_alpha_unknown_without_signals() -> None:
    out = compute_alpha_star({"steps": {}})
    assert out.source == "unknown"
    assert math.isnan(out.alpha_star)
