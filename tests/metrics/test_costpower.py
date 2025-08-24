from __future__ import annotations

import math

from metrics.costpower import CostPowerResult, compute_cost_power


def _fixture() -> dict:
    return {
        "steps": {
            "cost_power": {
                "data": {
                    "levels": {
                        "0": [{"capex_total": 100000.0, "power_total_watts": 5000.0}],
                        "1": [
                            {
                                "path": "metro1",
                                "platform_capex": 60000.0,
                                "optics_capex": 40000.0,
                                "capex_total": 100000.0,
                                "power_total_watts": 5000.0,
                            }
                        ],
                    }
                }
            }
        }
    }


def test_compute_cost_power_with_normalizations() -> None:
    res = _fixture()
    cp: CostPowerResult = compute_cost_power(
        res, offered_at_alpha1=1000.0, reliable_at_p999=800.0
    )
    assert cp.capex_total == 100000.0
    assert cp.power_total_w == 5000.0
    assert math.isclose(cp.per_offered_demand__usd_per_gbit, 100.0)
    assert math.isclose(cp.per_offered_demand__watt_per_gbit, 5.0)
    assert math.isclose(cp.per_reliable_p999__usd_per_gbit, 125.0)
    assert math.isclose(cp.per_reliable_p999__watt_per_gbit, 6.25)
    # Per-metro dataframe present and consistent
    assert list(cp.per_metro.columns) == [
        "path",
        "platform_capex",
        "optics_capex",
        "capex_total",
        "power_total_watts",
    ]
