from __future__ import annotations

from pathlib import Path

import numpy as np

from metrics.iterops import compute_iter_ops


def _make_results(spf_base: float, spf_fails: list[float]) -> dict:
    fr = []
    # baseline
    fr.append(
        {
            "failure_id": "baseline",
            "flows": [
                {
                    "source": "a/b",
                    "destination": "c/d",
                    "demand": 1.0,
                    "placed": 1.0,
                }
            ],
            "data": {"iteration_metrics": {"spf_calls_total": spf_base}},
        }
    )
    # failures
    for i, v in enumerate(spf_fails):
        fr.append(
            {
                "failure_id": f"f{i}",
                "flows": [
                    {
                        "source": "a/b",
                        "destination": "c/d",
                        "demand": 1.0,
                        "placed": 1.0,
                    }
                ],
                "data": {"iteration_metrics": {"spf_calls_total": v}},
            }
        )
    return {
        "steps": {
            "tm_placement": {
                "metadata": {"baseline": True},
                "data": {"flow_results": fr},
            }
        }
    }


def test_iterops_basic_totals_and_avg(tmp_path: Path) -> None:
    res = _make_results(12.0, [2.0, 3.0, 5.0])
    it = compute_iter_ops(res)
    js = it.to_jsonable()
    assert js["baseline"]["spf_calls_total"] == 12.0
    assert js["failures"]["spf_calls_total"] == 10.0
    assert js["totals_all"]["spf_calls_total"] == 22.0
    # per-iteration averages over failures only
    ser = it.flat_series()
    avg = float(ser["spf_calls_total_per_iter"])  # (2+3+5)/3
    assert np.isclose(avg, (2.0 + 3.0 + 5.0) / 3.0)


def test_iterops_missing_fields_are_zero(tmp_path: Path) -> None:
    # When keys are missing, they should sum to 0.0 and not crash
    fr = [
        {"failure_id": "baseline", "flows": [], "data": {"iteration_metrics": {}}},
        {"failure_id": "f1", "flows": [], "data": {}},
    ]
    res = {
        "steps": {
            "tm_placement": {
                "metadata": {"baseline": True},
                "data": {"flow_results": fr},
            }
        }
    }
    it = compute_iter_ops(res)
    js = it.to_jsonable()
    assert js["baseline"]["spf_calls_total"] == 0.0
    assert js["failures"]["spf_calls_total"] == 0.0
    assert js["totals_all"]["spf_calls_total"] == 0.0
