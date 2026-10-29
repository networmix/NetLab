"""Small native JSON samples; all supplied bandwidth amounts are whole units."""

from copy import deepcopy
from fractions import Fraction


def row(
    source="A", target="B", delivered=100, requested=100, *, costs=None, base_id="tm|0"
):
    return {
        "kind": "placement",
        "id": f"{base_id}|0",
        "base_id": base_id,
        "endpoints": {
            "sources": [source] if isinstance(source, str) else source,
            "destinations": [target] if isinstance(target, str) else target,
        },
        "reference_request_units": requested,
        "reference_shortfall_units": requested - delivered,
        "result": {
            "delivered_units": delivered,
            "requested_units": requested,
            "unadmitted_units": requested - delivered,
            "source_drop_units": 0,
            "edge_drop_units": 0,
        },
        **(
            {
                "cost_distribution": {
                    "kind": "delivered_by_path_cost",
                    "cost_domain": "physical_network",
                    "bins": [
                        {"physical_path_cost": cost, "delivered_units": amount}
                        for cost, amount in costs.items()
                    ],
                }
            }
            if costs is not None
            else {}
        ),
    }


def case(
    records=(),
    count=1,
    *,
    kind="failure",
    name="f1",
    excluded_nodes=(),
    excluded_links=(),
    capacity=False,
):
    request = sum(r["reference_request_units"] for r in records)
    delivered = sum(r["result"]["delivered_units"] for r in records)
    ratio = Fraction(delivered, request) if request else Fraction(1)
    return {
        "case": {
            "id": name,
            "kind": kind,
            "occurrences": count,
            "pattern": {
                "excluded_nodes": list(excluded_nodes),
                "excluded_links": list(excluded_links),
            },
        },
        "states": {
            "converged": {
                "rows": list(records),
                "totals": None
                if capacity
                else {
                    "reference_request_units": request,
                    "delivered_units": delivered,
                    "reference_shortfall_units": request - delivered,
                    "unadmitted_units": request - delivered,
                    "source_drop_units": 0,
                    "edge_drop_units": 0,
                    "delivery_ratio": {
                        "numerator": ratio.numerator,
                        "denominator": ratio.denominator,
                    },
                },
            }
        },
    }


def placement(
    baseline,
    samples=(),
    *,
    name="tm_placement",
    baseline_count=0,
    policy="selected",
    capacity=False,
):
    base = case(
        baseline, baseline_count, kind="baseline", name="baseline", capacity=capacity
    )
    records = [base, *samples]
    return {
        "name": name,
        "step_type": "MaxFlow" if capacity else "TrafficMatrixPlacement",
        "question": {
            "failures": {"policy_name": policy, "inline_policy": None},
            "demands": [
                {
                    "id": base_id,
                    "source": r["endpoints"]["sources"][0],
                    "target": r["endpoints"]["destinations"][0],
                    "mode": "pairwise",
                    "group_mode": "flatten",
                }
                for base_id, r in {r["base_id"]: r for r in baseline}.items()
            ],
        },
        "result": {
            "cases": records,
            "accounting": [
                {
                    "base_id": r["base_id"],
                    "declared_units": r["reference_request_units"],
                    "resolved_units": r["reference_request_units"],
                    "discarded_units": 0,
                    "discards": [],
                }
                for r in baseline
            ],
            "metadata": {
                "sampling": {
                    "iterations": sum(c["case"]["occurrences"] for c in records)
                }
            },
        },
    }


def msd(alpha=1, volumes=(100,), *, status="feasible", name="msd_baseline"):
    exact = Fraction(str(alpha)) if alpha is not None else None
    return {
        "name": name,
        "step_type": "MaximumSupportedDemand",
        "question": {
            "demand_set": "tm",
            "demands": [
                {
                    "id": f"tm|{i}",
                    "source": "A",
                    "target": "B",
                    "physical_volume": v,
                    "mode": "pairwise",
                    "group_mode": "flatten",
                }
                for i, v in enumerate(volumes)
            ],
        },
        "result": {
            "search_status": status,
            "alpha_feasible": {
                "value": float(exact),
                "numerator": exact.numerator,
                "denominator": exact.denominator,
            }
            if exact is not None
            else None,
        },
    }


def exported(*steps, q=1, seed=42):
    steps = list(steps)
    msd_step = next(
        (s for s in steps if s["step_type"] == "MaximumSupportedDemand"), None
    )
    if msd_step is not None:
        for step in steps:
            if step["step_type"] == "TrafficMatrixPlacement":
                step["question"].update(deepcopy(msd_step["question"]))
                step["result"]["alpha"] = deepcopy(msd_step["result"]["alpha_feasible"])
    return {"metadata": {"seed": seed}, "units_report": {"q": q}, "steps": steps}


def timing_record(results, duration=10.0):
    """Explicit hand-set timing input for synthetic result tests."""
    return {
        "status": "success",
        "step_durations_sec": {step["name"]: duration for step in results["steps"]},
    }
