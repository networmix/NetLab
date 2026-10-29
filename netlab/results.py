"""Read exported NetGraph results in physical bandwidth and sampled cases."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from typing import Iterator

NO_FEASIBLE_ALPHA = "none_feasible: no tested feasible alpha"


def steps(results: dict) -> list[dict]:
    """Read the ordered native step list, requiring unique names."""
    records = results.get("steps")
    if not isinstance(records, list):
        raise ValueError("results.steps must be a list")
    names = [record["name"] for record in records]
    if len(set(names)) != len(names):
        raise ValueError("results.steps requires unique names")
    return records


def find_step(results: dict, name: str) -> dict | None:
    """Return the named step, or None when it is absent."""
    return next((record for record in steps(results) if record["name"] == name), None)


def require_step(results: dict, name: str) -> dict:
    """Read a required named step, refusing its absence."""
    record = find_step(results, name)
    if record is None:
        raise ValueError(f"Missing required step in results: {name}")
    return record


def physical_bandwidth(results: dict, units: int) -> float:
    """Convert nonnegative whole bandwidth units to physical bandwidth exactly once using q."""
    if type(units) is not int or units < 0:
        raise ValueError("bandwidth units must be a nonnegative integer")
    q = results["units_report"]["q"]
    if (
        isinstance(q, bool)
        or not isinstance(q, (int, float))
        or not math.isfinite(q)
        or q <= 0
    ):
        raise ValueError("units_report.q must be positive and finite")
    return units * q


def case_occurrences(case: dict) -> int:
    """Read the nonnegative number of sampled draws represented by a case."""
    count = case["case"].get("occurrences")
    if type(count) is not int or count < 0:
        raise ValueError("case.occurrences must be a nonnegative integer")
    return count


def cases(step: dict) -> list[dict]:
    """Read the native baseline/failure cases and validate their occurrence counts."""
    records = step["result"].get("cases")
    if not isinstance(records, list):
        raise ValueError(f"{step['name']}: result.cases must be a list")
    for case in records:
        if case["case"]["kind"] not in {"baseline", "failure"}:
            raise ValueError("Unknown case kind")
        case_occurrences(case)
    return records


def baseline_case(step: dict) -> dict:
    """Require exactly one unsampled-reference baseline case in a step."""
    baseline = [case for case in cases(step) if case["case"]["kind"] == "baseline"]
    if len(baseline) != 1:
        raise ValueError(f"{step['name']}: exactly one baseline case required")
    return baseline[0]


def has_failure_policy(step: dict) -> bool:
    """Report whether the question declares a named or inline failure policy."""
    failures = step["question"]["failures"]
    return failures["policy_name"] is not None or failures["inline_policy"] is not None


def sampled_cases(step: dict) -> list[dict]:
    """With a policy include every positive-weight case, including empty baseline draws; otherwise return none."""
    records = cases(step)
    if not has_failure_policy(step):
        return []
    return [case for case in records if case_occurrences(case) > 0]


def expand_cases(records: list[dict]) -> list[dict]:
    """Repeat case references by their occurrence counts so each sampled draw has equal weight."""
    return [case for case in records for _ in range(case_occurrences(case))]


def converged_state(case: dict) -> dict:
    """Require a successful converged state containing native rows."""
    state = case["states"].get("converged")
    if not isinstance(state, dict):
        raise ValueError(f"{case['case']['id']}: converged state required")
    if state.get("failure") is not None:
        raise ValueError(
            f"{case['case']['id']}: converged state failed: {state['failure']}"
        )
    if not isinstance(state.get("rows"), list):
        raise ValueError("converged.rows must be a list")
    return state


def rows(case: dict) -> list[dict]:
    """Read converged rows, omitting inert zero self capacities and refusing nonzero self capacities."""
    records = converged_state(case)["rows"]
    for row in records:
        if row["kind"] == "self" and row["capacity_units"] != 0:
            raise ValueError("self capacity must be zero")
    return [row for row in records if row["kind"] != "self"]


def row_units(row: dict, field: str) -> int:
    """Read a required nonnegative whole-unit row quantity without converting its units."""
    value = row["result"][field] if field == "delivered_units" else row[field]
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def total_bandwidth(results: dict, case: dict, field: str = "delivered_units") -> float:
    """Sum a case quantity in whole units before converting the total once."""
    return physical_bandwidth(results, sum(row_units(row, field) for row in rows(case)))


EndpointKey = tuple[tuple[str, ...], tuple[str, ...]]


def row_endpoints(row: dict) -> EndpointKey:
    """Return complete sorted source and destination rosters without splitting groups."""
    endpoints = row["endpoints"]
    return tuple(sorted(endpoints["sources"])), tuple(sorted(endpoints["destinations"]))


def endpoint_totals(results: dict, case: dict, field: str) -> dict[EndpointKey, float]:
    """Sum priority classes by complete physical endpoint sets and convert once."""
    totals: dict[EndpointKey, int] = {}
    for row in rows(case):
        source, target = row_endpoints(row)
        if (
            not source
            or not target
            or (len(source) == len(target) == 1 and source == target)
        ):
            continue
        key = (source, target)
        totals[key] = totals.get(key, 0) + row_units(row, field)
    return {key: physical_bandwidth(results, units) for key, units in totals.items()}


def _selector_label(selector: str | dict) -> str:
    path = selector if isinstance(selector, str) else selector.get("path")
    return (
        path.strip("^$") if path is not None else json.dumps(selector, sort_keys=True)
    )


def row_pair(step: dict, row: dict) -> tuple[str, str]:
    """Label combine rows by selectors and, for per-group expansion, their complete physical group keys."""
    source, target = row_endpoints(row)
    if step["step_type"] == "TrafficMatrixPlacement":
        demand = next(
            d for d in step["question"]["demands"] if d["id"] == row["base_id"]
        )
        if demand["mode"] == "combine":
            source_label = _selector_label(demand["source"])
            target_label = _selector_label(demand["target"])
            if demand["group_mode"] != "flatten":
                source_label += f" {json.dumps(source)}"
                target_label += f" {json.dumps(target)}"
            return source_label, target_label
    if len(source) == len(target) == 1:
        return source[0], target[0]
    return _selector_label(step["question"]["source"]), _selector_label(
        step["question"]["target"]
    )


def pair_totals(
    results: dict, step: dict, case: dict, field: str
) -> dict[tuple[str, str], float]:
    """Sum each directed labelled demand as one unit, preserving grouped selectors without pair expansion."""
    totals: dict[tuple[str, str], int] = {}
    for row in rows(case):
        source, target = row_endpoints(row)
        if (
            not source
            or not target
            or (len(source) == len(target) == 1 and source == target)
        ):
            continue
        pair = row_pair(step, row)
        totals[pair] = totals.get(pair, 0) + row_units(row, field)
    return {pair: physical_bandwidth(results, units) for pair, units in totals.items()}


def delivery_ratio(case: dict) -> float:
    """Read the required exact native delivery fraction, which is one for a zero request."""
    totals = converged_state(case)["totals"]
    if totals is None:
        raise ValueError("placement totals required for delivery ratio")
    ratio = totals["delivery_ratio"]
    return float(Fraction(ratio["numerator"], ratio["denominator"]))


def feasible_alpha(step: dict) -> float:
    """Read and validate the exact feasible MSD fraction, refusing an absent feasible probe."""
    result = step["result"]
    alpha = result["alpha_feasible"]
    status = result["search_status"]
    if alpha is None:
        raise ValueError(NO_FEASIBLE_ALPHA)
    exact = Fraction(alpha["numerator"], alpha["denominator"])
    value = float(exact)
    if (
        not math.isfinite(value)
        or value < 0
        or status == "none_feasible"
        or isinstance(alpha["value"], bool)
        or not math.isfinite(alpha["value"])
        or alpha["value"] != value
    ):
        raise ValueError(f"{step['name']}: invalid feasible alpha ({status})")
    return value


def placement_steps(results: dict) -> Iterator[dict]:
    """Iterate placement steps in their original execution order."""
    return (
        step for step in steps(results) if step["step_type"] == "TrafficMatrixPlacement"
    )
