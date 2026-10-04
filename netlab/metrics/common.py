"""Shared utilities for metrics modules."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from netlab.artifacts import write_json_atomic


def nonnegative_number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite nonnegative number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return number


def pair_totals(iteration: dict, field: str) -> dict[tuple[str, str], float]:
    """Sum priority classes for each exact directed endpoint pair."""
    totals: dict[tuple[str, str], float] = {}
    flows = iteration.get("flows")
    if not isinstance(flows, list):
        raise ValueError("Flow iteration requires a flows list")
    for record in flows:
        source, target = record["source"], record["destination"]
        if not source or not target or source == target:
            continue
        pair = (source, target)
        totals[pair] = totals.get(pair, 0.0) + nonnegative_number(record[field], field)
    return totals


def flow_occurrence_count(iteration: dict) -> int:
    """Read the required positive integer weight of a flow result."""
    count = iteration.get("occurrence_count")
    if type(count) is not int or count < 1:
        raise ValueError("flow result occurrence_count must be a positive integer")
    return count


def expand_flow_results(flow_results: list[dict]) -> list[dict]:
    """Repeat each failure pattern by its ``occurrence_count``.

    NetGraph stores identical patterns once. Expansion gives each simulated
    iteration equal weight in downstream statistics.
    """
    expanded: list[dict] = []
    for it in flow_results:
        count = flow_occurrence_count(it)
        expanded.extend([it] * count)
    return expanded


def baseline_demand_map(
    results: dict, step_name: str = "tm_placement"
) -> Dict[Tuple[str, str], float]:
    """Extract per-pair baseline demand from a placement step.

    Returns mapping ``(source, destination) -> demand``.
    Pairs with zero or negative demand are excluded.
    """
    step = results.get("steps", {}).get(step_name, {}) or {}
    data = step.get("data", {}) or {}
    base = data.get("baseline")
    if not isinstance(base, dict):
        return {}
    return {
        pair: demand
        for pair, demand in pair_totals(base, "demand").items()
        if demand > 0
    }


def get_tm_baseline_and_failures(
    results: dict, step_name: str = "tm_placement"
) -> Tuple[dict, List[dict]]:
    """Extract baseline dict and expanded failure list from a placement step.

    The returned failure list is expanded by ``occurrence_count`` so each
    Monte Carlo iteration is represented as a separate entry.
    """
    tm_step = results.get("steps", {}).get(step_name, {}) or {}
    tm_data = tm_step.get("data", {}) or {}
    baseline = tm_data.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError(f"{step_name}.data.baseline dict required")
    flow_results = tm_data.get("flow_results", []) or []
    if not isinstance(flow_results, list):
        raise ValueError(f"{step_name}.data.flow_results must be a list")
    return baseline, expand_flow_results(flow_results)


def require_capacity_pairs(results: dict) -> None:
    """Refuse unrelated MaxFlow group labels instead of reporting false outages."""
    required = set(baseline_demand_map(results))
    data = (
        results.get("steps", {}).get("node_to_node_capacity_matrix", {}).get("data", {})
    )
    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("node_to_node_capacity_matrix baseline required")
    available = set(pair_totals(baseline, "placed"))
    missing = required - available
    if missing:
        raise ValueError(
            f"MaxFlow baseline does not cover exact placement endpoints: {sorted(missing)}. Use selectors with full endpoint capture groups; grouped capacities cannot be assigned to individual demands."
        )


def write_metric_json(path: Path, data: Any) -> None:
    """Represent unavailable numeric metrics as JSON null, never NaN/Infinity."""

    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    write_json_atomic(path, clean(data))
