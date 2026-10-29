"""Shared utilities for metrics modules."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from netlab.artifacts import write_json_atomic
from netlab.results import (
    baseline_case,
    endpoint_totals,
    expand_cases,
    pair_totals,
    require_step,
    sampled_cases,
)


def nonnegative_number(value: object, field: str) -> float:
    """Require a finite nonnegative numeric value and name the invalid field."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite nonnegative number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return number


def baseline_demand_map(
    results: dict, step_name: str = "tm_placement"
) -> dict[tuple[str, str], float]:
    """Read positive physical baseline requests by their complete directed demand labels."""
    step = require_step(results, step_name)
    return {
        pair: demand
        for pair, demand in pair_totals(
            results, step, baseline_case(step), "reference_request_units"
        ).items()
        if demand > 0
    }


def get_tm_baseline_and_failures(
    results: dict, step_name: str = "tm_placement"
) -> tuple[dict, list[dict]]:
    """Read the reference baseline and expand all sampled draws, including empty ones."""
    step = require_step(results, step_name)
    return baseline_case(step), expand_cases(sampled_cases(step))


def require_capacity_pairs(results: dict) -> None:
    """Require matching labelled physical endpoint sets before using MaxFlow capacities for demands."""
    required = set(baseline_demand_map(results))
    step = require_step(results, "node_to_node_capacity_matrix")
    available = set(pair_totals(results, step, baseline_case(step), "delivered_units"))
    placement = require_step(results, "tm_placement")
    required_endpoints = set(
        endpoint_totals(results, baseline_case(placement), "reference_request_units")
    )
    available_endpoints = set(
        endpoint_totals(results, baseline_case(step), "delivered_units")
    )
    missing = required - available
    missing_endpoints = required_endpoints - available_endpoints
    if missing or missing_endpoints:
        raise ValueError(
            f"MaxFlow baseline does not cover exact placement endpoints: {sorted(missing or missing_endpoints)}. grouped capacities cannot be assigned to individual demands."
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
