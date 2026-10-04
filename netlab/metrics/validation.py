"""Validate arithmetic and sample accounting in current NetGraph results."""

from __future__ import annotations

import math

from .common import flow_occurrence_count, nonnegative_number
from .msd import compute_alpha_star


def _require_steps(res: dict, require_maxflow: bool) -> None:
    required = ["msd_baseline", "tm_placement"]
    if require_maxflow:
        required.append("node_to_node_capacity_matrix")
    for step in required:
        if step not in res.get("steps", {}):
            raise ValueError(f"Missing required step in results: {step}")


def _validate_alpha_and_base_demands(res: dict) -> float:
    demands = (
        res.get("steps", {}).get("msd_baseline", {}).get("data", {}).get("base_demands")
    )
    if not isinstance(demands, list) or not demands:
        raise ValueError("MSD base_demands must be a nonempty list")
    for demand in demands:
        for field in ("source", "target"):
            value = demand.get(field)
            if not isinstance(value, (str, dict)) or (
                isinstance(value, str) and not value.strip()
            ):
                raise ValueError(
                    f"MSD demand requires a nonempty source/target selector: {field}"
                )
        volume = nonnegative_number(demand.get("volume"), "MSD demand volume")
        if volume == 0:
            raise ValueError("MSD base_demands contains a zero-demand entry")
    alpha = compute_alpha_star(res)
    if alpha.base_total_demand <= 0:
        raise ValueError("MSD base_total_demand must be positive")
    return nonnegative_number(
        alpha.base_total_demand * alpha.alpha_star, "scaled demand"
    )


def _close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"{context}: {actual} does not match {expected}")


def _validate_iteration(it: dict, name: str, expected_demand: float | None) -> None:
    flows = it.get("flows")
    if not isinstance(flows, list):
        raise ValueError(f"{name}: flows must be a list")
    total_demand = total_placed = 0.0
    for flow in flows:
        source, target = flow.get("source"), flow.get("destination")
        if (
            not isinstance(source, str)
            or not isinstance(target, str)
            or not source
            or not target
        ):
            raise ValueError(f"{name}: invalid flow endpoints")
        demand, placed, dropped = [
            nonnegative_number(flow.get(k), f"{name}.{k}")
            for k in ("demand", "placed", "dropped")
        ]
        if source == target and (demand or placed or dropped):
            raise ValueError(f"{name}: self-pair must have zero flow")
        _close(placed + dropped, demand, f"{name}: placed + dropped vs demand")
        costs = flow.get("cost_distribution", {})
        if not isinstance(costs, dict):
            raise ValueError(f"{name}: cost_distribution must be a mapping")
        if costs:
            volume = 0.0
            for cost, weight in costs.items():
                nonnegative_number(cost, f"{name}.cost")
                volume += nonnegative_number(weight, f"{name}.cost volume")
            _close(volume, placed, f"{name}: cost_distribution volume vs placed")
        total_demand += demand
        total_placed += placed
    if expected_demand is not None:
        _close(
            total_demand,
            expected_demand,
            f"{name}: total flow demand vs base_demands × alpha_star",
        )
    summary = it.get("summary", {})
    for key, actual in [("total_demand", total_demand), ("total_placed", total_placed)]:
        if key in summary:
            _close(
                nonnegative_number(summary[key], key),
                actual,
                f"{name}: summary.{key} vs flow sum",
            )
    if "overall_ratio" in summary and total_demand > 0:
        _close(
            nonnegative_number(summary["overall_ratio"], "overall_ratio"),
            total_placed / total_demand,
            f"{name}: overall_ratio",
        )
    if "num_flows" in summary and summary["num_flows"] != len(flows):
        raise ValueError(f"{name}: num_flows disagrees with flow records")


def validate_sample_counts(step: dict, name: str) -> None:
    data = step.get("data", {})
    failures = data.get("flow_results")
    if not isinstance(failures, list):
        raise ValueError(f"{name}: flow_results must be a list")
    counts = [flow_occurrence_count(it) for it in failures]
    metadata = step.get("metadata", {})
    for key, expected in [
        ("iterations", sum(counts)),
        ("unique_patterns", len(counts)),
    ]:
        if key in metadata and (
            type(metadata[key]) is not int or metadata[key] != expected
        ):
            raise ValueError(f"{name}: metadata.{key} disagrees with occurrence counts")
    if "occurrence_counts" in metadata and metadata["occurrence_counts"] != counts:
        raise ValueError(
            f"{name}: metadata.occurrence_counts disagrees with flow results"
        )


def _validate_step(res: dict, name: str, expected_demand: float | None) -> None:
    step = res["steps"][name]
    data = step.get("data", {})
    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError(f"{name}.data.baseline dict required")
    validate_sample_counts(step, name)
    _validate_iteration(baseline, f"{name}.baseline", expected_demand)
    for index, iteration in enumerate(data["flow_results"]):
        _validate_iteration(iteration, f"{name}.failure[{index}]", expected_demand)


def _validate_tm_placement_baseline(
    res: dict, expected_total_at_alpha: float | None
) -> None:
    _validate_step(res, "tm_placement", expected_total_at_alpha)


def _validate_maxflow_baseline(res: dict) -> None:
    _validate_step(res, "node_to_node_capacity_matrix", None)
