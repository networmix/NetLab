"""Validate native integer accounting and sampled placement results."""

from __future__ import annotations

from fractions import Fraction

from netlab.results import (
    baseline_case,
    case_occurrences,
    cases,
    converged_state,
    has_failure_policy,
    physical_bandwidth,
    require_step,
    row_units,
    rows,
    sampled_cases,
)

from .common import nonnegative_number
from .msd import compute_alpha_star


def _require_steps(res: dict, require_maxflow: bool) -> None:
    required = ["msd_baseline", "tm_placement"]
    if require_maxflow:
        required.append("node_to_node_capacity_matrix")
    for name in required:
        require_step(res, name)


def _validate_alpha_and_base_demands(res: dict) -> None:
    step = require_step(res, "msd_baseline")
    demands = step["question"]["demands"]
    if not isinstance(demands, list) or not demands:
        raise ValueError("MSD question.demands must be a nonempty list")
    for demand in demands:
        for field in ("source", "target"):
            value = demand.get(field)
            if not isinstance(value, (str, dict)) or (
                isinstance(value, str) and not value.strip()
            ):
                raise ValueError(
                    f"MSD demand requires a nonempty source/target selector: {field}"
                )
        if (
            nonnegative_number(demand.get("physical_volume"), "MSD physical_volume")
            == 0
        ):
            raise ValueError("MSD question contains a zero-demand entry")
    alpha = compute_alpha_star(res)
    nonnegative_number(alpha.base_total_demand * alpha.alpha_star, "scaled demand")


def _units(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def validate_sample_counts(step: dict, name: str) -> None:
    records = cases(step)
    baseline_case(step)
    count = sum(case_occurrences(case) for case in records)
    expected = step["result"]["metadata"]["sampling"]["iterations"]
    if type(expected) is not int or expected < 0 or count != expected:
        raise ValueError(f"{name}: sampling.iterations disagrees with case occurrences")
    # No-policy baseline occurrences describe requested draws, not failure samples.
    if not has_failure_policy(step) and any(
        case["case"]["kind"] == "failure" for case in records
    ):
        raise ValueError(f"{name}: failure case without a failure policy")


def _validate_case(res: dict, case: dict, capacity: bool) -> None:
    state = converged_state(case)
    request = delivered = shortfall = unadmitted = source_drop = edge_drop = 0
    for row in rows(case):
        for side in ("sources", "destinations"):
            endpoints = row["endpoints"][side]
            if (
                not isinstance(endpoints, list)
                or not endpoints
                or any(not isinstance(node, str) or not node for node in endpoints)
            ):
                raise ValueError(f"invalid row endpoints: {side}")
        amount = row_units(row, "delivered_units")
        delivered += amount
        physical_bandwidth(res, amount)
        if not capacity:
            reference = row_units(row, "reference_request_units")
            loss = row_units(row, "reference_shortfall_units")
            native = row["result"]
            u, s, e = [
                _units(native[key], key)
                for key in ("unadmitted_units", "source_drop_units", "edge_drop_units")
            ]
            if reference != amount + loss or loss != u + s + e:
                raise ValueError(
                    "row reference request / delivery / shortfall accounting disagrees"
                )
            request += reference
            shortfall += loss
            unadmitted += u
            source_drop += s
            edge_drop += e
        costs = row.get("cost_distribution")
        if costs is not None:
            kind = costs["kind"]
            if kind == "delivered_by_path_cost":
                volume_field, cost_field = "delivered_units", "physical_path_cost"
            elif kind == "marginal_curve":
                volume_field, cost_field = "units", "marginal_cost"
            else:
                raise ValueError(f"Unknown cost report kind: {kind}")
            volume = 0
            for bin in costs["bins"]:
                nonnegative_number(bin[cost_field], cost_field)
                volume += _units(bin[volume_field], volume_field)
            if volume != amount:
                raise ValueError(
                    "cost distribution volume disagrees with delivered_units"
                )
    totals = state["totals"]
    if capacity:
        if totals is not None:
            raise ValueError("capacity totals must be null")
        return
    if totals is None:
        raise ValueError("placement totals required")
    for field, expected in (
        ("reference_request_units", request),
        ("delivered_units", delivered),
        ("reference_shortfall_units", shortfall),
        ("unadmitted_units", unadmitted),
        ("source_drop_units", source_drop),
        ("edge_drop_units", edge_drop),
    ):
        if _units(totals[field], field) != expected:
            raise ValueError(f"totals.{field} disagrees with row sum")
    ratio = totals["delivery_ratio"]
    if ratio["denominator"] <= 0 or Fraction(
        ratio["numerator"], ratio["denominator"]
    ) != (Fraction(delivered, request) if request else Fraction(1)):
        raise ValueError("totals.delivery_ratio disagrees with row sum")


def _validate_step(res: dict, name: str, capacity: bool) -> None:
    step = require_step(res, name)
    validate_sample_counts(step, name)
    baseline = baseline_case(step)
    _validate_case(res, baseline, capacity)
    for case in sampled_cases(step):
        if case is not baseline:
            _validate_case(res, case, capacity)
    if not capacity:
        accounting = step["result"]["accounting"]
        for item in accounting:
            declared, resolved, discarded = [
                _units(item[key], key)
                for key in ("declared_units", "resolved_units", "discarded_units")
            ]
            if declared != resolved + discarded:
                raise ValueError("expansion accounting disagrees")
        resolved = sum(item["resolved_units"] for item in accounting)
        request = sum(
            row_units(row, "reference_request_units") for row in rows(baseline)
        )
        if resolved != request:
            raise ValueError(
                "resolved_units disagrees with baseline reference requests"
            )


def _validate_tm_placement_baseline(res: dict) -> None:
    msd = require_step(res, "msd_baseline")
    placement = require_step(res, "tm_placement")
    if (
        placement["question"]["demand_set"] != msd["question"]["demand_set"]
        or placement["question"]["demands"] != msd["question"]["demands"]
    ):
        raise ValueError("placement declared demands must match the MSD question")
    expected = msd["result"]["alpha_feasible"]
    actual = placement["result"]["alpha"]
    if Fraction(actual["numerator"], actual["denominator"]) != Fraction(
        expected["numerator"], expected["denominator"]
    ):
        raise ValueError("placement alpha must equal the exact MSD feasible alpha")
    _validate_step(res, "tm_placement", False)


def _validate_maxflow_baseline(res: dict) -> None:
    _validate_step(res, "node_to_node_capacity_matrix", True)
