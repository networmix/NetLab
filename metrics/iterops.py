"""Iteration-level operation metrics for tm_placement.

This module extracts per-iteration aggregate operation counters recorded in
tm_placement results under ``flow_results[*].data.iteration_metrics``.

We summarize these into seed-level totals (baseline, failures, all) and
per-iteration averages for cross-seed and cross-scenario comparisons.

Primary counters (when available):
- spf_calls_total
- flows_created_total
- reopt_calls_total
- place_iterations_total

Notes:
- Missing counters are treated as 0.0 for totals and NaN for per-iteration averages.
- We use only iteration-level aggregates (not per-flow ``policy_metrics``) to avoid
  double counting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

ITER_KEYS: Tuple[str, ...] = (
    "spf_calls_total",
    "flows_created_total",
    "reopt_calls_total",
    "place_iterations_total",
)


@dataclass
class IterOpsResult:
    """Iteration operation metrics for a single seed.

    Attributes:
        baseline: Totals for the baseline iteration only.
        failures: Totals summed across failure iterations.
        totals_all: Totals across all iterations (baseline + failures).
        per_iteration: Per-iteration values across failures (lists aligned with
            the original failure iteration order). Baseline is excluded.
        failures_count: Number of failure iterations observed.
        total_iterations_count: Baseline + failures.
    """

    baseline: Dict[str, float]
    failures: Dict[str, float]
    totals_all: Dict[str, float]
    per_iteration: Dict[str, List[float]]
    failures_count: int
    total_iterations_count: int
    # Total wall-clock duration of tm_placement step (seconds), and per-iteration average
    total_duration_sec: float = float("nan")
    per_iter_duration_sec: float = float("nan")

    def flat_series(self) -> pd.Series:
        """Return a flat series with totals and per-iteration averages.

        Keys include:
            - <key>_total_fail, <key>_total_base, <key>_total_all
            - <key>_per_iter (failures only average)
            - iters_fail, iters_total
        """

        data: Dict[str, float] = {
            "iters_fail": float(self.failures_count),
            "iters_total": float(self.total_iterations_count),
        }
        for k in ITER_KEYS:
            base = float(self.baseline.get(k, 0.0))
            fail = float(self.failures.get(k, 0.0))
            allv = float(self.totals_all.get(k, 0.0))
            data[f"{k}_total_base"] = base
            data[f"{k}_total_fail"] = fail
            data[f"{k}_total_all"] = allv
            # Per-iteration average over failures only
            if self.failures_count > 0:
                data[f"{k}_per_iter"] = float(fail / self.failures_count)
            else:
                data[f"{k}_per_iter"] = float("nan")
        # Timing metrics for the entire tm_placement step
        data["tm_duration_total_sec"] = float(self.total_duration_sec)
        data["tm_duration_per_iter_sec"] = float(self.per_iter_duration_sec)
        return pd.Series(data)

    def to_jsonable(self) -> dict:
        return {
            "baseline": {k: float(self.baseline.get(k, 0.0)) for k in ITER_KEYS},
            "failures": {k: float(self.failures.get(k, 0.0)) for k in ITER_KEYS},
            "totals_all": {k: float(self.totals_all.get(k, 0.0)) for k in ITER_KEYS},
            "per_iteration": {
                k: [float(x) for x in self.per_iteration.get(k, [])] for k in ITER_KEYS
            },
            "iters_fail": int(self.failures_count),
            "iters_total": int(self.total_iterations_count),
            "tm_duration_total_sec": float(self.total_duration_sec),
            "tm_duration_per_iter_sec": float(self.per_iter_duration_sec),
        }


def _get_iter_metrics(it: dict) -> Dict[str, float]:
    data = (it.get("data") or {}).get("iteration_metrics") or {}
    out: Dict[str, float] = {}
    for k in ITER_KEYS:
        v = data.get(k)
        try:
            out[k] = float(v) if v is not None else 0.0
        except Exception:
            out[k] = 0.0
    return out


def compute_iter_ops(results: dict) -> IterOpsResult:
    """Extract and aggregate iteration operation metrics from tm_placement.

    Args:
        results: A single run results dictionary containing tm_placement.

    Returns:
        IterOpsResult with totals for baseline, failures, and all iterations, as
        well as per-failure-iteration sequences for each metric.

    Raises:
        ValueError: If tm_placement step or its iterations are missing/invalid.
    """

    tm_step = results.get("steps", {}).get("tm_placement", {}) or {}
    meta = tm_step.get("metadata", {}) or {}
    if bool(meta.get("baseline")) is not True:
        raise ValueError(
            "tm_placement.metadata.baseline must be true and baseline must be included"
        )
    data = tm_step.get("data", {}) or {}
    fr = data.get("flow_results", []) or []
    if not isinstance(fr, list) or not fr:
        raise ValueError("tm_placement.data.flow_results missing or empty")
    if str(fr[0].get("failure_id", "")) != "baseline":
        raise ValueError(
            "tm_placement baseline must be first (flow_results[0].failure_id == 'baseline')"
        )

    base_it = fr[0]
    base = _get_iter_metrics(base_it)

    # Aggregate failures
    fail_totals: Dict[str, float] = {k: 0.0 for k in ITER_KEYS}
    per_iter: Dict[str, List[float]] = {k: [] for k in ITER_KEYS}
    for it in fr[1:]:
        m = _get_iter_metrics(it)
        for k in ITER_KEYS:
            val = float(m.get(k, 0.0))
            fail_totals[k] += val
            per_iter[k].append(val)

    totals_all = {
        k: float(base.get(k, 0.0)) + fail_totals.get(k, 0.0) for k in ITER_KEYS
    }
    fail_count = max(0, len(fr) - 1)
    total_count = len(fr)
    # Timing: prefer tm_placement.metadata.duration_sec; fallback to execution_time
    total_duration = float("nan")
    try:
        dur = meta.get("duration_sec")
        if dur is None:
            dur = meta.get("execution_time")
        if dur is not None:
            total_duration = float(dur)
    except Exception:
        total_duration = float("nan")

    per_iter_duration = (
        float(total_duration / total_count)
        if total_count > 0 and pd.notna(total_duration)
        else float("nan")
    )

    return IterOpsResult(
        baseline=base,
        failures=fail_totals,
        totals_all=totals_all,
        per_iteration=per_iter,
        failures_count=int(fail_count),
        total_iterations_count=int(total_count),
        total_duration_sec=float(total_duration),
        per_iter_duration_sec=float(per_iter_duration),
    )
