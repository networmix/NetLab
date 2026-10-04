"""Read failure counts and elapsed time from placement results."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from netlab.metrics.common import flow_occurrence_count

from .common import nonnegative_number
from .validation import validate_sample_counts


@dataclass
class IterOpsResult:
    """Counts and timing for one seed.

    ``failures_count`` sums pattern occurrence counts; ``total_iterations_count``
    also includes one baseline. ``per_iter_duration_sec`` divides the recorded
    step duration by that total, including repeated patterns.
    """

    failures_count: int
    unique_patterns: int
    total_iterations_count: int
    total_duration_sec: float = float("nan")
    per_iter_duration_sec: float = float("nan")

    def flat_series(self) -> pd.Series:
        return pd.Series(
            {
                "iters_fail": float(self.failures_count),
                "iters_total": float(self.total_iterations_count),
                "unique_patterns": float(self.unique_patterns),
                "tm_duration_total_sec": float(self.total_duration_sec),
                "tm_duration_per_iter_sec": float(self.per_iter_duration_sec),
            }
        )

    def to_jsonable(self) -> dict:
        return {
            "iters_fail": int(self.failures_count),
            "iters_total": int(self.total_iterations_count),
            "unique_patterns": int(self.unique_patterns),
            "tm_duration_total_sec": float(self.total_duration_sec),
            "tm_duration_per_iter_sec": float(self.per_iter_duration_sec),
        }


def compute_iter_ops(results: dict) -> IterOpsResult:
    """Sum failure occurrence counts and read the recorded placement duration."""
    tm_step = results.get("steps", {}).get("tm_placement", {}) or {}
    meta = tm_step.get("metadata", {}) or {}
    data = tm_step.get("data", {}) or {}
    baseline_it = data.get("baseline")
    if not isinstance(baseline_it, dict):
        raise ValueError("tm_placement.data.baseline dict required")
    fr = data.get("flow_results", []) or []
    if not isinstance(fr, list):
        raise ValueError("tm_placement.data.flow_results must be a list")

    validate_sample_counts(tm_step, "tm_placement")

    fail_count = sum(flow_occurrence_count(it) for it in fr)
    unique_patterns = len(fr)
    total_count = 1 + fail_count  # baseline + failures

    total_duration = float("nan")
    dur = meta.get("duration_sec")
    if dur is not None:
        total_duration = nonnegative_number(dur, "duration_sec")

    per_iter_duration = (
        float(total_duration / total_count)
        if total_count > 0 and pd.notna(total_duration)
        else float("nan")
    )

    return IterOpsResult(
        failures_count=int(fail_count),
        unique_patterns=int(unique_patterns),
        total_iterations_count=int(total_count),
        total_duration_sec=float(total_duration),
        per_iter_duration_sec=float(per_iter_duration),
    )
