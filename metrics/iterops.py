"""Iteration-level summary metrics for tm_placement.

Extracts iteration counts, unique pattern counts, and wall-clock timing
from tm_placement step metadata. Uses ``occurrence_count`` to recover
the true number of Monte Carlo iterations from deduplicated results.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class IterOpsResult:
    """Iteration summary for a single seed.

    Attributes:
        failures_count: Total failure iterations (sum of occurrence_count).
        unique_patterns: Number of unique failure patterns after dedup.
        total_iterations_count: 1 (baseline) + failures_count.
        total_duration_sec: Wall-clock duration of the tm_placement step.
        per_iter_duration_sec: total_duration_sec / total_iterations_count.
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
    """Extract iteration counts and timing from tm_placement metadata.

    The true failure iteration count is recovered by summing
    ``occurrence_count`` across all deduplicated flow_results entries.
    """
    tm_step = results.get("steps", {}).get("tm_placement", {}) or {}
    meta = tm_step.get("metadata", {}) or {}
    data = tm_step.get("data", {}) or {}
    baseline_it = data.get("baseline")
    if not isinstance(baseline_it, dict):
        raise ValueError("tm_placement.data.baseline dict required")
    fr = data.get("flow_results", []) or []
    if not isinstance(fr, list):
        raise ValueError("tm_placement.data.flow_results must be a list")

    # Recover true iteration count from occurrence_count
    fail_count = sum(max(1, int(it.get("occurrence_count", 1))) for it in fr)
    unique_patterns = len(fr)
    total_count = 1 + fail_count  # baseline + failures

    # Timing: prefer metadata.duration_sec; fallback to execution_time
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
        failures_count=int(fail_count),
        unique_patterns=int(unique_patterns),
        total_iterations_count=int(total_count),
        total_duration_sec=float(total_duration),
        per_iter_duration_sec=float(per_iter_duration),
    )
