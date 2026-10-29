"""Read placement counts and NetLab-measured step wall time."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from netlab.results import baseline_case, case_occurrences, require_step, sampled_cases
from netlab.run_record import step_duration

from .validation import validate_sample_counts


@dataclass
class IterOpsResult:
    """Counts and timing for one seed.

    ``failures_count`` includes sampled empty draws; ``total_iterations_count``
    adds one baseline reference. Timings come from the NetLab run record.
    """

    failures_count: int
    unique_patterns: int
    total_iterations_count: int
    total_duration_sec: float
    per_iter_duration_sec: float

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


def compute_iter_ops(results: dict, run_record: dict) -> IterOpsResult:
    """Count sampled draws and the additional baseline reference."""
    step = require_step(results, "tm_placement")
    baseline_case(step)
    validate_sample_counts(step, "tm_placement")
    samples = sampled_cases(step)
    fail_count = sum(case_occurrences(case) for case in samples)
    duration = step_duration(run_record, "tm_placement")
    return IterOpsResult(
        failures_count=fail_count,
        unique_patterns=len(samples),
        total_iterations_count=1 + fail_count,
        total_duration_sec=duration,
        per_iter_duration_sec=duration / (1 + fail_count),
    )
