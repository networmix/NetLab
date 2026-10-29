"""NetLab-owned wall-clock measurements stored beside native results."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .results import steps


def run_record_path(results_path: Path) -> Path:
    return results_path.with_suffix(".run.json")


def step_duration(record: dict, name: str) -> float:
    """Require a finite nonnegative wall time; absent timing is never a number."""
    durations = record.get("step_durations_sec")
    value = durations.get(name) if isinstance(durations, dict) else None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"Run record requires a finite nonnegative duration for step {name!r}"
        )
    return float(value)


def validate_step_durations(results: dict, record: dict) -> None:
    for step in steps(results):
        step_duration(record, step["name"])


def load_run_record(results_path: Path, results: dict) -> dict:
    path = run_record_path(results_path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read run record {path}: {exc}") from exc
    if not isinstance(record, dict) or record.get("status") != "success":
        raise ValueError(f"Run record {path} must describe a successful run")
    validate_step_durations(results, record)
    return record
