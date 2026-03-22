"""Objective function for autoresearch.

Provides:
- ObjectiveResult: dataclass holding evaluation outcome (score, status, metrics).
- ObjectiveFunction: parses objective.yml, extracts metrics from results dicts,
  checks constraints, and computes a scalar score for ranking hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ObjectiveResult:
    score: float
    status: str  # "feasible" or "infeasible"
    primary_value: float
    violated_constraints: list[str] = field(default_factory=list)
    all_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class _ConstraintDef:
    metric: str
    operator: str  # ">=", "<=", "=="
    value: float
    name: str


_OPERATORS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


class ObjectiveFunction:
    """Parse objective.yml and evaluate results dicts against it."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._direction: str = ""
        self._primary_metric: str = ""
        self._metrics: dict[str, str] = {}  # metric_key -> dot-path
        self._constraints: list[_ConstraintDef] = []
        self._parse(self._path)

    def _parse(self, path: Path) -> None:
        with open(path) as f:
            data = yaml.safe_load(f)

        self._direction = data["direction"]
        if self._direction not in ("maximize", "minimize"):
            raise ValueError(
                f"direction must be 'maximize' or 'minimize', got {self._direction!r}"
            )

        self._primary_metric = data["primary_metric"]

        metrics_data = data.get("metrics", {})
        for key, spec in metrics_data.items():
            self._metrics[key] = spec["path"]

        for cdef in data.get("constraints", []):
            op = cdef["operator"]
            if op not in _OPERATORS:
                raise ValueError(
                    f"Unsupported constraint operator: {op!r}. "
                    f"Supported: {list(_OPERATORS.keys())}"
                )
            self._constraints.append(
                _ConstraintDef(
                    metric=cdef["metric"],
                    operator=op,
                    value=float(cdef["value"]),
                    name=cdef.get("name", f"{cdef['metric']}_{op}_{cdef['value']}"),
                )
            )

        # Validate that primary_metric is defined in metrics
        if self._primary_metric not in self._metrics:
            raise ValueError(
                f"primary_metric {self._primary_metric!r} not found in metrics definitions"
            )

    @property
    def direction(self) -> str:
        """'maximize' or 'minimize'."""
        return self._direction

    @property
    def primary_metric(self) -> str:
        return self._primary_metric

    def evaluate(self, results: dict) -> ObjectiveResult:
        """Extract metrics from ngraph results, check constraints, compute score."""
        # Extract all defined metrics
        all_metrics: dict[str, float] = {}
        for key in self._metrics:
            all_metrics[key] = self.extract_metric(results, key)

        primary_value = all_metrics[self._primary_metric]

        # Check constraints
        violated: list[str] = []
        for cdef in self._constraints:
            metric_val = all_metrics.get(cdef.metric)
            if metric_val is None:
                # Metric not extracted — treat as violation
                violated.append(cdef.name)
                continue
            check_fn = _OPERATORS[cdef.operator]
            if not check_fn(metric_val, cdef.value):
                violated.append(cdef.name)

        # Compute score
        if self._direction == "maximize":
            score = primary_value
        else:
            score = -primary_value

        status = "feasible" if not violated else "infeasible"

        # Apply penalty for constraint violations
        if violated:
            score = score - 1e6 * len(violated)

        return ObjectiveResult(
            score=score,
            status=status,
            primary_value=primary_value,
            violated_constraints=violated,
            all_metrics=all_metrics,
        )

    def extract_metric(self, results: dict, key: str) -> float:
        """Extract a single metric value by navigating the dot-path.

        Raises KeyError if the metric key is not defined or the path
        does not exist in the results dict.
        """
        if key not in self._metrics:
            raise KeyError(f"Metric {key!r} not defined in objective")

        dot_path = self._metrics[key]
        return _navigate_dot_path(results, dot_path, key)


def _navigate_dot_path(data: dict, dot_path: str, metric_key: str) -> float:
    """Walk a dot-separated path through nested dicts.

    Raises KeyError with a descriptive message if any segment is missing.
    """
    parts = dot_path.split(".")
    current: Any = data
    for i, part in enumerate(parts):
        if not isinstance(current, dict):
            traversed = ".".join(parts[:i])
            raise KeyError(
                f"Metric {metric_key!r}: cannot navigate into non-dict at "
                f"'{traversed}' (got {type(current).__name__})"
            )
        if part not in current:
            traversed = ".".join(parts[: i + 1])
            raise KeyError(
                f"Metric {metric_key!r}: path segment '{part}' not found "
                f"(full path: '{dot_path}', failed at: '{traversed}')"
            )
        current = current[part]

    return float(current)
