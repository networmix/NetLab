"""Extract and aggregate failure metrics from TrafficMatrixPlacement results."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from netlab.metrics.common import expand_flow_results, nonnegative_number


@dataclass
class FailureStats:
    """Statistics for a single failure type."""

    step_name: str
    iterations: int
    min_ratio: float
    avg_ratio: float
    max_ratio: float
    std_dev: float
    ratios: List[float] = field(default_factory=list, repr=False)


@dataclass
class AggregatedFailureStats:
    """Aggregated statistics for a failure type across seeds."""

    step_name: str
    total_iterations: int
    min_ratio: float
    avg_ratio: float
    max_ratio: float
    std_dev: float
    seeds: int


@dataclass
class FailureAnalysisSummary:
    """Summary of failure analysis across all failure types."""

    failure_stats: Dict[str, FailureStats]
    worst_failures: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "failure_analysis": {
                name: {
                    "iterations": stats.iterations,
                    "min_ratio": stats.min_ratio,
                    "avg_ratio": stats.avg_ratio,
                    "max_ratio": stats.max_ratio,
                    "std_dev": stats.std_dev,
                }
                for name, stats in self.failure_stats.items()
            }
        }
        if self.worst_failures:
            result["worst_failures"] = self.worst_failures
        return result


def extract_failure_ratios(
    results: dict,
    step_prefix: str = "tm_",
) -> Dict[str, List[float]]:
    """Return occurrence-weighted failure ratios, keyed by placement step name."""
    failure_ratios: Dict[str, List[float]] = {}
    steps = results.get("steps", {})

    for step_name, step_data in steps.items():
        if not step_name.startswith(step_prefix):
            continue

        flow_results = step_data.get("data", {}).get("flow_results", [])
        if not flow_results:
            continue

        ratios = [
            fr["summary"]["overall_ratio"] for fr in expand_flow_results(flow_results)
        ]
        failure_ratios[step_name] = ratios

    return failure_ratios


def compute_failure_stats(
    failure_ratios: Dict[str, List[float]],
) -> Dict[str, FailureStats]:
    """Summarize failure ratios for each placement step."""
    stats: Dict[str, FailureStats] = {}

    for step_name, ratios in failure_ratios.items():
        if not ratios:
            continue

        ratios = [nonnegative_number(value, "overall_ratio") for value in ratios]
        if any(value > 1.0 + 1e-9 for value in ratios):
            raise ValueError("overall_ratio cannot exceed one")
        stats[step_name] = FailureStats(
            step_name=step_name,
            iterations=len(ratios),
            min_ratio=min(ratios),
            avg_ratio=statistics.mean(ratios),
            max_ratio=max(ratios),
            std_dev=statistics.stdev(ratios) if len(ratios) > 1 else 0,
            ratios=ratios,
        )

    return stats


def find_worst_failures(
    failure_stats: Dict[str, FailureStats],
    tolerance: float = 0.001,
) -> Optional[Dict[str, Any]]:
    """Return all step names within ``tolerance`` of the lowest ratio.

    The result has ``types`` and ``min_ratio`` keys, or is None without failures.
    Tolerance is an absolute ratio difference (default 0.001).
    """
    nonnegative_number(tolerance, "worst failure tolerance")
    if not failure_stats:
        return None

    min_ratio = min(stats.min_ratio for stats in failure_stats.values())
    worst_types = [
        name
        for name, stats in failure_stats.items()
        if abs(stats.min_ratio - min_ratio) <= tolerance
    ]

    return {
        "types": sorted(worst_types),
        "min_ratio": min_ratio,
    }


def analyze_results(
    results: dict,
    step_prefix: str = "tm_",
    worst_tolerance: float = 0.001,
) -> FailureAnalysisSummary:
    """Summarize failure ratios and identify the worst failure types."""
    ratios = extract_failure_ratios(results, step_prefix)
    stats = compute_failure_stats(ratios)
    worst = find_worst_failures(stats, worst_tolerance)

    return FailureAnalysisSummary(
        failure_stats=stats,
        worst_failures=worst,
    )


def aggregate_failure_metrics(
    metrics_by_seed: List[Dict[str, FailureStats]],
) -> Dict[str, AggregatedFailureStats]:
    """Pool ratios across seeds for each failure step."""
    all_ratios: Dict[str, List[float]] = {}
    for seed_metrics in metrics_by_seed:
        for step_name, stats in seed_metrics.items():
            all_ratios.setdefault(step_name, []).extend(stats.ratios)

    aggregated: Dict[str, AggregatedFailureStats] = {}
    for step_name, ratios in all_ratios.items():
        if not ratios:
            continue

        aggregated[step_name] = AggregatedFailureStats(
            step_name=step_name,
            total_iterations=len(ratios),
            min_ratio=min(ratios),
            avg_ratio=statistics.mean(ratios),
            max_ratio=max(ratios),
            std_dev=statistics.stdev(ratios) if len(ratios) > 1 else 0,
            seeds=sum(step_name in seed_metrics for seed_metrics in metrics_by_seed),
        )

    return aggregated


def extract_alpha_star(
    results: dict, msd_step: str = "msd_baseline"
) -> Optional[float]:
    """Read alpha_star from the named MSD step; return None when absent."""
    steps = results.get("steps", {})
    msd_data = steps.get(msd_step, {}).get("data", {})
    value = msd_data.get("alpha_star")
    return nonnegative_number(value, "alpha_star") if value is not None else None


def extract_network_stats(results: dict) -> Optional[Dict[str, Any]]:
    """Return node count, link count, and total capacity, or None when absent."""
    steps = results.get("steps", {})
    net_stats = steps.get("network_statistics", {}).get("data", {})

    if not net_stats:
        return None

    return {
        "node_count": net_stats.get("node_count"),
        "link_count": net_stats.get("link_count"),
        "total_capacity": net_stats.get("total_capacity"),
    }
