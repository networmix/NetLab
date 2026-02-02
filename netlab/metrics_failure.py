"""
Failure Analysis Metrics Module

Provides utilities for extracting and aggregating failure metrics from
ngraph TrafficMatrixPlacement results.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    """
    Extract failure ratios from a results JSON.

    Args:
        results: Loaded results JSON dictionary
        step_prefix: Prefix for failure analysis steps (default: "tm_")

    Returns:
        Dict mapping step name to list of overall_ratio values
    """
    failure_ratios: Dict[str, List[float]] = {}
    steps = results.get("steps", {})

    for step_name, step_data in steps.items():
        if not step_name.startswith(step_prefix):
            continue

        flow_results = step_data.get("data", {}).get("flow_results", [])
        if not flow_results:
            continue

        # Collect ratios from all iterations
        ratios = [fr["summary"]["overall_ratio"] for fr in flow_results]
        failure_ratios[step_name] = ratios

    return failure_ratios


def compute_failure_stats(
    failure_ratios: Dict[str, List[float]],
) -> Dict[str, FailureStats]:
    """
    Compute statistics for each failure type.

    Args:
        failure_ratios: Dict mapping step name to list of ratios

    Returns:
        Dict mapping step name to FailureStats
    """
    stats: Dict[str, FailureStats] = {}

    for step_name, ratios in failure_ratios.items():
        if not ratios:
            continue

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
    """
    Find the worst failure types (all within tolerance of minimum).

    Args:
        failure_stats: Dict of failure statistics
        tolerance: Tolerance for considering failures equal (default: 0.1%)

    Returns:
        Dict with "types" and "min_ratio", or None if no failures
    """
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
    """
    Perform complete failure analysis on a results JSON.

    Args:
        results: Loaded results JSON dictionary
        step_prefix: Prefix for failure analysis steps
        worst_tolerance: Tolerance for worst failure detection

    Returns:
        FailureAnalysisSummary with all statistics
    """
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
    """
    Aggregate failure metrics across multiple seeds.

    Args:
        metrics_by_seed: List of failure stats dicts (one per seed)

    Returns:
        Dict mapping step name to aggregated statistics
    """
    # Collect all ratios per step
    all_ratios: Dict[str, List[float]] = {}
    for seed_metrics in metrics_by_seed:
        for step_name, stats in seed_metrics.items():
            all_ratios.setdefault(step_name, []).extend(stats.ratios)

    # Compute aggregated stats
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
            seeds=len(metrics_by_seed),
        )

    return aggregated


def extract_alpha_star(results: dict, msd_step: str = "msd") -> Optional[float]:
    """
    Extract alpha_star from MSD step results.

    Args:
        results: Loaded results JSON dictionary
        msd_step: Name of the MSD step (default: "msd")

    Returns:
        Alpha star value, or None if not found
    """
    steps = results.get("steps", {})
    msd_data = steps.get(msd_step, {}).get("data", {})
    return msd_data.get("alpha_star")


def extract_network_stats(results: dict) -> Optional[Dict[str, Any]]:
    """
    Extract network statistics from results.

    Args:
        results: Loaded results JSON dictionary

    Returns:
        Dict with node_count, link_count, total_capacity, or None
    """
    steps = results.get("steps", {})
    net_stats = steps.get("network_statistics", {}).get("data", {})

    if not net_stats:
        return None

    return {
        "node_count": net_stats.get("node_count"),
        "link_count": net_stats.get("link_count"),
        "total_capacity": net_stats.get("total_capacity"),
    }
