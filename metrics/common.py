"""Shared utilities for metrics modules."""

from __future__ import annotations

from typing import Dict, List, Tuple


def expand_flow_results(flow_results: list[dict]) -> list[dict]:
    """Expand deduplicated flow_results by occurrence_count.

    ngraph deduplicates identical failure patterns during Monte Carlo
    simulation. Each entry's ``occurrence_count`` indicates how many
    iterations produced that exact pattern. This function repeats each
    entry accordingly so that downstream statistical operations weight
    each iteration equally.

    Entries without ``occurrence_count`` default to 1 (backward compatible).
    """
    expanded: list[dict] = []
    for it in flow_results:
        count = max(1, int(it.get("occurrence_count", 1)))
        for _ in range(count):
            expanded.append(it)
    return expanded


def canonical_dc(endpoint: str) -> str:
    """Normalize endpoint to canonical DC-level path ``metro/dc``.

    Examples::

        'metro1/dc1'           → 'metro1/dc1'
        'metro1/dc1/dc/dc'     → 'metro1/dc1'
        'metro1/dc1/rack/node' → 'metro1/dc1'
    """
    if not endpoint:
        return endpoint
    parts = endpoint.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return endpoint


def baseline_demand_map(
    results: dict, step_name: str = "tm_placement"
) -> Dict[Tuple[str, str], float]:
    """Extract per-pair baseline demand from a placement step.

    Returns mapping ``(canonical_src, canonical_dst) -> demand``.
    Pairs with zero or negative demand are excluded.
    """
    step = results.get("steps", {}).get(step_name, {}) or {}
    data = step.get("data", {}) or {}
    base = data.get("baseline")
    if not isinstance(base, dict):
        return {}
    out: Dict[Tuple[str, str], float] = {}
    for rec in base.get("flows", []) or []:
        s = canonical_dc(rec.get("source", ""))
        d = canonical_dc(rec.get("destination", ""))
        if not s or not d or s == d:
            continue
        try:
            dem = float(rec.get("demand", 0.0))
        except Exception:
            dem = 0.0
        if dem <= 0.0:
            continue
        out[(s, d)] = dem
    return out


def get_tm_baseline_and_failures(results: dict) -> Tuple[dict, List[dict]]:
    """Extract baseline dict and expanded failure list from tm_placement.

    The returned failure list is expanded by ``occurrence_count`` so each
    Monte Carlo iteration is represented as a separate entry.
    """
    tm_step = results.get("steps", {}).get("tm_placement", {}) or {}
    tm_data = tm_step.get("data", {}) or {}
    baseline = tm_data.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("tm_placement.data.baseline dict required")
    flow_results = tm_data.get("flow_results", []) or []
    if not isinstance(flow_results, list):
        raise ValueError("tm_placement.data.flow_results must be a list")
    return baseline, expand_flow_results(flow_results)
