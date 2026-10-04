"""Paired comparisons on matching seeds, using the same metrics as project tables."""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .paired import holm_adjust, paired_t
from .seed_data import collect_seed_metrics

COMPARISON_METRICS = (
    "alpha_star",
    "bw_p999_pct",
    "lat_fail_p99",
    "USD_per_Gbit_p999",
    "Watt_per_Gbit_p999",
    "capex_total",
)


def compare_scenarios(
    analysis_root: Path,
    *,
    alpha: float = 0.05,
    scenarios: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Compare A-B; adjust across scenario pairs within each metric family."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one")
    directories = sorted(
        p for p in analysis_root.iterdir() if p.is_dir() and not p.name.startswith("_")
    )
    per_scenario = {
        p.name: collect_seed_metrics(p)
        for p in directories
        if scenarios is None or p.name in scenarios
    }
    if scenarios is not None and (
        scenarios[0] == scenarios[1] or any(s not in per_scenario for s in scenarios)
    ):
        raise ValueError(
            "Select two distinct scenarios present in the metrics directory"
        )
    pairs = [scenarios] if scenarios else list(combinations(per_scenario, 2))
    records = []
    for metric in COMPARISON_METRICS:
        comparisons = {}
        for a, b in pairs:
            common = sorted(per_scenario[a].keys() & per_scenario[b].keys())
            values = [
                (
                    per_scenario[a][seed].get(metric, math.nan),
                    per_scenario[b][seed].get(metric, math.nan),
                )
                for seed in common
            ]
            finite = [
                (a_value, b_value)
                for a_value, b_value in values
                if math.isfinite(a_value) and math.isfinite(b_value)
            ]
            if len(finite) < 3:
                continue
            a_values, b_values = np.asarray(finite).T
            result = paired_t(a_values, b_values, alpha=alpha)
            result.update(metric=metric, scen_a=a, scen_b=b)
            comparisons[a, b] = result
        adjusted = holm_adjust(
            [(pair, result["p"]) for pair, result in comparisons.items()]
        )
        for pair, result in comparisons.items():
            result["p_adj"] = adjusted[pair]
            records.append(result)
    return records
