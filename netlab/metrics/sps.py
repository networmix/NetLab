from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .common import expand_flow_results, pair_totals, require_capacity_pairs
from .distributions import threshold_at_probability


@dataclass
class SpsResult:
    # Structural Pair Survivability per iteration (0..1)
    series: pd.Series  # index=failure_id or iteration id
    # Tails (quantiles of SPS)
    tails: Dict[str, float]
    # SPS at probability p (threshold met/exceeded with probability p)
    sps_at_probability: Dict[float, float]

    def to_jsonable(self) -> dict:
        return {
            "series": list(map(float, self.series.values)),
            "tails": {str(k): float(v) for k, v in self.tails.items()},
            "sps_at_probability": {
                str(k): float(v) for k, v in self.sps_at_probability.items()
            },
        }


def _extract_baseline_demands_tm(results: dict) -> Dict[str, float]:
    """Per-pair baseline demand from tm_placement baseline. Key is 's→d'."""
    tm_step = results.get("steps", {}).get("tm_placement", {}) or {}
    data = tm_step.get("data", {}) or {}
    base = data.get("baseline")
    if not isinstance(base, dict):
        return {}
    return {
        f"{s}→{d}": demand
        for (s, d), demand in pair_totals(base, "demand").items()
        if demand > 0
    }


def _per_iteration_pair_caps(results: dict) -> pd.DataFrame:
    """Build a pair-capacity table with one row per weighted failure iteration.

    Columns are source/destination pairs; missing capacities are zero.
    """
    mf_step = results.get("steps", {}).get("node_to_node_capacity_matrix", {}) or {}
    data = mf_step.get("data", {}) or {}
    fr = expand_flow_results(data.get("flow_results", []) or [])
    pairs: Dict[str, Dict[str, float]] = {}
    for idx, it in enumerate(fr):
        pairs[f"it{idx}"] = {
            f"{s}→{d}": capacity
            for (s, d), capacity in pair_totals(it, "placed").items()
        }
    if not pairs:
        return pd.DataFrame()
    df = pd.DataFrame(list(pairs.values()), index=list(pairs)).fillna(0.0)
    df = df.reindex(columns=list(_extract_baseline_demands_tm(results)), fill_value=0.0)
    df.index.name = "failure_id"
    return df


def compute_sps(results: dict) -> SpsResult:
    dem_base = _extract_baseline_demands_tm(results)
    if not dem_base:
        return SpsResult(
            series=pd.Series(dtype=float),
            tails={},
            sps_at_probability={},
        )
    require_capacity_pairs(results)
    caps = _per_iteration_pair_caps(results)
    if caps is None or caps.empty:
        return SpsResult(
            series=pd.Series(dtype=float),
            tails={},
            sps_at_probability={},
        )

    pairs = list(dem_base.keys())
    caps = caps.reindex(columns=pairs, fill_value=0.0)
    dem_vec = np.array([dem_base[p] for p in pairs], dtype=float)
    total_dem = float(np.sum(dem_vec))
    if not np.isfinite(total_dem) or total_dem <= 0.0:
        return SpsResult(
            series=pd.Series(dtype=float),
            tails={},
            sps_at_probability={},
        )

    sps_vals: List[float] = []
    for _, row in caps.iterrows():
        cap_vec = np.asarray(row.values, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dem_vec > 0.0, cap_vec / dem_vec, 0.0)
        headroom = np.clip(ratio, 0.0, 1.0)
        sps = float(np.sum(headroom * dem_vec) / total_dem)
        sps_vals.append(sps)

    series = pd.Series(sps_vals, index=caps.index, dtype=float)

    tails = {
        "p50": float(series.quantile(0.50, interpolation="lower")),
        "p90": float(series.quantile(0.90, interpolation="lower")),
        "p95": float(series.quantile(0.95, interpolation="lower")),
        "p99": float(series.quantile(0.99, interpolation="lower")),
        "p999": float(series.quantile(0.999, interpolation="lower")),
        "p9999": float(series.quantile(0.9999, interpolation="lower")),
    }
    sps_at_p = {}
    for p in (90.0, 95.0, 99.0, 99.9, 99.99):
        sps_at_p[p] = threshold_at_probability(series.to_numpy(), p)

    return SpsResult(series=series, tails=tails, sps_at_probability=sps_at_p)
