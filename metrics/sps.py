from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class SpsResult:
    # Structural Pair Survivability per iteration (0..1)
    series: pd.Series  # index=failure_id or iteration id
    # Tails (quantiles of SPS)
    tails: Dict[str, float]
    # SPS at probability p (threshold met/exceeded with probability p)
    sps_at_probability: Dict[float, float]
    # Optional gaps vs placement BW@p (computed by caller)
    gap_vs_bw_at_p: Dict[float, float]

    def to_jsonable(self) -> dict:
        return {
            "series": list(map(float, self.series.values)),
            "tails": {str(k): float(v) for k, v in self.tails.items()},
            "sps_at_probability": {
                str(k): float(v) for k, v in self.sps_at_probability.items()
            },
            "gap_vs_bw_at_p": {
                str(k): float(v) for k, v in self.gap_vs_bw_at_p.items()
            },
        }


def _extract_baseline_demands_tm(results: dict) -> Dict[str, float]:
    """Per-pair baseline demand from tm_placement baseline (first iteration). Key is 's→d'."""
    tm_step = results.get("steps", {}).get("tm_placement", {}) or {}
    data = tm_step.get("data", {}) or {}
    fr = data.get("flow_results", []) or []
    if not isinstance(fr, list) or not fr:
        return {}
    base = fr[0]
    if str(base.get("failure_id", "")) != "baseline":
        return {}
    out: Dict[str, float] = {}
    for rec in base.get("flows", []) or []:
        s = rec.get("source", "")
        d = rec.get("destination", "")
        if not s or not d or s == d:
            continue
        try:
            dem = float(rec.get("demand", 0.0))
        except Exception:
            dem = 0.0
        if dem <= 0.0:
            continue
        out[f"{s}→{d}"] = dem
    return out


def _per_iteration_pair_caps(results: dict) -> pd.DataFrame:
    """
    Build a DataFrame of per-pair maxflow capacities for each iteration.
    Rows indexed by iteration (failure_id order), columns by 's→d', values are capacities (Gb/s).
    Missing entries filled with 0.
    """
    mf_step = results.get("steps", {}).get("node_to_node_capacity_matrix", {}) or {}
    data = mf_step.get("data", {}) or {}
    fr = data.get("flow_results", []) or []
    pairs: Dict[str, Dict[str, float]] = {}
    ids: List[str] = []
    for it in fr:
        fid = str(it.get("failure_id", f"it{len(ids)}"))
        ids.append(fid)
        col: Dict[str, float] = {}
        for rec in it.get("flows", []) or []:
            s = rec.get("source", "")
            d = rec.get("destination", "")
            if not s or not d or s == d:
                continue
            try:
                cap = float(rec.get("placed", 0.0))
            except Exception:
                cap = 0.0
            col[f"{s}→{d}"] = cap
        pairs[fid] = col
    if not pairs:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(pairs, orient="index").fillna(0.0)
    # Preserve order if 'baseline' is present first
    df.index.name = "failure_id"
    return df


def compute_sps(results: dict) -> SpsResult:
    dem_base = _extract_baseline_demands_tm(results)
    if not dem_base:
        # No baseline demands → empty result
        return SpsResult(
            series=pd.Series(dtype=float),
            tails={},
            sps_at_probability={},
            gap_vs_bw_at_p={},
        )
    caps = _per_iteration_pair_caps(results)
    if caps is None or caps.empty:
        return SpsResult(
            series=pd.Series(dtype=float),
            tails={},
            sps_at_probability={},
            gap_vs_bw_at_p={},
        )

    # Align columns (pairs); missing caps treated as 0
    pairs = list(dem_base.keys())
    caps = caps.reindex(columns=pairs, fill_value=0.0)
    dem_vec = np.array([dem_base[p] for p in pairs], dtype=float)
    total_dem = float(np.sum(dem_vec))
    if not np.isfinite(total_dem) or total_dem <= 0.0:
        return SpsResult(
            series=pd.Series(dtype=float),
            tails={},
            sps_at_probability={},
            gap_vs_bw_at_p={},
        )

    sps_vals: List[float] = []
    for _, row in caps.iterrows():
        cap_vec = np.asarray(row.values, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dem_vec > 0.0, cap_vec / dem_vec, 0.0)
        headroom = np.clip(ratio, 0.0, 1.0)
        sps = float(np.sum(headroom * dem_vec) / total_dem)
        sps_vals.append(sps)

    # Per-iteration series in the same order
    series = pd.Series(sps_vals, index=caps.index, dtype=float)

    # Tails and SPS@p
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
        q = max(0.0, 1.0 - (p / 100.0))
        sps_at_p[p] = float(series.quantile(q, interpolation="lower"))

    return SpsResult(
        series=series, tails=tails, sps_at_probability=sps_at_p, gap_vs_bw_at_p={}
    )
