from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

Pair = Tuple[str, str]


def _canonical_dc(endpoint: str) -> str:
    if not endpoint:
        return endpoint
    parts = endpoint.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return endpoint


def _baseline_demand_map_from_tm(results: dict) -> Dict[Pair, float]:
    tm_step = results.get("steps", {}).get("tm_placement", {}) or {}
    data = tm_step.get("data", {}) or {}
    fr = data.get("flow_results", []) or []
    if not isinstance(fr, list) or not fr:
        return {}
    base = fr[0]
    if str(base.get("failure_id", "")) != "baseline":
        return {}
    out: Dict[Pair, float] = {}
    for rec in base.get("flows", []) or []:
        s = _canonical_dc(rec.get("source", ""))
        d = _canonical_dc(rec.get("destination", ""))
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


def _collect_per_iteration_matrix(results: dict, step_name: str) -> pd.DataFrame:
    step = results.get("steps", {}).get(step_name, {}) or {}
    data = step.get("data", {}) or {}
    fr = data.get("flow_results", []) or []
    by_iter: Dict[str, Dict[str, float]] = {}
    for it in fr:
        fid = str(it.get("failure_id", f"it{len(by_iter)}"))
        row: Dict[str, float] = {}
        for rec in it.get("flows", []) or []:
            s = _canonical_dc(rec.get("source", ""))
            d = _canonical_dc(rec.get("destination", ""))
            if not s or not d or s == d:
                continue
            try:
                val = float(rec.get("placed", 0.0))
            except Exception:
                val = 0.0
            row[f"{s}→{d}"] = val
        by_iter[fid] = row
    if not by_iter:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(by_iter, orient="index").fillna(0.0)
    df.index.name = "failure_id"
    return df


def _percentiles_per_pair(matrix: pd.DataFrame, probs: List[float]) -> pd.DataFrame:
    if matrix is None or matrix.empty:
        return pd.DataFrame()
    out = {}
    # columns are pairs
    for col in matrix.columns:
        series = matrix[col].astype(float)
        # stepwise lower interpolation to match BAC semantics
        out[col] = [float(series.quantile(p, interpolation="lower")) for p in probs]
    df = pd.DataFrame(
        out,
        index=[f"p{int(p * 10000) / 100 if p < 1 else int(p * 100)}" for p in probs],
    )
    # transpose to have rows as pairs, columns as tails
    return df.T


def compute_pair_matrices(
    results: dict, include_maxflow: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Returns (tm_abs, tm_norm, mf_abs, mf_norm), where each is a DataFrame with
    rows as pairs "s→d" and columns as percentiles (p50,p90,p99,p999,p9999).
    Normalization uses tm_placement baseline per-pair demand; values are clipped at 1.0 for interpretability.
    If include_maxflow is False or data missing, mf_* will be None.
    """
    probs = [0.50, 0.90, 0.99, 0.999, 0.9999]
    denom = _baseline_demand_map_from_tm(results)

    tm_mat = _collect_per_iteration_matrix(results, "tm_placement")
    tm_abs = _percentiles_per_pair(tm_mat, probs)
    # normalized
    if not tm_mat.empty and denom:
        norm_vals = tm_mat.copy()
        for col in norm_vals.columns:
            # parse pair
            try:
                s, d = col.split("→", 1)
                den = float(denom.get((s, d), float("nan")))
            except Exception:
                den = float("nan")
            if np.isfinite(den) and den > 0.0:
                norm_vals[col] = (norm_vals[col].astype(float) / den).clip(upper=1.0)
            else:
                norm_vals[col] = np.nan
        tm_norm = _percentiles_per_pair(norm_vals, probs)
    else:
        tm_norm = pd.DataFrame()

    mf_abs: Optional[pd.DataFrame] = None
    mf_norm: Optional[pd.DataFrame] = None
    if include_maxflow:
        mf_mat = _collect_per_iteration_matrix(results, "node_to_node_capacity_matrix")
        mf_abs = _percentiles_per_pair(mf_mat, probs)
        if not mf_mat.empty and denom:
            n2 = mf_mat.copy()
            for col in n2.columns:
                try:
                    s, d = col.split("→", 1)
                    den = float(denom.get((s, d), float("nan")))
                except Exception:
                    den = float("nan")
                if np.isfinite(den) and den > 0.0:
                    n2[col] = (n2[col].astype(float) / den).clip(upper=1.0)
                else:
                    n2[col] = np.nan
            mf_norm = _percentiles_per_pair(n2, probs)
        else:
            mf_norm = pd.DataFrame()

    return tm_abs, tm_norm, mf_abs, mf_norm
