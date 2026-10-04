from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import (
    baseline_demand_map,
    expand_flow_results,
    pair_totals,
    require_capacity_pairs,
)

Pair = Tuple[str, str]


def _collect_per_iteration_matrix(results: dict, step_name: str) -> pd.DataFrame:
    step = results.get("steps", {}).get(step_name, {}) or {}
    data = step.get("data", {}) or {}
    fr = expand_flow_results(data.get("flow_results", []) or [])
    by_iter: Dict[str, Dict[str, float]] = {}
    for idx, it in enumerate(fr):
        fid = f"it{idx}"
        row = {f"{s}→{d}": value for (s, d), value in pair_totals(it, "placed").items()}
        by_iter[fid] = row
    if not by_iter:
        return pd.DataFrame()
    df = pd.DataFrame(list(by_iter.values()), index=list(by_iter)).fillna(0.0)
    demand_pairs = {f"{s}→{d}" for s, d in baseline_demand_map(results)}
    df = df.reindex(columns=sorted(demand_pairs | set(df.columns)), fill_value=0.0)
    df.index.name = "iteration"
    return df


def _percentiles_per_pair(matrix: pd.DataFrame, probs: List[float]) -> pd.DataFrame:
    if matrix is None or matrix.empty:
        return pd.DataFrame()
    out = {}
    for col in matrix.columns:
        series = matrix[col].astype(float)
        # Descriptive quantiles use lower ranks; BW@p uses a survival threshold.
        out[col] = [float(series.quantile(p, interpolation="lower")) for p in probs]
    df = pd.DataFrame(
        out,
        index=[f"p{int(p * 10000) / 100 if p < 1 else int(p * 100)}" for p in probs],
    )
    return df.T


def compute_pair_matrices(
    results: dict, include_maxflow: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Return placement and MaxFlow percentile tables: tm_abs, tm_norm, mf_abs, mf_norm.

    Rows are exact "source→destination" pairs. Columns are p50.0, p90.0, p99.0, p99.9,
    and p99.99 using NumPy's lower quantile estimator over failures only.
    Normalized values divide by baseline placement demand and are capped at 1.
    The MaxFlow tables are None when disabled; enabled analysis requires matching pairs.
    """
    probs = [0.50, 0.90, 0.99, 0.999, 0.9999]
    denom = baseline_demand_map(results)

    tm_mat = _collect_per_iteration_matrix(results, "tm_placement")
    tm_abs = _percentiles_per_pair(tm_mat, probs)
    if not tm_mat.empty and denom:
        norm_vals = tm_mat.copy()
        for col in norm_vals.columns:
            try:
                s, d = col.split("→", 1)
                den = float(denom.get((s, d), float("nan")))
            except (ValueError, TypeError, KeyError, OSError):
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
        require_capacity_pairs(results)
        mf_mat = _collect_per_iteration_matrix(results, "node_to_node_capacity_matrix")
        mf_abs = _percentiles_per_pair(mf_mat, probs)
        if not mf_mat.empty and denom:
            n2 = mf_mat.copy()
            for col in n2.columns:
                try:
                    s, d = col.split("→", 1)
                    den = float(denom.get((s, d), float("nan")))
                except (ValueError, TypeError, KeyError, OSError):
                    den = float("nan")
                if np.isfinite(den) and den > 0.0:
                    n2[col] = (n2[col].astype(float) / den).clip(upper=1.0)
                else:
                    n2[col] = np.nan
            mf_norm = _percentiles_per_pair(n2, probs)
        else:
            mf_norm = pd.DataFrame()

    return tm_abs, tm_norm, mf_abs, mf_norm
