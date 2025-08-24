from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class LatencyResult:
    """Latency stretch summary for a single seed.

    Attributes:
        baseline: Volume-weighted tails for the baseline iteration (keys: p50, p95, p99,
            SLO_1_2, SLO_1_5, best_path_share, WES).
        failures: Per-tail medians across failure iterations (same keys as baseline).
        derived: Seed-level derived metrics (e.g., TD99, SLO drops, best_path_drop, WES_delta).
        per_iteration: Optional per-iteration tails for failures (keys as above), each a list
            with one value per failure iteration in original order. Useful for pooled
            cross-seed aggregations and uncertainty bands.
    """

    baseline: Dict[str, float]
    failures: Dict[str, float]
    derived: Dict[str, float]
    per_iteration: Dict[str, list[float]] | None = None

    def to_jsonable(self) -> dict:
        return {
            "baseline": self.baseline,
            "failures": self.failures,
            "derived": self.derived,
            **({"per_iteration": self.per_iteration} if self.per_iteration else {}),
        }


def _canonical_dc(endpoint: str) -> str:
    """
    Normalize endpoint identifiers to a canonical DC-level path 'metroX/dcY'.
    Examples:
      - 'metro1/dc1'           → 'metro1/dc1'
      - 'metro1/dc1/dc/dc'     → 'metro1/dc1'
      - 'metro1/dc1/rack/node' → 'metro1/dc1'
    """
    if not endpoint:
        return endpoint
    parts = endpoint.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return endpoint


def _baseline_cost_per_pair_tm(results: dict) -> Dict[Tuple[str, str], float]:
    """
    Build per-pair baseline costs strictly from tm_placement baseline.
    The baseline iteration must be present (validated upstream) and appear first.
    For each pair, choose the minimum cost with non-zero volume from cost_distribution.
    """
    per_pair: Dict[Tuple[str, str], float] = {}
    tm_step = results.get("steps", {}).get("tm_placement", {}) or {}
    tm_data = tm_step.get("data", {}) or {}
    fr = tm_data.get("flow_results", []) or []
    if not isinstance(fr, list) or not fr:
        return per_pair
    base = fr[0] if str(fr[0].get("failure_id", "")) == "baseline" else None
    if not base:
        # Upstream validation should prevent this; return empty to be safe
        return per_pair
    for rec in base.get("flows", []) or []:
        s = _canonical_dc(rec.get("source", ""))
        d = _canonical_dc(rec.get("destination", ""))
        if not s or not d or s == d:
            continue
        cdist = rec.get("cost_distribution", {}) or {}
        if not cdist:
            continue
        try:
            min_cost = min([float(k) for k, v in cdist.items() if float(v) > 0.0])
            per_pair[(s, d)] = min_cost
        except Exception:
            continue
    return per_pair


def compute_latency_stretch(results: dict) -> LatencyResult:
    # Validate baseline metadata and ordering for tm_placement
    tm_meta = results.get("steps", {}).get("tm_placement", {}).get("metadata", {}) or {}
    if bool(tm_meta.get("baseline")) is not True:
        raise ValueError(
            "tm_placement.metadata.baseline must be true and baseline must be included"
        )
    tm_data = results.get("steps", {}).get("tm_placement", {}).get("data", {}) or {}
    fr_chk = tm_data.get("flow_results", []) or []
    if not isinstance(fr_chk, list) or not fr_chk:
        raise ValueError("tm_placement.data.flow_results missing or empty")
    if str(fr_chk[0].get("failure_id", "")) != "baseline":
        raise ValueError(
            "tm_placement baseline must be first (flow_results[0].failure_id == 'baseline')"
        )

    base_cost = _baseline_cost_per_pair_tm(results)
    if not base_cost:
        return LatencyResult(baseline={}, failures={}, derived={})

    tm = results.get("steps", {}).get("tm_placement", {}).get("data", {}) or {}
    fr = tm.get("flow_results", []) or []
    if not isinstance(fr, list) or not fr:
        return LatencyResult(baseline={}, failures={}, derived={})

    def _iter_metrics(it: dict) -> Dict[str, float]:
        flows = it.get("flows", []) or []
        vals: list[float] = []
        wts: list[float] = []
        best_wts: float = 0.0
        for rec in flows:
            s = _canonical_dc(rec.get("source", ""))
            d = _canonical_dc(rec.get("destination", ""))
            if not s or not d or s == d:
                continue
            denom = base_cost.get((s, d), None)
            if denom is None or denom <= 0:
                continue
            cdist = rec.get("cost_distribution", {}) or {}
            for k, v in cdist.items():
                try:
                    c = float(k)
                    vol = float(v)
                except Exception:
                    continue
                if vol <= 0:
                    continue
                stretch = c / float(denom)
                vals.append(stretch)
                wts.append(vol)
                if abs(c - float(denom)) <= max(
                    1e-6, 1e-9 * max(abs(c), abs(float(denom)), 1.0)
                ):
                    best_wts += vol
        if not vals:
            return {}
        arr = np.asarray(vals, dtype=float)
        w = np.asarray(wts, dtype=float)
        order = np.argsort(arr)
        arr = arr[order]
        w = w[order]
        cw = np.cumsum(w)
        cw = cw / cw[-1]

        def wq(p: float) -> float:
            idx = int(np.searchsorted(cw, p, side="left"))
            idx = min(max(int(idx), 0), int(len(arr) - 1))
            return float(arr[idx])

        total_w = float(w.sum()) if w.size else 0.0
        comp_12 = (
            float((w[(arr <= 1.2)].sum() / total_w)) if total_w > 0 else float("nan")
        )
        comp_15 = (
            float((w[(arr <= 1.5)].sum() / total_w)) if total_w > 0 else float("nan")
        )
        best_share = float(best_wts / total_w) if total_w > 0 else float("nan")
        # Weighted Excess Stretch: E[(stretch-1)+]
        if total_w > 0:
            excess_sum = 0.0
            for i in range(int(len(arr))):
                excess = arr[i] - 1.0
                if excess > 0.0:
                    excess_sum += float(excess * w[i])
            wes = float(excess_sum / total_w)
        else:
            wes = float("nan")
        return {
            "p50": wq(0.50),
            "p95": wq(0.95),
            "p99": wq(0.99),
            "SLO_1_2": comp_12,
            "SLO_1_5": comp_15,
            "best_path_share": best_share,
            "WES": wes,
        }

    base_it = fr[0]
    baseline = _iter_metrics(base_it)

    failure_metrics: Dict[str, list[float]] = {
        k: []
        for k in ("p50", "p95", "p99", "SLO_1_2", "SLO_1_5", "best_path_share", "WES")
    }
    for it in fr[1:]:
        m = _iter_metrics(it)
        if not m:
            continue
        for k in failure_metrics.keys():
            v = float(m.get(k, float("nan")))
            if np.isfinite(v):
                failure_metrics[k].append(v)
    failures = {}
    for k, series in failure_metrics.items():
        failures[k] = (
            float(np.nanmedian(np.asarray(series, dtype=float)))
            if series
            else float("nan")
        )

    derived = {}
    for tail in ("p95", "p99"):
        b = float(baseline.get(tail, float("nan"))) if baseline else float("nan")
        f = float(failures.get(tail, float("nan"))) if failures else float("nan")
        derived[f"TD{tail[1:]}"] = (
            float(f / b)
            if (np.isfinite(f) and np.isfinite(b) and b > 0)
            else float("nan")
        )
    for thr in ("SLO_1_2", "SLO_1_5"):
        b = float(baseline.get(thr, float("nan"))) if baseline else float("nan")
        f = float(failures.get(thr, float("nan"))) if failures else float("nan")
        derived[f"{thr}_drop"] = (
            float(b - f) if (np.isfinite(b) and np.isfinite(f)) else float("nan")
        )
    b_bs = (
        float(baseline.get("best_path_share", float("nan")))
        if baseline
        else float("nan")
    )
    f_bs = (
        float(failures.get("best_path_share", float("nan")))
        if failures
        else float("nan")
    )
    derived["best_path_share_drop"] = (
        float(b_bs - f_bs)
        if (np.isfinite(b_bs) and np.isfinite(f_bs))
        else float("nan")
    )
    b_wes = float(baseline.get("WES", float("nan"))) if baseline else float("nan")
    f_wes = float(failures.get("WES", float("nan"))) if failures else float("nan")
    derived["WES_delta"] = (
        float(f_wes - b_wes)
        if (np.isfinite(b_wes) and np.isfinite(f_wes))
        else float("nan")
    )

    return LatencyResult(
        baseline=baseline,
        failures=failures,
        derived=derived,
        per_iteration={k: list(vs) for k, vs in failure_metrics.items()}
        if any(failure_metrics.values())
        else None,
    )


def plot_latency(lt: LatencyResult, save_to: Optional[Path] = None) -> None:
    if not (lt.baseline and lt.failures):
        return
    data = pd.DataFrame(
        [
            {
                "group": "baseline",
                **{k: lt.baseline.get(k) for k in ("p50", "p95", "p99")},
            },
            {
                "group": "failures",
                **{k: lt.failures.get(k) for k in ("p50", "p95", "p99")},
            },
        ]
    )
    tidy = data.melt(id_vars="group", var_name="tail", value_name="stretch").dropna()
    plt.figure()
    sns.barplot(data=tidy, x="tail", y="stretch", hue="group")
    plt.axhline(1.0, linestyle="--", linewidth=0.8)
    plt.xlabel("Tail")
    plt.ylabel("Latency stretch")
    plt.title("Latency stretch: baseline vs failures")
    plt.grid(True, linestyle=":", linewidth=0.5)
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to)
    plt.close()
