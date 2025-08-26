from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class BacResult:
    step_name: str
    mode: str  # 'placement' or 'maxflow'
    series: pd.Series  # delivered per iteration
    failure_ids: List[str]
    offered: float  # offered demand (if known; else = series.max())
    quantiles_abs: Dict[float, float]
    quantiles_pct: Dict[float, float]  # normalized by offered (0..1), if offered>0
    availability_at_pct_of_offer: Dict[float, float]  # {90: 0.97, 99.9: 0.88, ...}
    auc_normalized: float  # mean(min(delivered/offered,1.0))
    # Bandwidth threshold at probability p: smallest t s.t. P(delivered >= t) >= p
    # Absolute units (Gbps) and normalized by offered (0..1)
    bw_at_probability_abs: Dict[float, float]
    bw_at_probability_pct: Dict[float, float]

    def to_jsonable(self) -> dict:
        return {
            "step_name": self.step_name,
            "mode": self.mode,
            "series": list(map(float, self.series.values)),
            "failure_ids": list(self.failure_ids),
            "offered": float(self.offered),
            "quantiles_abs": {str(k): float(v) for k, v in self.quantiles_abs.items()},
            "quantiles_pct": {str(k): float(v) for k, v in self.quantiles_pct.items()},
            "availability_at_pct_of_offer": {
                str(k): float(v) for k, v in self.availability_at_pct_of_offer.items()
            },
            "auc_normalized": float(self.auc_normalized),
            "bw_at_probability_abs": {
                str(k): float(v) for k, v in self.bw_at_probability_abs.items()
            },
            "bw_at_probability_pct": {
                str(k): float(v) for k, v in self.bw_at_probability_pct.items()
            },
        }


def _get_step(results: dict, name: str) -> dict:
    return results.get("steps", {}).get(name, {}).get("data", {}) or {}


def _detect_mode(results: dict, step_name: str, mode: str) -> str:
    if mode != "auto":
        return mode
    st = results.get("workflow", {}).get(step_name, {}).get("step_type", "")
    if st == "TrafficMatrixPlacement":
        return "placement"
    if st == "MaxFlow":
        return "maxflow"
    return "placement"


def compute_bac(results: dict, step_name: str, mode: str = "auto") -> BacResult:
    mode = _detect_mode(results, step_name, mode)
    # Validate baseline metadata and ordering
    step_meta = results.get("steps", {}).get(step_name, {}).get("metadata", {}) or {}
    if bool(step_meta.get("baseline")) is not True:
        raise ValueError(
            f"{step_name}.metadata.baseline must be true and baseline must be included"
        )
    data = _get_step(results, step_name)
    flow_results = data.get("flow_results", [])
    if not isinstance(flow_results, list) or not flow_results:
        raise ValueError(f"No flow_results for step: {step_name}")
    first = flow_results[0]
    if str(first.get("failure_id", "")) != "baseline":
        raise ValueError(
            f"{step_name} baseline must be first (flow_results[0].failure_id == 'baseline')"
        )

    delivered = []
    demanded = []
    fids = []
    baseline_delivered: Optional[float] = None
    for it in flow_results:
        flows = it.get("flows", []) or []
        total_deliv = 0.0
        total_dem = 0.0
        for rec in flows:
            src = rec.get("source", "")
            dst = rec.get("destination", "")
            if not src or not dst or src == dst:
                continue
            placed = float(rec.get("placed", 0.0))
            demand = float(rec.get("demand", 0.0))
            total_deliv += placed
            total_dem += demand
        delivered.append(total_deliv)
        demanded.append(total_dem)
        fid = str(it.get("failure_id", f"it{len(fids)}"))
        fids.append(fid)
        if fid == "baseline":
            baseline_delivered = float(total_deliv)

    s = pd.Series(delivered, index=pd.Index(fids, name="failure_id"), dtype=float)
    # Normalize strictly by baseline delivered (no-failure). Require presence.
    if baseline_delivered is not None and np.isfinite(baseline_delivered):
        offered = float(baseline_delivered)
    else:
        raise ValueError(
            f"{step_name} baseline iteration missing or has non-finite delivered value"
        )

    probs = [0.50, 0.90, 0.95, 0.99, 0.999, 0.9999]
    q_abs = {p: float(s.quantile(p, interpolation="lower")) for p in probs}

    q_pct = {}
    if offered > 0:
        for p in probs:
            val = float(s.quantile(p, interpolation="lower") / offered)
            # Guard against rare >1 due to numerical noise or offered<iteration delivered
            q_pct[p] = float(min(val, 1.0))

    # Availability at thresholds (as fraction of iterations)
    avail = {}
    if offered > 0 and len(s) > 0:
        total = float(len(s))
        for pct in (90.0, 95.0, 99.0, 99.9, 99.99):
            thr = (pct / 100.0) * offered
            avail[pct] = float((s >= thr).sum()) / total

    # Bandwidth-at-probability (inverse availability)
    bw_abs: Dict[float, float] = {}
    bw_pct: Dict[float, float] = {}
    for p in (90.0, 95.0, 99.0, 99.9, 99.99):
        q = max(0.0, 1.0 - (p / 100.0))  # lower-tail quantile
        try:
            t_abs = float(s.quantile(q, interpolation="lower"))
        except Exception:
            t_abs = float("nan")
        bw_abs[p] = t_abs
        bw_pct[p] = float(t_abs / offered) if offered > 0 else float("nan")

    auc_norm = 1.0
    if offered > 0 and len(s) > 0:
        norm = s.astype(float) / offered
        auc_norm = float(norm.clip(upper=1.0).mean())

    return BacResult(
        step_name=step_name,
        mode=mode,
        series=s,
        failure_ids=list(fids),
        offered=float(offered),
        quantiles_abs=q_abs,
        quantiles_pct=q_pct,
        availability_at_pct_of_offer=avail,
        auc_normalized=auc_norm,
        bw_at_probability_abs=bw_abs,
        bw_at_probability_pct=bw_pct,
    )


def _availability_curve(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(series.values, dtype=float))
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    avail = 1.0 - cdf
    return xs, avail


def plot_bac(
    bac: BacResult, overlay: Optional[BacResult] = None, save_to: Optional[Path] = None
) -> None:
    x, a = _availability_curve(bac.series)
    if bac.offered > 0:
        x_plot = (x / bac.offered) * 100.0
        x_label = "Delivered bandwidth (% of offered)"
    else:
        x_plot = x
        x_label = "Delivered bandwidth (Gbps)"

    plt.figure()
    sns.lineplot(
        x=x_plot, y=a, drawstyle="steps-post", label=f"{bac.mode.capitalize()}"
    )

    if overlay is not None:
        xo, ao = _availability_curve(overlay.series)
        if bac.offered > 0 and overlay.offered > 0:
            xo = (xo / overlay.offered) * 100.0
        sns.lineplot(
            x=xo, y=ao, drawstyle="steps-post", label=f"{overlay.mode.capitalize()}"
        )

    plt.xlabel(x_label)
    plt.ylabel("Availability  (≥ x)")
    plt.title(
        f"Bandwidth–Availability Curve — {bac.step_name}  (AUC={bac.auc_normalized * 100:.1f}%)"
    )
    plt.grid(True, linestyle=":", linewidth=0.5)
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to)
    plt.close()
