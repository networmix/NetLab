from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .common import expand_flow_results, pair_totals
from .distributions import availability_curve, curve_on_grid, threshold_at_probability


@dataclass
class BacResult:
    step_name: str
    mode: str  # 'placement' or 'maxflow'
    series: pd.Series  # delivered per iteration
    failure_ids: List[str]
    offered: float  # baseline delivered bandwidth
    quantiles_abs: Dict[float, float]
    quantiles_pct: Dict[float, float]  # normalized by baseline delivery, uncapped
    availability_at_pct_of_offer: Dict[float, float]  # {90: 0.97, ...}
    auc_normalized: float  # mean(min(delivered/offered, 1.0))
    bw_at_probability_abs: Dict[float, float]
    bw_at_probability_pct: Dict[float, float]
    per_flow: Dict[str, "BacResult"] = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        d = {
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
        if self.per_flow:
            d["per_flow"] = {k: v.to_jsonable() for k, v in self.per_flow.items()}
        return d


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


def _sum_delivered(iteration: dict) -> float:
    """Sum placed bandwidth across all flows in one iteration result."""
    return sum(pair_totals(iteration, "placed").values())


_QUANTILE_PROBS = (0.50, 0.90, 0.95, 0.99, 0.999, 0.9999)
_AVAIL_THRESHOLDS = (90.0, 95.0, 99.0, 99.9, 99.99)


def _compute_bac_stats(
    series: pd.Series, offered: float
) -> Tuple[
    Dict[float, float],  # quantiles_abs
    Dict[float, float],  # quantiles_pct
    Dict[float, float],  # availability_at_pct_of_offer
    float,  # auc_normalized
    Dict[float, float],  # bw_at_probability_abs
    Dict[float, float],  # bw_at_probability_pct
]:
    """Compute BAC statistics for an aggregate or directional bandwidth series."""
    q_abs = {
        p: float(series.quantile(p, interpolation="lower")) for p in _QUANTILE_PROBS
    }

    q_pct: Dict[float, float] = {}
    if offered > 0:
        for p in _QUANTILE_PROBS:
            val = float(series.quantile(p, interpolation="lower") / offered)
            q_pct[p] = val

    avail: Dict[float, float] = {}
    if offered > 0 and len(series) > 0:
        total = float(len(series))
        for pct in _AVAIL_THRESHOLDS:
            thr = (pct / 100.0) * offered
            avail[pct] = float((series >= thr).sum()) / total  # pyright: ignore[reportOperatorIssue]

    bw_abs: Dict[float, float] = {}
    bw_pct: Dict[float, float] = {}
    for p in _AVAIL_THRESHOLDS:
        t_abs = threshold_at_probability(series.to_numpy(), p)
        bw_abs[p] = t_abs
        bw_pct[p] = float(t_abs / offered) if offered > 0 else float("nan")

    auc_norm = float("nan")
    if offered > 0 and len(series) > 0:
        norm = series.astype(float) / offered
        auc_norm = float(norm.clip(upper=1.0).mean())

    return q_abs, q_pct, avail, auc_norm, bw_abs, bw_pct


def _flow_label(flow_source: str, flow_destination: str) -> str:
    """Build a directional label for pairwise or combine-mode endpoints.

    Flow source format: ``_src_<source_pattern>|<target_pattern>|<hash>``
    Returns label like ``abc1/rsw>xyz1/rsw``.
    """
    demand_id = flow_source.removeprefix("_src_")
    parts = demand_id.split("|")
    if (
        flow_source.startswith("_src_")
        and flow_destination == f"_snk_{demand_id}"
        and len(parts) == 3
    ):
        src_part = parts[0].strip("^$")
        dst_part = parts[1].strip("^$")
        return f"{src_part}>{dst_part}"
    return f"{flow_source}>{flow_destination}"


def compute_bac(results: dict, step_name: str, mode: str = "auto") -> BacResult:
    mode = _detect_mode(results, step_name, mode)
    data = _get_step(results, step_name)

    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError(f"{step_name}: data.baseline dict required")
    flow_results = data.get("flow_results", [])
    if not isinstance(flow_results, list):
        raise ValueError(f"flow_results must be a list for step: {step_name}")

    offered = _sum_delivered(baseline)
    if not np.isfinite(offered) or offered < 0:
        raise ValueError(f"{step_name}: baseline delivered must be finite and >= 0")

    expanded = expand_flow_results(flow_results)

    delivered = [offered]
    fids: List[str] = ["baseline"]
    for idx, it in enumerate(expanded):
        delivered.append(_sum_delivered(it))
        fids.append(str(it.get("failure_id", f"it{idx}")))

    s = pd.Series(delivered, dtype=float)
    s.index.name = "iteration"

    q_abs, q_pct, avail, auc_norm, bw_abs, bw_pct = _compute_bac_stats(s, offered)

    # Aggregate priority classes by source/destination pair.
    flow_map = {
        pair: placed
        for pair, placed in pair_totals(baseline, "placed").items()
        if placed > 0
    }
    labels = {pair: _flow_label(*pair) for pair in flow_map}
    label_counts = Counter(labels.values())

    per_flow: Dict[str, BacResult] = {}
    if len(flow_map) > 1:
        flow_series: Dict[Tuple[str, str], List[float]] = {
            pair: [bl_placed] for pair, bl_placed in flow_map.items()
        }

        for it in expanded:
            it_flows = pair_totals(it, "placed")
            for pair in flow_map:
                flow_series[pair].append(it_flows.get(pair, 0.0))

        for pair, bl_placed in flow_map.items():
            label = labels[pair]
            if label_counts[label] > 1:
                label = f"{label} {pair!r}"
            fs = pd.Series(flow_series[pair], dtype=float)
            fs.index.name = "iteration"
            fq_abs, fq_pct, favail, fauc, fbw_abs, fbw_pct = _compute_bac_stats(
                fs, bl_placed
            )
            per_flow[label] = BacResult(
                step_name=step_name,
                mode=mode,
                series=fs,
                failure_ids=list(fids),
                offered=float(bl_placed),
                quantiles_abs=fq_abs,
                quantiles_pct=fq_pct,
                availability_at_pct_of_offer=favail,
                auc_normalized=fauc,
                bw_at_probability_abs=fbw_abs,
                bw_at_probability_pct=fbw_pct,
            )

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
        per_flow=per_flow,
    )


def plot_bac(
    bac: BacResult, overlay: Optional[BacResult] = None, save_to: Optional[Path] = None
) -> None:
    x, a = availability_curve(bac.series.to_numpy())
    if bac.offered > 0:
        x_plot = (x / bac.offered) * 100.0
        x_label = "Delivered bandwidth (% of offered)"
    else:
        x_plot = x
        x_label = "Delivered bandwidth (Gbps)"

    grid = np.unique(
        np.r_[0.0, x_plot, max(100.0 if bac.offered > 0 else 1.0, max(x_plot))]
    )
    a = curve_on_grid(x_plot, a, grid)
    x_plot = grid
    plt.figure(figsize=(8, 5), dpi=300)
    sns.lineplot(x=x_plot, y=a, drawstyle="steps-pre", label=f"{bac.mode.capitalize()}")

    if overlay is not None:
        xo, ao = availability_curve(overlay.series.to_numpy())
        if bac.offered > 0 and overlay.offered > 0:
            xo = (xo / overlay.offered) * 100.0
        grid_o = np.unique(np.r_[0.0, xo, max(grid[-1], max(xo))])
        ao = curve_on_grid(xo, ao, grid_o)
        sns.lineplot(
            x=grid_o, y=ao, drawstyle="steps-pre", label=f"{overlay.mode.capitalize()}"
        )

    plt.xlabel(x_label)
    plt.ylabel("Availability  (≥ x)")
    plt.title(
        f"Bandwidth–Availability Curve — {bac.step_name}  (AUC={bac.auc_normalized * 100:.1f}%)"
    )
    plt.grid(True, linestyle=":", linewidth=0.5)
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to, dpi=300, bbox_inches="tight")
    plt.close()
