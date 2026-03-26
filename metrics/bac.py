from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .common import expand_flow_results


@dataclass
class BacResult:
    step_name: str
    mode: str  # 'placement' or 'maxflow'
    series: pd.Series  # delivered per iteration
    failure_ids: List[str]
    offered: float  # baseline delivered bandwidth
    quantiles_abs: Dict[float, float]
    quantiles_pct: Dict[float, float]  # normalized by offered (0..1)
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
    total = 0.0
    for rec in iteration.get("flows", []) or []:
        src = rec.get("source", "")
        dst = rec.get("destination", "")
        if not src or not dst or src == dst:
            continue
        total += float(rec.get("placed", 0.0))
    return total


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
    """Compute all BAC statistics from a delivered-bandwidth series.

    This is the single source of truth for BAC math. Used for both
    aggregate and per-flow computation.
    """
    q_abs = {
        p: float(series.quantile(p, interpolation="lower")) for p in _QUANTILE_PROBS
    }

    q_pct: Dict[float, float] = {}
    if offered > 0:
        for p in _QUANTILE_PROBS:
            val = float(series.quantile(p, interpolation="lower") / offered)
            q_pct[p] = float(min(val, 1.0))

    avail: Dict[float, float] = {}
    if offered > 0 and len(series) > 0:
        total = float(len(series))
        for pct in _AVAIL_THRESHOLDS:
            thr = (pct / 100.0) * offered
            avail[pct] = float((series >= thr).sum()) / total  # pyright: ignore[reportOperatorIssue]

    bw_abs: Dict[float, float] = {}
    bw_pct: Dict[float, float] = {}
    for p in _AVAIL_THRESHOLDS:
        q = max(0.0, 1.0 - (p / 100.0))
        try:
            t_abs = float(series.quantile(q, interpolation="lower"))
        except Exception:
            t_abs = float("nan")
        bw_abs[p] = t_abs
        bw_pct[p] = float(t_abs / offered) if offered > 0 else float("nan")

    auc_norm = 1.0
    if offered > 0 and len(series) > 0:
        norm = series.astype(float) / offered
        auc_norm = float(norm.clip(upper=1.0).mean())

    return q_abs, q_pct, avail, auc_norm, bw_abs, bw_pct


def _flow_label(flow_source: str) -> str:
    """Extract a readable directional label from a flow's source field.

    Flow source format: ``_src_<source_pattern>|<target_pattern>|<hash>``
    Returns label like ``abc1/rsw>xyz1/rsw``.
    """
    demand_id = flow_source.removeprefix("_src_").removeprefix("_snk_")
    parts = demand_id.split("|")
    if len(parts) >= 2:
        src_part = parts[0].strip("^$")
        dst_part = parts[1].strip("^$")
        return f"{src_part}>{dst_part}"
    return demand_id[:30]


def compute_bac(results: dict, step_name: str, mode: str = "auto") -> BacResult:
    mode = _detect_mode(results, step_name, mode)
    data = _get_step(results, step_name)

    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError(f"{step_name}: data.baseline dict required")
    flow_results = data.get("flow_results", [])
    if not isinstance(flow_results, list) or not flow_results:
        raise ValueError(f"No flow_results for step: {step_name}")

    # Baseline determines offered bandwidth
    offered = _sum_delivered(baseline)
    if not np.isfinite(offered) or offered <= 0:
        raise ValueError(f"{step_name}: baseline delivered must be finite and > 0")

    # Expand deduplicated patterns by occurrence_count
    expanded = expand_flow_results(flow_results)

    # ── Aggregate series ──
    delivered = [offered]
    fids: List[str] = ["baseline"]
    for idx, it in enumerate(expanded):
        delivered.append(_sum_delivered(it))
        fids.append(str(it.get("failure_id", f"it{idx}")))

    s = pd.Series(delivered, dtype=float)
    s.index.name = "iteration"

    q_abs, q_pct, avail, auc_norm, bw_abs, bw_pct = _compute_bac_stats(s, offered)

    # ── Per-flow series ──
    # Build baseline per-flow map: source_field → (label, baseline_placed)
    flow_map: Dict[str, Tuple[str, float]] = {}
    for rec in baseline.get("flows", []) or []:
        src = rec.get("source", "")
        dst = rec.get("destination", "")
        if not src or not dst or src == dst:
            continue
        placed = float(rec.get("placed", 0.0))
        if placed <= 0:
            continue
        flow_map[src] = (_flow_label(src), placed)

    per_flow: Dict[str, BacResult] = {}
    if len(flow_map) > 1:
        # Only compute per-flow when there are multiple flows to separate
        flow_series: Dict[str, List[float]] = {
            src: [bl_placed] for src, (_label, bl_placed) in flow_map.items()
        }

        for it in expanded:
            it_flows = {f["source"]: f for f in it.get("flows", []) or []}
            for src, (_label, _bl_placed) in flow_map.items():
                if src in it_flows:
                    flow_series[src].append(float(it_flows[src].get("placed", 0.0)))
                else:
                    flow_series[src].append(0.0)

        for src, (label, bl_placed) in flow_map.items():
            fs = pd.Series(flow_series[src], dtype=float)
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
