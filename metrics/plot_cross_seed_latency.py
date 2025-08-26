"""Cross-seed latency exceedance curves (publishable figure).

Approach: pooled empirical (like BAC). For each seed, take per-iteration
latency tail (e.g., p99 stretch) across failure iterations, pool across seeds,
and plot exceedance probability P(stretch > x) = 1 − CDF(stretch). Also show an
IQR band by computing seed-wise exceedance on a common x-grid.

Usage:
    python3 -m metrics.plot_cross_seed_latency scenarios_metrics \
        --metric p99 --save scenarios_metrics/_figures/latency_p99_cross_seed.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (8.0, 5.0)
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


def _availability_curve(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(samples, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs_sorted = np.sort(xs)
    cdf = np.arange(1, xs_sorted.size + 1, dtype=float) / float(xs_sorted.size)
    avail = 1.0 - cdf
    return xs_sorted, avail


def _load_seed_latency(seed_dir: Path, metric: str) -> np.ndarray:
    p = seed_dir / "latency.json"
    if not p.exists():
        return np.array([], dtype=float)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return np.array([], dtype=float)
    per_it = (data.get("per_iteration") or {}).get(metric)
    if not isinstance(per_it, list):
        return np.array([], dtype=float)
    vals = []
    for v in per_it:
        try:
            vv = float(v)
            if math.isfinite(vv):
                vals.append(vv)
        except Exception:
            continue
    return np.asarray(vals, dtype=float)


def plot_cross_seed_latency(
    analysis_root: Path,
    metric: str = "p99",
    only: Optional[Iterable[str]] = None,
    save_to: Optional[Path] = None,
) -> Optional[Path]:
    analysis_root = analysis_root.resolve()
    scen_dirs = [
        p
        for p in sorted(analysis_root.iterdir())
        if p.is_dir() and not p.name.startswith("_")
    ]
    if only:
        only_set = set(only)
        scen_dirs = [p for p in scen_dirs if p.name in only_set]
    if not scen_dirs:
        return None
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()
    palette = sns.color_palette("tab10", n_colors=len(scen_dirs))

    for i, sd in enumerate(scen_dirs):
        seed_dirs = sorted([p for p in sd.glob("seed*") if p.is_dir()])
        pooled: List[float] = []
        grid = np.linspace(1.0, 5.0, 401)  # typical latency stretch range
        seed_curves: List[np.ndarray] = []
        for sdir in seed_dirs:
            arr = _load_seed_latency(sdir, metric)
            if arr.size == 0:
                continue
            pooled.extend(arr.tolist())
            xs, a = _availability_curve(arr)
            # interpolate availability on common grid
            agrid = np.interp(grid, xs, a, left=a[0], right=a[-1])
            seed_curves.append(agrid)
        if not pooled:
            continue
        color = palette[i % len(palette)]
        x_sorted, a_sorted = _availability_curve(np.asarray(pooled, dtype=float))
        ax.step(
            x_sorted, a_sorted, where="post", label=sd.name, color=color, linewidth=2.0
        )
        if len(seed_curves) >= 3:
            mat = np.vstack(seed_curves)
            q25 = np.nanpercentile(mat, 25, axis=0)
            q75 = np.nanpercentile(mat, 75, axis=0)
            ax.fill_between(grid, q25, q75, color=color, alpha=0.12, linewidth=0)
        # Overlay per-seed curves lightly to show distribution
        for sc in seed_curves:
            ax.step(grid, sc, where="post", color=color, alpha=0.12, linewidth=0.6)

    ax.set_xlabel("Latency stretch (× baseline)")
    ax.set_ylabel("Exceedance probability (stretch > x)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Scenario", loc="lower right", frameon=True)
    ax.set_title(f"Cross-seed latency exceedance (metric={metric})")
    if save_to is not None:
        save_to = save_to.resolve()
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
        plt.close(fig)
        return save_to
    plt.show()
    return None


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Plot cross-seed latency availability")
    ap.add_argument(
        "analysis_root",
        type=str,
        help="Root with per-scenario metrics (e.g., scenarios_metrics)",
    )
    ap.add_argument(
        "--metric", type=str, default="p99", help="Latency metric: p50|p95|p99|WES"
    )
    ap.add_argument(
        "--only", type=str, default="", help="Comma-separated scenarios to include"
    )
    ap.add_argument("--save", type=str, default="", help="Output figure path")
    args = ap.parse_args()

    root = Path(args.analysis_root)
    only: Optional[List[str]] = None
    if args.only.strip():
        only = [s.strip() for s in args.only.split(",") if s.strip()]
    out: Optional[Path] = None
    if args.save.strip():
        out = Path(args.save)
    res = plot_cross_seed_latency(root, metric=args.metric, only=only, save_to=out)
    if res is not None:
        print(f"Saved cross-seed latency figure → {res}")
    else:
        print("No latency data to plot.")


if __name__ == "__main__":
    main()
