"""Cross-seed BAC aggregation and plotting (publishable figure).

Approach (sound for our use case): pooled empirical BAC per scenario.

Rationale:
- Each seed runs the same Monte Carlo failure policy with the same number of
  iterations (≈100). Each iteration is a draw from the same distribution of
  failure patterns conditional on the scenario/topology.
- Pooling all normalized delivered samples across seeds uniformly estimates the
  population CDF of delivered/offered under that scenario. The availability
  curve is then 1 − CDF. With identical iteration counts, this implicitly gives
  equal weight per seed.
- This avoids the artifact of positional medians across seeds and yields a
  high-resolution tail (more samples ⇒ smoother BAC), which is critical around
  p90–p99.

For uncertainty visualization, we also compute per-seed availability curves on
an x-grid and show the cross-seed IQR band.

Usage:
    python -m metrics.plot_cross_seed_bac scenarios_metrics \
        --save scenarios_metrics/_figures/bac_all_cross_seed.png
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


def _availability_curve_from_samples(
    samples: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(samples, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs_sorted = np.sort(xs)
    cdf = np.arange(1, xs_sorted.size + 1, dtype=float) / float(xs_sorted.size)
    availability = 1.0 - cdf
    return xs_sorted, availability


def _seed_availability_on_grid(samples: np.ndarray, grid_pct: np.ndarray) -> np.ndarray:
    """Compute seed availability on a fixed x-grid (percent)."""
    xs, a = _availability_curve_from_samples(samples)
    if xs.size == 0:
        return np.full_like(grid_pct, fill_value=np.nan, dtype=float)
    # xs already in percent; step-post interpolation
    # For each grid x, availability is a(x) at the largest xs <= x
    return np.interp(grid_pct, xs, a, left=a[0], right=a[-1])


def _load_seed_bac(seed_dir: Path) -> Tuple[np.ndarray, float]:
    """Return (normalized_samples_pct, offered) for a single seed.

    Normalized samples are min(delivered/offered, 1.0) * 100.
    """
    p = seed_dir / "bac.json"
    if not p.exists():
        return np.array([], dtype=float), float("nan")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return np.array([], dtype=float), float("nan")
    offered = float(data.get("offered", float("nan")))
    series = data.get("series", []) or []
    vals = []
    for v in series:
        try:
            vv = float(v)
            vals.append(vv)
        except Exception:
            continue
    arr = np.asarray(vals, dtype=float)
    if not (math.isfinite(offered) and offered > 0.0) or arr.size == 0:
        return np.array([], dtype=float), float("nan")
    norm = np.minimum(arr / offered, 1.0) * 100.0
    return norm, offered


def plot_cross_seed_bac(
    analysis_root: Path,
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
        # Gather seeds
        seed_dirs = sorted([p for p in sd.glob("seed*") if p.is_dir()])
        pooled: List[float] = []
        seed_curves: List[np.ndarray] = []
        grid = np.linspace(0.0, 100.0, 401)
        for sdir in seed_dirs:
            samples_pct, _off = _load_seed_bac(sdir)
            if samples_pct.size == 0:
                continue
            pooled.extend(samples_pct.tolist())
            seed_curves.append(_seed_availability_on_grid(samples_pct, grid))

        if not pooled:
            continue
        color = palette[i % len(palette)]
        pooled_arr = np.asarray(pooled, dtype=float)
        x_sorted, a_sorted = _availability_curve_from_samples(pooled_arr)
        # Plot pooled curve
        ax.step(
            x_sorted, a_sorted, where="post", label=sd.name, color=color, linewidth=2.0
        )

        # IQR band across seeds on the common grid
        if len(seed_curves) >= 3:
            mat = np.vstack(seed_curves)
            q25 = np.nanpercentile(mat, 25, axis=0)
            q75 = np.nanpercentile(mat, 75, axis=0)
            ax.fill_between(grid, q25, q75, color=color, alpha=0.12, linewidth=0)

    ax.set_xlabel("Delivered bandwidth (% of offered)")
    ax.set_ylabel("Availability  (≥ x)")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Scenario", loc="lower right", frameon=True)
    ax.set_title("Cross-seed Bandwidth–Availability Curves")

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

    ap = argparse.ArgumentParser(description="Plot cross-seed BAC (pooled empirical)")
    ap.add_argument(
        "analysis_root",
        type=str,
        help="Root with per-scenario metrics (e.g., scenarios_metrics)",
    )
    ap.add_argument(
        "--only", type=str, default="", help="Comma-separated scenarios to include"
    )
    ap.add_argument(
        "--save", type=str, default="", help="Output figure path (PNG/JPG/SVG)"
    )
    args = ap.parse_args()

    root = Path(args.analysis_root)
    only: Optional[List[str]] = None
    if args.only.strip():
        only = [s.strip() for s in args.only.split(",") if s.strip()]
    out: Optional[Path] = None
    if args.save.strip():
        out = Path(args.save)

    res = plot_cross_seed_bac(root, only=only, save_to=out)
    if res is not None:
        print(f"Saved cross-seed BAC figure → {res}")
    else:
        print("No BAC data to plot.")


if __name__ == "__main__":
    main()
