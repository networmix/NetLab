"""Plot pooled bandwidth availability across seeds.

Samples are normalized by baseline delivery without clipping. Pooling gives
each sample equal weight, so seeds with more iterations contribute more weight.
With at least three seeds, the plot includes an IQR band of per-seed curves.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .distributions import availability_curve, curve_on_grid


def _seed_availability_on_grid(samples: np.ndarray, grid_pct: np.ndarray) -> np.ndarray:
    return curve_on_grid(*availability_curve(samples), grid_pct)


def _load_seed_bac(seed_dir: Path) -> Tuple[np.ndarray, float]:
    """Return (normalized_samples_pct, offered) for a single seed.

    Normalized samples are delivered/offered * 100.
    """
    p = seed_dir / "bac.json"
    if not p.exists():
        return np.array([], dtype=float), float("nan")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, TypeError, KeyError, OSError):
        return np.array([], dtype=float), float("nan")
    offered = float(data.get("offered", float("nan")))
    series = data.get("series", []) or []
    vals = []
    for v in series:
        try:
            vv = float(v)
            vals.append(vv)
        except (ValueError, TypeError, KeyError, OSError):
            continue
    arr = np.asarray(vals, dtype=float)
    if not (math.isfinite(offered) and offered > 0.0) or arr.size == 0:
        return np.array([], dtype=float), float("nan")
    norm = (arr / offered) * 100.0
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

    fig, ax = plt.subplots()
    palette = sns.color_palette("tab10", n_colors=len(scen_dirs))

    max_pct = 100.0
    plotted = False
    for i, sd in enumerate(scen_dirs):
        seed_dirs = sorted([p for p in sd.glob("seed*") if p.is_dir()])
        pooled: List[float] = []
        seed_samples: List[np.ndarray] = []
        for sdir in seed_dirs:
            samples_pct, _off = _load_seed_bac(sdir)
            if samples_pct.size == 0:
                continue
            pooled.extend(samples_pct.tolist())
            seed_samples.append(samples_pct)

        if not pooled:
            continue
        max_pct = max(max_pct, max(pooled))
        plotted = True
        color = palette[i % len(palette)]
        pooled_arr = np.asarray(pooled, dtype=float)
        grid = np.unique(np.r_[0.0, pooled_arr, max(100.0, max(pooled))])
        seed_curves = [
            _seed_availability_on_grid(samples, grid) for samples in seed_samples
        ]
        x_sorted = grid
        a_sorted = curve_on_grid(*availability_curve(pooled_arr), grid)
        ax.step(
            x_sorted, a_sorted, where="pre", label=sd.name, color=color, linewidth=2.0
        )

        # IQR band across seeds on the common grid
        if len(seed_curves) >= 3:
            mat = np.vstack(seed_curves)
            q25 = np.nanpercentile(mat, 25, axis=0)
            q75 = np.nanpercentile(mat, 75, axis=0)
            ax.fill_between(
                grid, q25, q75, step="pre", color=color, alpha=0.12, linewidth=0
            )
        for sc in seed_curves:
            ax.step(grid, sc, where="pre", color=color, alpha=0.12, linewidth=0.6)

    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("Delivered bandwidth (% of offered)")
    ax.set_ylabel("Availability  (≥ x)")
    ax.set_xlim(0.0, max_pct)
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
