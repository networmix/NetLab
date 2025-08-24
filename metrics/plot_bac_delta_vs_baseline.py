"""Cross-seed BAC delta vs baseline (publishable figure).

This module computes pooled BAC availability curves per scenario from seed-level
bac.json outputs, then plots the availability delta relative to a chosen
baseline over a configurable delivered-bandwidth range (default: 80–100%).

Inputs:
  analysis_root/
    <scenario>/seed<SEED>/bac.json  (as written by analysis.py)

Outputs:
  PNG figure saved to the requested path (or analysis_root by default).

Notes:
  - Normalization follows the project convention: delivered/offered, capped at 1.
  - Pooled empirical approach: combine normalized samples across seeds, then
    availability = 1 − CDF.
  - The x-axis is percent of offered (0..100); we restrict to [grid_min, grid_max].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _list_scenario_dirs(analysis_root: Path) -> list[Path]:
    return [
        p
        for p in sorted(analysis_root.iterdir())
        if p.is_dir() and not p.name.startswith("_")
    ]


def _pooled_availability_curve(scen_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return pooled availability curve for a scenario.

    Returns:
        (xs_pct, availability) where xs_pct is sorted in [0, 100].
    """
    pooled: list[float] = []
    for seed_dir in sorted(scen_dir.glob("seed*")):
        bac_path = seed_dir / "bac.json"
        if not bac_path.exists():
            continue
        try:
            data = json.loads(bac_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            offered = float(data.get("offered", float("nan")))
            series = [float(x) for x in (data.get("series", []) or [])]
        except Exception:
            continue
        if not series or not np.isfinite(offered) or offered <= 0.0:
            continue
        norm = np.minimum(np.asarray(series, dtype=float) / offered, 1.0) * 100.0
        pooled.extend([float(v) for v in norm if np.isfinite(v)])

    if not pooled:
        return np.array([], dtype=float), np.array([], dtype=float)

    xs = np.sort(np.asarray(pooled, dtype=float))
    cdf = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    availability = 1.0 - cdf
    return xs, availability


def plot_bac_delta_vs_baseline(
    analysis_root: Path,
    *,
    baseline: Optional[str] = "baseline_SingleRouter",
    only: Optional[Iterable[str]] = None,
    grid_min: float = 80.0,
    grid_max: float = 100.0,
    legend_loc: str = "upper left",
    save_to: Optional[Path] = None,
) -> Optional[Path]:
    """Plot BAC Δ-availability vs baseline over [grid_min, grid_max].

    Args:
        analysis_root: Root with per-scenario metrics (e.g., scenarios_metrics).
        baseline: Scenario name to use as baseline; if None or missing, the
            first scenario in alphabetical order is used.
        only: Optional list of scenario names to include (besides baseline).
        grid_min: Lower bound on delivered percent (x-axis), inclusive.
        grid_max: Upper bound on delivered percent (x-axis), inclusive.
        legend_loc: Matplotlib legend loc string (e.g., "upper left").
        save_to: Path to save the figure; defaults to analysis_root/BAC_delta_vs_baseline.png.

    Returns:
        The path to the saved figure, or None if no data.
    """
    analysis_root = analysis_root.resolve()
    scen_dirs = _list_scenario_dirs(analysis_root)
    if only:
        only_set = set(only)
        scen_dirs = [p for p in scen_dirs if p.name in only_set]
    if not scen_dirs:
        return None

    # Determine baseline directory
    base_dir: Optional[Path] = None
    if baseline:
        for p in scen_dirs:
            if p.name == baseline:
                base_dir = p
                break
    if base_dir is None:
        base_dir = scen_dirs[0]
    # Non-baseline scenarios
    comp_dirs = [p for p in scen_dirs if p != base_dir]
    if not comp_dirs:
        return None

    base_x, base_a = _pooled_availability_curve(base_dir)
    if base_x.size == 0:
        return None

    grid = np.linspace(
        float(grid_min), float(grid_max), 1 + int(4 * (grid_max - grid_min))
    )
    base_on_grid = np.interp(
        grid,
        base_x,
        base_a,
        left=(base_a[0] if base_a.size else 0.0),
        right=(base_a[-1] if base_a.size else 0.0),
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for sd in comp_dirs:
        sx, sa = _pooled_availability_curve(sd)
        if sx.size == 0:
            continue
        s_on_grid = np.interp(grid, sx, sa, left=sa[0], right=sa[-1])
        delta = s_on_grid - base_on_grid
        ax.plot(grid, delta, label=sd.name)

    ax.axhline(0.0, color="black", linewidth=0.8)
    for v in (80.0, 90.0, 95.0):
        if v >= grid_min and v <= grid_max:
            ax.axvline(v, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Delivered (% of offered)")
    ax.set_ylabel("Δ availability vs baseline")
    ax.set_title("BAC Δ-availability vs baseline")
    ax.legend(loc=legend_loc, frameon=True)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_xlim(float(grid_min), float(grid_max))

    out_path = (
        save_to
        if save_to is not None
        else (analysis_root / "BAC_delta_vs_baseline.png")
    )
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:  # pragma: no cover - convenience CLI
    import argparse

    ap = argparse.ArgumentParser(description="Plot BAC Δ-availability vs baseline")
    ap.add_argument(
        "analysis_root",
        type=str,
        help="Root with per-scenario metrics (e.g., scenarios_metrics)",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default="baseline_SingleRouter",
        help="Baseline scenario name",
    )
    ap.add_argument(
        "--only", type=str, default="", help="Comma-separated scenarios to include"
    )
    ap.add_argument("--xmin", type=float, default=80.0, help="Lower x bound (percent)")
    ap.add_argument("--xmax", type=float, default=100.0, help="Upper x bound (percent)")
    ap.add_argument("--legend", type=str, default="upper left", help="Legend location")
    ap.add_argument("--save", type=str, default="", help="Output figure path")
    args = ap.parse_args()

    root = Path(args.analysis_root)
    only: Optional[list[str]] = None
    if args.only.strip():
        only = [s.strip() for s in args.only.split(",") if s.strip()]
    out: Optional[Path] = None
    if args.save.strip():
        out = Path(args.save)
    res = plot_bac_delta_vs_baseline(
        root,
        baseline=args.baseline or None,
        only=only,
        grid_min=float(args.xmin),
        grid_max=float(args.xmax),
        legend_loc=args.legend,
        save_to=out,
    )
    if res is not None:
        print(f"Saved BAC Δ-availability figure → {res}")
    else:
        print("No data to plot.")


if __name__ == "__main__":  # pragma: no cover
    main()
