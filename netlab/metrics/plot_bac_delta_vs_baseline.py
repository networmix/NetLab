"""Plot pooled bandwidth availability relative to a baseline scenario.

Read ``<scenario>/seed*/bac.json``, normalize delivery by baseline bandwidth,
and compare availability over the selected percentage range.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .distributions import availability_curve, curve_on_grid
from .seed_data import select_baseline

LegendLocation = Literal[
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]


def _list_scenario_dirs(analysis_root: Path) -> list[Path]:
    return [
        p
        for p in sorted(analysis_root.iterdir())
        if p.is_dir() and not p.name.startswith("_")
    ]


def _pooled_availability_curve(scen_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted bandwidth thresholds (% of baseline) and pooled availability."""
    pooled: list[float] = []
    for seed_dir in sorted(scen_dir.glob("seed*")):
        bac_path = seed_dir / "bac.json"
        if not bac_path.exists():
            continue
        try:
            data = json.loads(bac_path.read_text(encoding="utf-8"))
        except (ValueError, TypeError, KeyError, OSError):
            continue
        try:
            offered = float(data.get("offered", float("nan")))
            series = [float(x) for x in (data.get("series", []) or [])]
        except (ValueError, TypeError, KeyError, OSError):
            continue
        if not series or not np.isfinite(offered) or offered <= 0.0:
            continue
        norm = (np.asarray(series, dtype=float) / offered) * 100.0
        pooled.extend([float(v) for v in norm if np.isfinite(v)])

    if not pooled:
        return np.array([], dtype=float), np.array([], dtype=float)

    return availability_curve(np.asarray(pooled, dtype=float))


def plot_bac_delta_vs_baseline(
    analysis_root: Path,
    *,
    baseline: Optional[str] = None,
    only: Optional[Iterable[str]] = None,
    grid_min: float = 80.0,
    grid_max: float = 100.0,
    legend_loc: LegendLocation = "upper left",
    save_to: Optional[Path] = None,
) -> Optional[Path]:
    """Save availability differences over the inclusive bandwidth-percentage range.

    ``only`` selects scenarios to compare with the chosen baseline. ``save_to``
    defaults to analysis_root/BAC_delta_vs_baseline.png. Return None without data.
    """
    analysis_root = analysis_root.resolve()
    scen_dirs = _list_scenario_dirs(analysis_root)
    if not scen_dirs:
        return None
    base_name = select_baseline([p.name for p in scen_dirs], baseline)
    if only:
        only_set = set(only)
        scen_dirs = [p for p in scen_dirs if p.name in only_set or p.name == base_name]
    if not scen_dirs:
        return None

    base_dir = analysis_root / base_name
    comp_dirs = [p for p in scen_dirs if p != base_dir]
    if not comp_dirs:
        return None

    base_x, base_a = _pooled_availability_curve(base_dir)
    if base_x.size == 0:
        return None

    curves = {sd.name: _pooled_availability_curve(sd) for sd in comp_dirs}
    if grid_min >= grid_max:
        raise ValueError("grid_min must be below grid_max")
    observations = np.concatenate([base_x, *(curve[0] for curve in curves.values())])
    grid = np.unique(
        np.r_[
            grid_min,
            observations[(observations >= grid_min) & (observations <= grid_max)],
            grid_max,
        ]
    )
    base_on_grid = curve_on_grid(base_x, base_a, grid)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for sd in comp_dirs:
        sx, sa = curves[sd.name]
        if sx.size == 0:
            continue
        s_on_grid = curve_on_grid(sx, sa, grid)
        delta = s_on_grid - base_on_grid
        ax.step(grid, delta, where="pre", label=sd.name)

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
