"""Compare iteration counts and timing across seeds and scenarios.

Each panel shows individual seeds, the scenario median, and its interquartile
range from the per-scenario iterops_summary.csv files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_METRICS = {
    "iters_fail": "Failure iterations",
    "unique_patterns": "Unique patterns",
    "tm_duration_per_iter_sec": "Seconds/iter",
}


def _load_iterops(
    analysis_root: Path, only: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    selected = set(only) if only else None
    frames = []
    for directory in sorted(analysis_root.iterdir()):
        if directory.name.startswith("_") or (
            selected is not None and directory.name not in selected
        ):
            continue
        path = directory / "iterops_summary.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)[list(_METRICS)].apply(pd.to_numeric)
        frames.append(frame.assign(scenario=directory.name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_cross_seed_iterops(
    analysis_root: Path,
    only: Optional[Iterable[str]] = None,
    save_to: Optional[Path] = None,
) -> Optional[Path]:
    data = _load_iterops(analysis_root.resolve(), only)
    if data.empty:
        return None

    fig, axes = plt.subplots(1, len(_METRICS), figsize=(14.0, 4.5))
    for ax, (column, label) in zip(axes, _METRICS.items(), strict=True):
        samples = data.loc[np.isfinite(data[column]), ["scenario", column]]
        if samples.empty:
            ax.axis("off")
            continue
        order = (
            samples.groupby("scenario")[column]
            .median()
            .sort_values(ascending=False)
            .index
        )
        sns.stripplot(
            data=samples,
            x="scenario",
            y=column,
            order=order,
            ax=ax,
            jitter=0.2,
            alpha=0.4,
            color="gray",
        )
        sns.pointplot(
            data=samples,
            x="scenario",
            y=column,
            order=order,
            ax=ax,
            estimator=np.median,
            errorbar=("pi", 50),
            linestyle="none",
            color="C0",
        )
        ax.set_title(label)
        ax.set_xlabel("scenario")
        ax.set_ylabel(label)
        ax.grid(True, linestyle=":", linewidth=0.5, axis="y")
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")

    fig.suptitle("Iteration counts and timing (medians and interquartile ranges)")
    fig.tight_layout()
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

    ap = argparse.ArgumentParser(
        description="Plot cross-scenario iteration counts and timing"
    )
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

    res = plot_cross_seed_iterops(root, only=only, save_to=out)
    if res is not None:
        print(f"Saved cross-scenario iterops figure → {res}")
    else:
        print("No iterops data to plot.")


if __name__ == "__main__":
    main()
