from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_significance_heatmap(
    analysis_root: Path, save_to: Optional[Path] = None
) -> Optional[Path]:
    """Plot heatmap of normalized effect sizes with significance highlighting.

    Expects normalized insights at analysis_root/normalized_insights.csv with columns:
      scenario, <metric>__mean, <metric>__n, <metric>__p for multiple metrics.

    Effect size is mean-1.0 for ratio metrics (suffix _r__mean) and mean-0.0 for delta metrics (suffix _d__mean).
    Cells are annotated (dot overlay) where p < 0.05.
    """
    csv = analysis_root / "normalized_insights.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv).set_index("scenario")
    if df.empty:
        return None

    # Identify metric bases by scanning __mean columns
    metrics = []
    for c in df.columns:
        if c.endswith("__mean"):
            base = c[:-6]
            metrics.append(base)
    if not metrics:
        return None

    # Build effect size matrix and p-value matrix
    eff = pd.DataFrame(index=df.index, columns=metrics, dtype=float)
    pvals = pd.DataFrame(index=df.index, columns=metrics, dtype=float)
    for m in metrics:
        m_mean = f"{m}__mean"
        m_p = f"{m}__p_adj" if f"{m}__p_adj" in df.columns else f"{m}__p"
        if m_mean not in df.columns:
            continue
        vals = pd.to_numeric(df[m_mean], errors="coerce")
        if m.endswith("_r"):
            eff[m] = vals - 1.0
        else:
            eff[m] = vals - 0.0
        if m_p in df.columns:
            pvals[m] = pd.to_numeric(df[m_p], errors="coerce")

    # Order metrics for display: ratios first, then deltas
    metrics_sorted = sorted(metrics, key=lambda x: (0 if x.endswith("_r") else 1, x))
    eff = eff[metrics_sorted]
    pvals = pvals[metrics_sorted]

    eff_t = eff.T
    pvals_t = pvals.T

    # Heatmap with diverging palette centered at 0
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(
        figsize=(max(8.0, 1.0 + 0.55 * eff_t.shape[1]), 1.0 + 0.45 * eff_t.shape[0])
    )
    vmax = np.nanmax(np.abs(eff_t.values))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    sns.heatmap(
        eff_t,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label": "Effect size (ratio-1 or delta)"},
        linewidths=0.3,
        linecolor="white",
        square=False,
    )
    ax.set_xlabel("scenario")
    ax.set_ylabel("metric (normalized)")
    ax.set_title("Baseline-normalized effects with significance (p < 0.05)")

    # Overlay significance markers
    for i, m in enumerate(eff_t.index):
        for j, scen in enumerate(eff_t.columns):
            p = (
                pvals_t.at[m, scen]
                if (m in pvals_t.index and scen in pvals_t.columns)
                else np.nan
            )
            if np.isfinite(p) and p < 0.05:
                ax.plot(j + 0.5, i + 0.5, marker="o", markersize=4, color="black")

    out = save_to if save_to is not None else (analysis_root / "effects_heatmap.png")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:  # pragma: no cover - convenience
    import argparse

    ap = argparse.ArgumentParser(
        description="Plot significance heatmap from normalized insights"
    )
    ap.add_argument("analysis_root", type=str, help="Root with *_metrics")
    ap.add_argument("--save", type=str, default="", help="Output path")
    args = ap.parse_args()
    root = Path(args.analysis_root)
    out: Optional[Path] = None
    if args.save.strip():
        out = Path(args.save)
    res = plot_significance_heatmap(root, save_to=out)
    if res is not None:
        print(f"Saved effects heatmap → {res}")
    else:
        print("No insights to plot.")


if __name__ == "__main__":  # pragma: no cover
    main()
