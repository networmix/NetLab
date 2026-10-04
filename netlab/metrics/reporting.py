#!/usr/bin/env python3
"""Print summary tables and render figures from saved metric files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import netlab.metrics.summary as summary_mod


def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (ValueError, TypeError, KeyError, OSError):
        return float("nan")


def print_summary_from_csv(
    root: Path, plots: bool = False, quiet: bool = False
) -> None:
    out_root = root.parent / f"{root.name}_metrics"
    project_csv = out_root / "project.csv"
    norm_csv = out_root / "project_baseline_normalized.csv"
    if not project_csv.exists():
        raise FileNotFoundError(
            f"Missing {project_csv}; run 'netlab metrics {root}' first"
        )
    import pandas as _pd

    if not quiet:
        df_proj = _pd.read_csv(project_csv).set_index("scenario")
        summary_mod.print_pretty_table(df_proj, title="Consolidated project metrics")
        print("\n\n")
        if norm_csv.exists():
            df_norm = _pd.read_csv(norm_csv).set_index("scenario")
            summary_mod.print_pretty_table(
                df_norm, title="Baseline-normalized metrics (scenario / baseline)"
            )
            print("\n\n")
            summary_mod._print_normalized_insights(out_root)
        else:
            raise FileNotFoundError(f"Missing normalized table: {norm_csv}")

    if plots:
        from netlab.metrics.plot_bac_delta_vs_baseline import (
            plot_bac_delta_vs_baseline as _plot_bac_delta_vs_baseline,
        )
        from netlab.metrics.plot_cross_seed_bac import (
            plot_cross_seed_bac as _plot_cross_seed_bac,
        )
        from netlab.metrics.plot_cross_seed_iterops import (
            plot_cross_seed_iterops as _plot_cross_seed_iterops,
        )
        from netlab.metrics.plot_cross_seed_latency import (
            plot_cross_seed_latency as _plot_cross_seed_latency,
        )
        from netlab.metrics.plot_significance_heatmap import (
            plot_significance_heatmap as _plot_significance_heatmap,
        )

        fig_dir = out_root
        fig_dir.mkdir(parents=True, exist_ok=True)

        out_bac_png = fig_dir / "BAC.png"
        bac_path = _plot_cross_seed_bac(out_root, save_to=out_bac_png)
        if bac_path is None:
            print("(normalized BAC unavailable)")
        else:
            print(f"Saved BAC summary figure: {bac_path}")

        out_lat_png = fig_dir / "Latency_p99.png"
        lat_path = _plot_cross_seed_latency(out_root, metric="p99", save_to=out_lat_png)
        if lat_path is None:
            print("(latency unavailable)")
        else:
            print(f"Saved latency summary figure: {lat_path}")

        out_iterops_png = fig_dir / "IterationOps.png"
        iterops_path = _plot_cross_seed_iterops(out_root, save_to=out_iterops_png)
        if iterops_path is None:
            raise ValueError("No iterops data found to plot")
        print(f"Saved iteration-ops figure: {iterops_path}")

        out_delta_png = fig_dir / "BAC_delta_vs_baseline.png"
        delta_path = _plot_bac_delta_vs_baseline(
            out_root,
            grid_min=80.0,
            grid_max=100.0,
            legend_loc="upper left",
            save_to=out_delta_png,
        )
        if delta_path is not None:
            print(f"Saved BAC Δ-availability figure: {delta_path}")

        out_heatmap_png = fig_dir / "effects_heatmap.png"
        heatmap_path = _plot_significance_heatmap(out_root, save_to=out_heatmap_png)
        if heatmap_path is None:
            print("(no insights to plot for significance heatmap)")
        else:
            print(f"Saved effects heatmap: {heatmap_path}")

        import pandas as _pd

        df_proj = _pd.read_csv(project_csv).set_index("scenario")

        def _plot_dist_abs(column: str, title: str, ylabel: str, fname: str) -> None:
            if df_proj.empty or column not in df_proj.columns:
                return
            import seaborn as sns

            per_seed_csv = out_root / "project_per_seed_abs.csv"
            if not per_seed_csv.exists():
                return
            df_ps = _pd.read_csv(per_seed_csv)
            if column not in df_proj.columns:
                return
            plt.figure(figsize=(8.5, 5.2))
            data = (
                df_proj[[column]]
                .copy()
                .reset_index()
                .rename(columns={"index": "scenario"})
            )
            data = data.sort_values(by=column, ascending=False)
            order = data["scenario"].tolist()
            col_map = {
                "bac_auc": "auc_norm",
                "bw_p99": "bw_p99_pct",
                "lat_fail_p99": "lat_fail_p99",
                "USD_per_Gbit_offered": "USD_per_Gbit_offered",
                "USD_per_Gbit_p999": "USD_per_Gbit_p999",
                "Watt_per_Gbit_offered": "Watt_per_Gbit_offered",
                "Watt_per_Gbit_p999": "Watt_per_Gbit_p999",
                "capex_total": "capex_total",
                "node_count": "node_count",
                "link_count": "link_count",
            }
            ps_col = col_map.get(column, column)
            if ps_col in df_ps.columns:
                sns.stripplot(
                    data=df_ps,
                    x="scenario",
                    y=ps_col,
                    order=order,
                    jitter=0.25,
                    alpha=0.35,
                    color="gray",
                )
            sns.pointplot(
                data=data,
                x="scenario",
                y=column,
                order=order,
                linestyle="none",
                color="C0",
                errorbar=None,
            )
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel("scenario")
            plt.grid(True, linestyle=":", linewidth=0.5, axis="y")
            plt.xticks(rotation=20, ha="right")
            outp = fig_dir / fname
            plt.tight_layout()
            plt.savefig(outp)
            plt.close()
            print(f"Saved project metric figure: {outp}")

        def _plot_dist_norm(column: str, title: str, ylabel: str, fname: str) -> None:
            base_df_local = (
                _pd.read_csv(norm_csv).set_index("scenario")
                if norm_csv.exists()
                else None
            )
            if (
                base_df_local is None
                or base_df_local.empty
                or column not in base_df_local.columns
            ):
                return
            import seaborn as sns

            per_seed_norm_csv = out_root / "project_baseline_normalized_per_seed.csv"
            if not per_seed_norm_csv.exists():
                return
            df_psn = _pd.read_csv(per_seed_norm_csv)
            plt.figure(figsize=(8.5, 5.2))
            data = (
                base_df_local[[column]]
                .copy()
                .reset_index()
                .rename(columns={"index": "scenario"})
            )
            data = data.sort_values(by=column, ascending=False)
            order = data["scenario"].tolist()
            if column in df_psn.columns:
                sns.stripplot(
                    data=df_psn,
                    x="scenario",
                    y=column,
                    order=order,
                    jitter=0.25,
                    alpha=0.35,
                    color="gray",
                )
            sns.pointplot(
                data=data,
                x="scenario",
                y=column,
                order=order,
                linestyle="none",
                color="C0",
                errorbar=None,
            )
            ref = (
                1.0
                if column.endswith("_r")
                or column
                in (
                    "node_count_r",
                    "link_count_r",
                )
                else 0.0
            )
            plt.axhline(ref, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel("scenario")
            plt.grid(True, linestyle=":", linewidth=0.5, axis="y")
            plt.xticks(rotation=20, ha="right")
            outp = fig_dir / fname
            plt.tight_layout()
            plt.savefig(outp)
            plt.close()
            print(f"Saved project metric figure: {outp}")

        _plot_dist_abs("node_count", "Node count", "nodes", "abs_nodes.png")
        _plot_dist_abs("link_count", "Link count", "links", "abs_links.png")
        _plot_dist_abs(
            "bac_auc",
            title="BAC AUC (median across seeds)",
            ylabel="AUC (0..1)",
            fname="abs_AUC.png",
        )
        _plot_dist_abs(
            "bw_p90",
            title="Bandwidth at 90% (ratio to offered)",
            ylabel="ratio",
            fname="abs_BW_p90.png",
        )
        _plot_dist_abs(
            "bw_p95",
            title="Bandwidth at 95% (ratio to offered)",
            ylabel="ratio",
            fname="abs_BW_p95.png",
        )
        _plot_dist_abs(
            "bw_p99",
            title="Bandwidth at 99% (ratio to offered)",
            ylabel="ratio",
            fname="abs_BW_p99.png",
        )
        _plot_dist_abs(
            "USD_per_Gbit_offered",
            title="Cost per Gbps (offered)",
            ylabel="USD/Gbps",
            fname="abs_USD_per_Gbit_offered.png",
        )
        _plot_dist_abs(
            "USD_per_Gbit_p999",
            title="Cost per Gbps at p99.9",
            ylabel="USD/Gbps",
            fname="abs_USD_per_Gbit_p999.png",
        )
        _plot_dist_abs(
            "Watt_per_Gbit_offered",
            title="Power per Gbps (offered)",
            ylabel="W/Gbps",
            fname="abs_Watt_per_Gbit_offered.png",
        )
        _plot_dist_abs(
            "Watt_per_Gbit_p999",
            title="Power per Gbps at p99.9",
            ylabel="W/Gbps",
            fname="abs_Watt_per_Gbit_p999.png",
        )
        _plot_dist_abs(
            "lat_fail_p99",
            title="Latency p99 under failures (median across seeds)",
            ylabel="stretch (×)",
            fname="abs_Latency_fail_p99.png",
        )
        _plot_dist_abs(
            "capex_total",
            title="Total CapEx",
            ylabel="USD",
            fname="abs_CapEx.png",
        )

        _plot_dist_norm(
            "node_count_r",
            "Nodes (relative to baseline)",
            "ratio",
            "norm_nodes.png",
        )
        _plot_dist_norm(
            "link_count_r",
            "Links (relative to baseline)",
            "ratio",
            "norm_links.png",
        )
        _plot_dist_norm("auc_norm_r", "BAC AUC (relative)", "ratio", "norm_AUC.png")
        _plot_dist_norm("bw_p90_pct_r", "BW@90% (relative)", "ratio", "norm_BW_p90.png")
        _plot_dist_norm("bw_p95_pct_r", "BW@95% (relative)", "ratio", "norm_BW_p95.png")
        _plot_dist_norm("bw_p99_pct_r", "BW@99% (relative)", "ratio", "norm_BW_p99.png")
        _plot_dist_norm(
            "USD_per_Gbit_offered_r",
            "Cost per Gbps (offered, relative)",
            "ratio",
            "norm_USD_per_Gbit_offered.png",
        )
        _plot_dist_norm(
            "USD_per_Gbit_p999_r",
            "Cost per Gbps p99.9 (relative)",
            "ratio",
            "norm_USD_per_Gbit_p999.png",
        )
        _plot_dist_norm(
            "Watt_per_Gbit_offered_r",
            "Power per Gbps (offered, relative)",
            "ratio",
            "norm_Watt_per_Gbit_offered.png",
        )
        _plot_dist_norm(
            "Watt_per_Gbit_p999_r",
            "Power per Gbps p99.9 (relative)",
            "ratio",
            "norm_Watt_per_Gbit_p999.png",
        )
        _plot_dist_norm(
            "lat_fail_p99_r",
            "Latency p99 under failures (relative)",
            "ratio",
            "norm_Latency_fail_p99.png",
        )
