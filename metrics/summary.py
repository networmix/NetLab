from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# SciPy is required for rigorous t-tests and confidence intervals
from scipy import stats as scipy_stats  # type: ignore

from .aggregate import write_csv_atomic


def _try_import_rich() -> Tuple[Any, Any]:
    from rich.console import Console  # type: ignore
    from rich.table import Table  # type: ignore

    return Console, Table


Console, RichTable = _try_import_rich()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if _is_number(value):
        v = float(value)
        # Integer-like numbers within tolerance: show as integer with thousands separators
        if math.isfinite(v):
            nearest = round(v)
            if abs(v - nearest) <= max(1e-9, 1e-9 * max(abs(v), 1.0)):
                return f"{int(nearest):,}"
        # General numbers: fixed digits with thousands separators
        return f"{v:,.{digits}f}"
    return str(value)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _median_ignore_nan(values: List[Any]) -> float:
    arr = np.array([_safe_float(v) for v in values], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmedian(arr))


def _collect_latency_medians(latency_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(latency_csv)
    # Columns: seed, p50, p90, p99, p999, p9999
    meds: Dict[str, float] = {}
    for col in ("p50", "p90", "p99", "p999", "p9999"):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            vals = np.asarray(series.values, dtype=float)
            meds[col] = float(np.nanmedian(vals))
    return meds


def build_project_summary_table(analysis_root: Path) -> pd.DataFrame:
    """
    Build a consolidated summary across all scenarios under analysis_root.

    Each scenario directory should contain:
      - alpha_summary.json
      - bac_summary.json with keys:
          {"per_seed": ..., "tail": {p50,p90,p99,p999,p9999,auc_norm,
                                       bw_p90_pct,bw_p95_pct,bw_p99_pct,bw_p999_pct,bw_p9999_pct}}
      - latency_summary.csv (per-seed rows with columns p50..p9999)
      - costpower_summary.csv (per-seed rows with capex/power and normalizations)
    """
    rows: List[Dict[str, Any]] = []
    if not analysis_root.exists():
        return pd.DataFrame()

    for scen_dir in sorted([p for p in analysis_root.iterdir() if p.is_dir()]):
        scenario = scen_dir.name
        if scenario.startswith("_"):
            continue

        alpha_med = float("nan")
        bw_p90 = float("nan")
        bw_p95 = float("nan")
        bw_p99 = float("nan")
        bw_p999 = float("nan")
        auc_norm = float("nan")
        lat_base_p50 = float("nan")
        lat_fail_p99 = float("nan")
        lat_TD99 = float("nan")
        lat_SLO_1_2_drop = float("nan")
        lat_best_path_drop = float("nan")
        lat_WES_delta = float("nan")
        usd_per_gbit_p999 = float("nan")
        watt_per_gbit_p999 = float("nan")
        usd_per_gbit_offered = float("nan")
        watt_per_gbit_offered = float("nan")
        capex_total = float("nan")
        node_count = float("nan")
        link_count = float("nan")
        seeds_count = 0
        # Iteration-ops medians across seeds (per-iteration averages)
        spf_per_iter = float("nan")
        flows_created_per_iter = float("nan")
        reopt_calls_per_iter = float("nan")
        # tm_placement timing (seconds): total and per-iteration median across seeds
        tm_duration_total_sec = float("nan")
        tm_duration_per_iter_sec = float("nan")

        # alpha*
        ap = scen_dir / "alpha_summary.json"
        if ap.exists():
            import json

            a2 = json.loads(ap.read_text(encoding="utf-8"))
            alpha_med = _safe_float(a2.get("median"))

        # BAC cross-seed medians (keep AUC and BW@p)
        bp = scen_dir / "bac_summary.json"
        if bp.exists():
            import json

            b = json.loads(bp.read_text(encoding="utf-8"))
            tail = b.get("tail", {}) or {}
            auc_norm = _safe_float(tail.get("auc_norm"))
            # bandwidth at probability (ratio to baseline/offered)
            bw_p90 = (
                _safe_float(tail.get("bw_p90_pct"))
                if "bw_p90_pct" in tail
                else float("nan")
            )
            bw_p95 = (
                _safe_float(tail.get("bw_p95_pct"))
                if "bw_p95_pct" in tail
                else float("nan")
            )
            bw_p99 = (
                _safe_float(tail.get("bw_p99_pct"))
                if "bw_p99_pct" in tail
                else float("nan")
            )
            bw_p999 = (
                _safe_float(tail.get("bw_p999_pct"))
                if "bw_p999_pct" in tail
                else float("nan")
            )

        # Latency cross-seed medians
        lp = scen_dir / "latency_summary.csv"
        if lp.exists():
            df_lat = pd.read_csv(lp)

            def _med(col: str, _df_lat: pd.DataFrame = df_lat) -> float:
                if col in _df_lat.columns:
                    series = pd.to_numeric(_df_lat[col], errors="coerce")
                    vals = np.asarray(series.values, dtype=float)
                    return float(np.nanmedian(vals))
                return float("nan")

            lat_base_p50 = _med("base_p50")
            lat_fail_p99 = _med("fail_p99")
            lat_TD99 = _med("TD99")
            lat_SLO_1_2_drop = _med("SLO_1_2_drop")
            lat_best_path_drop = _med("best_path_share_drop")
            lat_WES_delta = _med("WES_delta")

        # Network stats (node/link counts)
        nsp = scen_dir / "network_stats_summary.csv"
        if nsp.exists():
            df_ns = pd.read_csv(nsp)
            if "node_count" in df_ns.columns:
                node_count = float(
                    np.nanmedian(pd.to_numeric(df_ns["node_count"], errors="coerce"))
                )
            if "link_count" in df_ns.columns:
                link_count = float(
                    np.nanmedian(pd.to_numeric(df_ns["link_count"], errors="coerce"))
                )
            # seeds counted as number of rows if not set by costpower
            seeds_count = max(seeds_count, int(df_ns.shape[0]))

        # Cost/Power medians across seeds
        cpp = scen_dir / "costpower_summary.csv"
        # Iteration operations per-iteration medians across seeds
        iop = scen_dir / "iterops_summary.csv"
        if iop.exists():
            df_io = pd.read_csv(iop)

            # Per-iteration averages stored as *_per_iter
            def _med_col(name: str, _df_io: pd.DataFrame = df_io) -> float:
                if name not in _df_io.columns:
                    return float("nan")
                series = pd.to_numeric(_df_io[name], errors="coerce")
                vals = np.asarray(series.values, dtype=float)
                return float(np.nanmedian(vals))

            spf_per_iter = _med_col("spf_calls_total_per_iter")
            flows_created_per_iter = _med_col("flows_created_total_per_iter")
            reopt_calls_per_iter = _med_col("reopt_calls_total_per_iter")
            tm_duration_total_sec = _med_col("tm_duration_total_sec")
            tm_duration_per_iter_sec = _med_col("tm_duration_per_iter_sec")
            # seeds counted from rows (keep max with other sources)
            seeds_count = max(seeds_count, int(df_io.shape[0]))

        if cpp.exists():
            df_cp = pd.read_csv(cpp)
            # seeds counted from rows
            seeds_count = max(seeds_count, int(df_cp.shape[0]))
            for col, var in (
                ("USD_per_Gbit_p999", "usd_per_gbit_p999"),
                ("Watt_per_Gbit_p999", "watt_per_gbit_p999"),
                ("USD_per_Gbit_offered", "usd_per_gbit_offered"),
                ("Watt_per_Gbit_offered", "watt_per_gbit_offered"),
                ("capex_total", "capex_total"),
            ):
                if col in df_cp.columns:
                    series = pd.to_numeric(df_cp[col], errors="coerce")
                    vals = np.asarray(series.values, dtype=float)
                    val = float(np.nanmedian(vals))
                    if var == "usd_per_gbit_p999":
                        usd_per_gbit_p999 = val
                    elif var == "watt_per_gbit_p999":
                        watt_per_gbit_p999 = val
                    elif var == "usd_per_gbit_offered":
                        usd_per_gbit_offered = val
                    elif var == "watt_per_gbit_offered":
                        watt_per_gbit_offered = val
                    elif var == "capex_total":
                        capex_total = val

        # Require non-zero seeds
        if seeds_count == 0:
            raise ValueError(f"No seeds counted for scenario {scenario}")

        row = {
            "scenario": scenario,
            "seeds": seeds_count,
            "node_count": node_count,
            "link_count": link_count,
            "alpha_star": alpha_med,
            "bw_p90": bw_p90,
            "bw_p95": bw_p95,
            "bw_p99": bw_p99,
            "bw_p999": bw_p999,
            "bac_auc": auc_norm,
            "lat_base_p50": lat_base_p50,
            "lat_fail_p99": lat_fail_p99,
            "lat_TD99": lat_TD99,
            "lat_SLO_1_2_drop": lat_SLO_1_2_drop,
            "lat_best_path_drop": lat_best_path_drop,
            "lat_WES_delta": lat_WES_delta,
            # Iteration operations (median per-iteration across seeds)
            "spf_calls_per_iter": spf_per_iter,
            "flows_created_per_iter": flows_created_per_iter,
            "reopt_calls_per_iter": reopt_calls_per_iter,
            # tm_placement timing (seconds)
            "tm_duration_total_sec": tm_duration_total_sec,
            "tm_duration_per_iter_sec": tm_duration_per_iter_sec,
            "USD_per_Gbit_offered": usd_per_gbit_offered,
            "Watt_per_Gbit_offered": watt_per_gbit_offered,
            "USD_per_Gbit_p999": usd_per_gbit_p999,
            "Watt_per_Gbit_p999": watt_per_gbit_p999,
            "capex_total": capex_total,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("scenario").sort_index()
    # Preferred display order
    cols = [
        "seeds",
        "node_count",
        "link_count",
        "alpha_star",
        # Bandwidth-at-probability ratios (primary availability figures)
        "bw_p90",
        "bw_p95",
        "bw_p99",
        "bw_p999",
        "bac_auc",
        # Latency summary (baseline vs failures + derived)
        "lat_base_p50",
        "lat_fail_p99",
        "lat_TD99",
        "lat_SLO_1_2_drop",
        "lat_best_path_drop",
        "lat_WES_delta",
        # Iteration operation intensity (median per-iteration across seeds)
        "spf_calls_per_iter",
        "flows_created_per_iter",
        "reopt_calls_per_iter",
        # tm_placement timing
        "tm_duration_total_sec",
        "tm_duration_per_iter_sec",
        # Cost/Power at offered and reliable
        "USD_per_Gbit_offered",
        "Watt_per_Gbit_offered",
        "USD_per_Gbit_p999",
        "Watt_per_Gbit_p999",
        "capex_total",
    ]
    cols = [c for c in cols if c in df.columns] + [
        c for c in df.columns if c not in cols
    ]
    return df[cols]


def print_pretty_table(
    df: pd.DataFrame, title: Optional[str] = None, digits: int = 3
) -> None:
    if df is None or df.empty:
        return
    if Console and RichTable:
        console = Console()
        table = RichTable(
            title=title, show_lines=False, show_header=True, pad_edge=False
        )
        # Short display labels for readability
        label_map = {
            "node_count": "nodes",
            "link_count": "links",
            "node_count_r": "nodes r",
            "link_count_r": "links r",
            "alpha_star": "alpha*",
            "bw_p90": "BW@90%",
            "bw_p95": "BW@95%",
            "bw_p99": "BW@99%",
            "bw_p999": "BW@99.9%",
            "bac_auc": "BAC AUC",
            "lat_base_p50": "lat base p50",
            "lat_fail_p99": "lat fail p99",
            "lat_TD99": "TD99",
            "lat_SLO_1_2_drop": "SLO≤1.2 drop",
            "lat_best_path_drop": "best-path drop",
            "lat_WES_delta": "WES Δ",
            "spf_calls_per_iter": "SPF/iter",
            "flows_created_per_iter": "flows created/iter",
            "reopt_calls_per_iter": "reopt/iter",
            "USD_per_Gbit_offered": "USD/Gbps offered",
            "Watt_per_Gbit_offered": "W/Gbps offered",
            "USD_per_Gbit_p999": "USD/Gbps p99.9",
            "Watt_per_Gbit_p999": "W/Gbps p99.9",
            "capex_total": "CapEx (USD)",
        }
        index_label = str(df.index.name) if df.index.name is not None else "scenario"
        table.add_column(index_label)
        for col in df.columns:
            table.add_column(label_map.get(str(col), str(col)), justify="right")
        for idx, row in df.iterrows():
            table.add_row(str(idx), *[_fmt(v, digits) for v in row.tolist()])
        console.print(table)
        # Add a compact legend for directions
        console.print(
            "\n[dim]- Ratios: higher is better (BW@p, BAC AUC); lower is better (lat_fail_p99, cost/power).\n- Drops/deltas: closer to 0 is better (SLO drop, best-path drop, WES Δ).[/dim]"
        )
    else:
        # Fallback: manual fixed-width table without wrapping
        if title:
            print(title)
        # Short display labels for readability
        label_map = {
            "node_count": "nodes",
            "link_count": "links",
            "node_count_r": "nodes r",
            "link_count_r": "links r",
            "alpha_star": "alpha*",
            "bw_p90": "BW@90%",
            "bw_p95": "BW@95%",
            "bw_p99": "BW@99%",
            "bw_p999": "BW@99.9%",
            "bac_auc": "BAC AUC",
            "lat_base_p50": "lat base p50",
            "lat_fail_p99": "lat fail p99",
            "lat_TD99": "TD99",
            "lat_SLO_1_2_drop": "SLO≤1.2 drop",
            "lat_best_path_drop": "best-path drop",
            "lat_WES_delta": "WES Δ",
            "spf_calls_per_iter": "SPF/iter",
            "flows_created_per_iter": "flows created/iter",
            "reopt_calls_per_iter": "reopt/iter",
            "USD_per_Gbit_offered": "USD/Gbps offered",
            "Watt_per_Gbit_offered": "W/Gbps offered",
            "USD_per_Gbit_p999": "USD/Gbps p99.9",
            "Watt_per_Gbit_p999": "W/Gbps p99.9",
            "capex_total": "CapEx (USD)",
        }
        # Pre-format values
        formatted_rows = []
        for idx, row in df.iterrows():
            formatted_rows.append([str(idx)] + [_fmt(v, digits) for v in row.tolist()])
        index_label = str(df.index.name) if df.index.name is not None else "scenario"
        headers = [index_label] + [label_map.get(str(c), str(c)) for c in df.columns]
        # Compute column widths
        widths = [len(h) for h in headers]
        for r in formatted_rows:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(cell))
        # Print header
        header_line = "  ".join(
            h.ljust(widths[i]) if i == 0 else h.rjust(widths[i])
            for i, h in enumerate(headers)
        )
        print(header_line)
        # Print separator
        print("  ".join(("-" * widths[i]) for i in range(len(widths))))
        # Print rows
        for r in formatted_rows:
            line = "  ".join(
                r[i].ljust(widths[i]) if i == 0 else r[i].rjust(widths[i])
                for i in range(len(widths))
            )
            print(line)


def save_project_csv_incremental(df: pd.DataFrame, cwd: Optional[Path] = None) -> Path:
    """Save df to cwd as project_{n}.csv (n increments to avoid overwrite)."""
    out_dir = cwd or Path.cwd()
    n = 0
    while True:
        out_path = out_dir / f"project_{n}.csv"
        if not out_path.exists():
            write_csv_atomic(out_path, df.reset_index())
            return out_path
        n += 1


def summarize_and_print(
    analysis_root: Path, title: str = "Project summary", write_project_csv: bool = True
) -> Optional[Path]:
    df = build_project_summary_table(analysis_root)
    if df.empty:
        print("(no scenarios summarized)")
        return None
    print_pretty_table(df, title=title)
    # Also print baseline-normalized view for thesis-friendly comparisons
    base_df = build_baseline_normalized_table(analysis_root)
    if not base_df.empty:
        print_pretty_table(
            base_df, title="Baseline-normalized metrics (scenario / baseline)"
        )
        # Save CSV alongside the main project CSV
        write_csv_atomic(
            (analysis_root / "project_baseline_normalized.csv"),
            base_df.reset_index(),
        )
        # Normalized insights (all + filtered)
        _print_normalized_insights(analysis_root)
    # Normalized insights are preferred; project-level absolute insights suppressed for simplicity
    if write_project_csv:
        return save_project_csv_incremental(df)
    return None


# -------------------- Baseline-normalized insights only --------------------


def write_normalized_insights_csv(analysis_root: Path) -> Optional[Path]:
    """Compute baseline-normalized insights and write them to normalized_insights.csv.

    The output is a wide CSV with one row per scenario and columns of the form
    <metric_base_name>__mean, <metric_base_name>__n, <metric_base_name>__p.
    Returns the path if written, else None when no insights are available.
    """
    res = _build_normalized_insights(analysis_root)
    if not res:
        return None
    df = pd.DataFrame(res).set_index("scenario").sort_index()

    # Add Holm-adjusted p-values per metric base across scenarios (within-metric family)
    # For each column trio base__mean/base__n/base__p, compute base__p_adj
    def _holm_adjust_1d(p_series: pd.Series) -> pd.Series:
        # p_series indexed by scenario; returns adjusted p indexed the same
        pairs = sorted(
            [(idx, float(p)) for idx, p in p_series.items() if np.isfinite(float(p))],
            key=lambda x: x[1],
        )
        m = len(pairs)
        out: Dict[str, float] = {}
        prev = 0.0
        for i, (sc, p) in enumerate(pairs, start=1):
            adj = min(1.0, (m - i + 1) * p)
            adj = max(prev, adj)
            out[str(sc)] = float(adj)
            prev = 0.0 if math.isnan(adj) else adj
        # fill others with NaN
        return pd.Series({k: out.get(str(k), float("nan")) for k in p_series.index})

    p_adj_cols: Dict[str, pd.Series] = {}
    for c in sorted(df.columns):
        if c.endswith("__p"):
            base = c[:-3]
            p_adj_cols[f"{base}__p_adj"] = _holm_adjust_1d(
                pd.to_numeric(df[c], errors="coerce")
            )
    for name, series in p_adj_cols.items():
        df[name] = series
    out_path = analysis_root / "normalized_insights.csv"
    write_csv_atomic(out_path, df.reset_index())
    return out_path


def write_project_per_seed_abs_csv(analysis_root: Path) -> Optional[Path]:
    """Write per-seed absolute (non-baseline-normalized) metrics across all scenarios.

    Columns: scenario, seed, node_count, link_count, auc_norm, bw_p99_pct,
    lat_fail_p99, USD_per_Gbit_offered, Watt_per_Gbit_offered,
    USD_per_Gbit_p999, Watt_per_Gbit_p999, capex_total.
    """
    scenarios = [
        p
        for p in sorted(analysis_root.iterdir())
        if p.is_dir() and not p.name.startswith("_")
    ]
    if not scenarios:
        return None
    rows: List[Dict[str, Any]] = []
    for scen_dir in scenarios:
        scen = scen_dir.name
        # Network stats (per-seed)
        ns_csv = scen_dir / "network_stats_summary.csv"
        ns_map: Dict[int, Tuple[float, float]] = {}
        if ns_csv.exists():
            df_ns = pd.read_csv(ns_csv)
            for _, r in df_ns.iterrows():
                seed_val = r.get("seed")
                try:
                    s = int(seed_val) if seed_val is not None else None
                except Exception:
                    s = None
                if s is None:
                    continue
                ns_map[s] = (
                    float(r.get("node_count", float("nan"))),
                    float(r.get("link_count", float("nan"))),
                )
        # Per-seed JSONs
        for seed_dir in sorted([p for p in scen_dir.glob("seed*") if p.is_dir()]):
            try:
                seed = int(seed_dir.name.replace("seed", ""))
            except Exception:
                continue
            rec: Dict[str, Any] = {"scenario": scen, "seed": int(seed)}
            # Node/link
            if seed in ns_map:
                rec["node_count"], rec["link_count"] = ns_map[seed]
            # BAC
            try:
                import json as _json

                bj = _json.loads((seed_dir / "bac.json").read_text(encoding="utf-8"))
            except Exception:
                bj = {}
            if bj:
                auc = bj.get("auc_normalized")
                if isinstance(auc, (int, float)):
                    rec["auc_norm"] = float(auc)
                bwp = bj.get("bw_at_probability_pct", {}) or {}
                v = bwp.get("99.0") if "99.0" in bwp else bwp.get(99.0)
                if isinstance(v, (int, float)):
                    rec["bw_p99_pct"] = float(v)
            # Latency
            try:
                import json as _json

                lj = _json.loads(
                    (seed_dir / "latency.json").read_text(encoding="utf-8")
                )
            except Exception:
                lj = {}
            if lj:
                failures = lj.get("failures", {}) or {}
                v = failures.get("p99")
                if isinstance(v, (int, float)):
                    rec["lat_fail_p99"] = float(v)
            # Cost/Power
            try:
                import json as _json

                cpj = _json.loads(
                    (seed_dir / "costpower.json").read_text(encoding="utf-8")
                )
            except Exception:
                cpj = {}
            if cpj:
                for k in (
                    "USD_per_Gbit_offered",
                    "Watt_per_Gbit_offered",
                    "USD_per_Gbit_p999",
                    "Watt_per_Gbit_p999",
                    "capex_total",
                ):
                    v = cpj.get(k)
                    if isinstance(v, (int, float)):
                        rec[k] = float(v)
            rows.append(rec)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out_path = analysis_root / "project_per_seed_abs.csv"
    write_csv_atomic(out_path, df)
    return out_path


def write_normalized_per_seed_csv(analysis_root: Path) -> Optional[Path]:
    """Write per-seed baseline-normalized metrics across all scenarios (excluding baseline).

    Uses the same per-seed normalization as insights: ratios vs 1.0 and deltas vs 0.0.
    Columns: scenario, seed, then *_r and *_d metrics.
    """
    data = _collect_normalized_per_seed(analysis_root)
    if not data:
        return None
    rows: List[Dict[str, Any]] = []
    for scen, seed_map in data.items():
        for seed, metrics in seed_map.items():
            rec = {"scenario": scen, "seed": int(seed)}
            for k, v in metrics.items():
                rec[str(k)] = float(v)
            rows.append(rec)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out_path = analysis_root / "project_baseline_normalized_per_seed.csv"
    write_csv_atomic(out_path, df)
    return out_path


def _read_json_safely(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _collect_per_seed_metrics(scen_dir: Path) -> Dict[int, Dict[str, float]]:
    """
    For a scenario directory, collect per-seed scalar metrics for paired comparisons.
    Returns {seed: {metric_name: value, ...}}
    Metrics:
      - alpha_star
      - bac_p999: from seed*/bac.json quantiles_pct[0.999]
      - latency_p99: median across iterations from seed*/latency_tails.csv
      - USD_per_Gbit_p999, Watt_per_Gbit_p999, capex_total: from seed*/costpower.json
    """
    out: Dict[int, Dict[str, float]] = {}
    for seed_dir in sorted([p for p in scen_dir.glob("seed*") if p.is_dir()]):
        seed = int(seed_dir.name.replace("seed", ""))

        metrics: Dict[str, float] = {}

        # alpha*
        aj = _read_json_safely(seed_dir / "alpha.json")
        if aj:
            v = aj.get("alpha_star")
            if isinstance(v, (int, float)):
                metrics["alpha_star"] = float(v)

        # BAC p999 (normalized)
        bj = _read_json_safely(seed_dir / "bac.json")
        if bj:
            qp = bj.get("quantiles_pct", {}) or {}
            v = qp.get("0.999") if "0.999" in qp else qp.get(0.999)
            if isinstance(v, (int, float)):
                metrics["bac_p999"] = float(v)

        # Latency p99: median across iterations for this seed
        lt_csv = seed_dir / "latency_tails.csv"
        if lt_csv.exists():
            lt_df = pd.read_csv(lt_csv)
            if "p99" in lt_df.columns and not lt_df["p99"].empty:
                vals = np.asarray(lt_df["p99"].astype(float).values, dtype=float)
                metrics["latency_p99"] = float(np.nanmedian(vals))

        # Cost/Power per-seed
        cpj = _read_json_safely(seed_dir / "costpower.json")
        if cpj:
            for k in ("USD_per_Gbit_p999", "Watt_per_Gbit_p999", "capex_total"):
                v = cpj.get(k)
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)

        if metrics:
            out[seed] = metrics

    return out


def _collect_per_seed_full_metrics(scen_dir: Path) -> Dict[int, Dict[str, float]]:
    """
    Collect per-seed metrics needed for baseline-normalized comparisons.
    Returns {seed: {metric_name: value}}
    Metrics (per seed):
      - bw_p99_pct, bw_p999_pct, auc_norm: from seed*/bac.json
      - lat_fail_p99, lat_TD99, lat_SLO_1_2_drop, lat_best_path_drop, lat_WES_delta: from seed*/latency.json
      - USD_per_Gbit_offered, Watt_per_Gbit_offered, USD_per_Gbit_p999, Watt_per_Gbit_p999: from seed*/costpower.json
    """

    def _read_json(path: Path) -> Dict[str, Any]:
        try:
            import json

            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    out: Dict[int, Dict[str, float]] = {}
    for seed_dir in sorted([p for p in scen_dir.glob("seed*") if p.is_dir()]):
        try:
            seed = int(seed_dir.name.replace("seed", ""))
        except Exception:
            continue
        m: Dict[str, float] = {}
        # BAC
        bj = _read_json(seed_dir / "bac.json")
        if bj:
            auc = bj.get("auc_normalized")
            if isinstance(auc, (int, float)):
                m["auc_norm"] = float(auc)
            bwp = bj.get("bw_at_probability_pct", {}) or {}

            def _g(p: float, bwp: Dict[str, Any] = bwp) -> Optional[float]:
                v = bwp.get(str(p))
                if v is None:
                    from typing import cast

                    v = cast(Dict[Any, Any], bwp).get(p)
                return float(v) if isinstance(v, (int, float)) else None

            v90 = _g(90.0)
            if v90 is not None:
                m["bw_p90_pct"] = v90
            v95 = _g(95.0)
            if v95 is not None:
                m["bw_p95_pct"] = v95
            v99 = _g(99.0)
            if v99 is not None:
                m["bw_p99_pct"] = v99
            v999 = _g(99.9)
            if v999 is not None:
                m["bw_p999_pct"] = v999
        # Latency
        lj = _read_json(seed_dir / "latency.json")
        if lj:
            failures = lj.get("failures", {}) or {}
            derived = lj.get("derived", {}) or {}
            for k_src, k_dst in (("p99", "lat_fail_p99"),):
                v = failures.get(k_src)
                if isinstance(v, (int, float)):
                    m[k_dst] = float(v)
            for k_src, k_dst in (
                ("TD99", "lat_TD99"),
                ("SLO_1_2_drop", "lat_SLO_1_2_drop"),
                ("best_path_share_drop", "lat_best_path_drop"),
                ("WES_delta", "lat_WES_delta"),
            ):
                v = derived.get(k_src)
                if isinstance(v, (int, float)):
                    m[k_dst] = float(v)
        # PairCap omitted from normalized comparisons
        # Cost/Power
        cpj = _read_json(seed_dir / "costpower.json")
        if cpj:
            for k in (
                "USD_per_Gbit_offered",
                "Watt_per_Gbit_offered",
                "USD_per_Gbit_p999",
                "Watt_per_Gbit_p999",
            ):
                v = cpj.get(k)
                if isinstance(v, (int, float)):
                    m[k] = float(v)
        if m:
            out[seed] = m
    return out


def build_baseline_normalized_table(
    analysis_root: Path, baseline_scenario: Optional[str] = None
) -> pd.DataFrame:
    """
    Build a table of scenario metrics normalized to a chosen baseline scenario.
    Ratios are computed per seed on shared seeds, then aggregated as median across seeds.
    Ratio metrics (scenario / baseline):
      - bw_p99_pct, bw_p999_pct, auc_norm, lat_fail_p99,
        USD_per_Gbit_offered, Watt_per_Gbit_offered, USD_per_Gbit_p999, Watt_per_Gbit_p999
    Drop/delta metrics (scenario - baseline):
      - lat_SLO_1_2_drop, lat_best_path_drop, lat_WES_delta
    Keep as-is (no extra normalization):
      - lat_TD99
    """
    if not analysis_root.exists():
        raise FileNotFoundError(f"Analysis root not found: {analysis_root}")
    scenarios = [
        p
        for p in sorted(analysis_root.iterdir())
        if p.is_dir() and not p.name.startswith("_")
    ]
    if not scenarios:
        raise ValueError("No scenarios found under analysis root")
    # Choose baseline
    bas = baseline_scenario or os.environ.get("NGRAPH_BASELINE_SCENARIO", "")
    if bas:
        base_name = bas
    else:
        # Heuristic: pick name containing 'baseline' else first
        cand = [p.name for p in scenarios if "baseline" in p.name.lower()]
        base_name = cand[0] if cand else scenarios[0].name
    scen_to_seed_metrics = {
        p.name: _collect_per_seed_full_metrics(p) for p in scenarios
    }
    if base_name not in scen_to_seed_metrics:
        raise ValueError(f"Baseline scenario not found: {base_name}")
    base = scen_to_seed_metrics[base_name]
    rows: List[Dict[str, Any]] = []

    ratio_keys = [
        "bw_p90_pct",
        "bw_p95_pct",
        "bw_p99_pct",
        "bw_p999_pct",
        "auc_norm",
        "lat_fail_p99",
        "USD_per_Gbit_offered",
        "Watt_per_Gbit_offered",
        "USD_per_Gbit_p999",
        "Watt_per_Gbit_p999",
    ]
    delta_keys = ["lat_SLO_1_2_drop", "lat_best_path_drop", "lat_WES_delta"]
    passthrough_keys = ["lat_TD99"]

    for scen_name, seed_map in scen_to_seed_metrics.items():
        if scen_name == base_name:
            # Baseline row: ratios = 1, deltas = 0, passthrough as baseline medians
            row: Dict[str, Any] = {"scenario": scen_name, "baseline": base_name}
            for k in ratio_keys:
                row[f"{k}_r"] = 1.0
            for k in delta_keys:
                row[f"{k}_d"] = 0.0
            # Structural ratios also default to 1.0 for baseline
            row["node_count_r"] = 1.0
            row["link_count_r"] = 1.0
            # lat_TD99 baseline median
            vals = [
                v.get("lat_TD99")
                for v in base.values()
                if isinstance(v.get("lat_TD99"), (int, float))
            ]
            row["lat_TD99"] = (
                float(np.nanmedian(np.asarray(vals, dtype=float)))
                if vals
                else float("nan")
            )
            rows.append(row)
            continue
        # Common seeds
        common = sorted(set(seed_map.keys()) & set(base.keys()))
        if not common:
            continue
        ratios: Dict[str, List[float]] = {k: [] for k in ratio_keys}
        deltas: Dict[str, List[float]] = {k: [] for k in delta_keys}
        passthrough: Dict[str, List[float]] = {k: [] for k in passthrough_keys}
        for s in common:
            sm = seed_map.get(s, {})
            bm = base.get(s, {})
            for k in ratio_keys:
                a = sm.get(k)
                b = bm.get(k)
                a = float(a) if isinstance(a, (int, float)) else float("nan")
                b = float(b) if isinstance(b, (int, float)) else float("nan")
                r = (
                    (a / b)
                    if (np.isfinite(a) and np.isfinite(b) and b != 0.0)
                    else float("nan")
                )
                ratios[k].append(r)
            for k in delta_keys:
                a = sm.get(k)
                b = bm.get(k)
                a = float(a) if isinstance(a, (int, float)) else float("nan")
                b = float(b) if isinstance(b, (int, float)) else float("nan")
                d = (a - b) if (np.isfinite(a) and np.isfinite(b)) else float("nan")
                deltas[k].append(d)
            for k in passthrough_keys:
                v = sm.get(k)
                v = float(v) if isinstance(v, (int, float)) else float("nan")
                passthrough[k].append(v)

        row = {"scenario": scen_name, "baseline": base_name}

        # Structural ratios (node/link counts) — per-seed normalization, then median
        def _counts_per_seed(name: str) -> Dict[int, Tuple[float, float]]:
            out: Dict[int, Tuple[float, float]] = {}
            p = analysis_root / name / "network_stats_summary.csv"
            if not p.exists():
                return out
            dfc = pd.read_csv(p)
            if "seed" not in dfc.columns:
                return out
            for _, r in dfc.iterrows():
                seed_val = r.get("seed")
                s = int(seed_val) if seed_val is not None else None
                if s is None:
                    continue
                node_count_val = float(r.get("node_count", float("nan")))
                link_count_val = float(r.get("link_count", float("nan")))
                out[s] = (node_count_val, link_count_val)
            return out

        scen_counts = _counts_per_seed(scen_name)
        base_counts = _counts_per_seed(base_name)
        node_rs: List[float] = []
        link_rs: List[float] = []
        for s in sorted(set(scen_counts.keys()) & set(base_counts.keys())):
            sn, sl = scen_counts.get(s, (float("nan"), float("nan")))
            bn, bl = base_counts.get(s, (float("nan"), float("nan")))
            if np.isfinite(sn) and np.isfinite(bn) and bn > 0:
                node_rs.append(float(sn / bn))
            if np.isfinite(sl) and np.isfinite(bl) and bl > 0:
                link_rs.append(float(sl / bl))
        row["node_count_r"] = (
            float(np.nanmedian(np.asarray(node_rs, dtype=float)))
            if node_rs
            else float("nan")
        )
        row["link_count_r"] = (
            float(np.nanmedian(np.asarray(link_rs, dtype=float)))
            if link_rs
            else float("nan")
        )
        for k, series in ratios.items():
            arr = np.asarray(series, dtype=float)
            arr = arr[np.isfinite(arr)]
            row[f"{k}_r"] = float(np.nanmedian(arr)) if arr.size else float("nan")
        for k, series in deltas.items():
            arr = np.asarray(series, dtype=float)
            arr = arr[np.isfinite(arr)]
            row[f"{k}_d"] = float(np.nanmedian(arr)) if arr.size else float("nan")
        for k, series in passthrough.items():
            arr = np.asarray(series, dtype=float)
            arr = arr[np.isfinite(arr)]
            row[k] = float(np.nanmedian(arr)) if arr.size else float("nan")
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("scenario").sort_index()
    # Order columns
    col_order = [
        "baseline",
        "node_count_r",
        "link_count_r",
        # Ratios (≥1 better for bw/auc; ≤1 better for cost/power/lat_fail_p99)
        "bw_p90_pct_r",
        "bw_p95_pct_r",
        "bw_p99_pct_r",
        "bw_p999_pct_r",
        "auc_norm_r",
        "lat_fail_p99_r",
        "USD_per_Gbit_offered_r",
        "Watt_per_Gbit_offered_r",
        "USD_per_Gbit_p999_r",
        "Watt_per_Gbit_p999_r",
        # Drops/deltas (closer to 0 is better)
        "lat_SLO_1_2_drop_d",
        "lat_best_path_drop_d",
        "lat_WES_delta_d",
        # Passthrough
        "lat_TD99",
    ]
    cols = [c for c in col_order if c in df.columns] + [
        c for c in df.columns if c not in col_order
    ]
    return df[cols]


def _paired_t_with_ci(
    a: np.ndarray, b: np.ndarray, alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Paired t-test for a and b with 95% CI for the mean difference (a-b).
    Uses SciPy when available; falls back to normal approx if SciPy is unavailable.
    Returns dict with keys: n, t_stat, p, mean_diff, ci_low, ci_high.
    """
    assert a.shape == b.shape
    # Ensure arrays are 1-D and same shape
    if a.shape != b.shape:
        raise ValueError("paired_t requires arrays with the same shape")
    d = a.astype(float) - b.astype(float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return {
            "n": float(n),
            "t_stat": float("nan"),
            "p": float("nan"),
            "mean_diff": float(np.nanmean(d) if n > 0 else np.nan),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    mean_diff = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / math.sqrt(n) if n > 0 else float("nan")

    # Treat near-zero SE as degenerate to avoid precision-loss paths
    eps = 1e-12
    if not math.isfinite(se) or se <= eps:
        t_stat = float("inf") if abs(mean_diff) > eps else 0.0
        # p-value under t-dist if available, else normal
        if scipy_stats is not None and t_stat != 0.0:
            p = 0.0
            tcrit = (
                float(scipy_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
                if n > 1
                else float("nan")
            )
        elif scipy_stats is not None and t_stat == 0.0:
            p = 1.0
            tcrit = (
                float(scipy_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
                if n > 1
                else float("nan")
            )
        else:
            # Normal approx
            p = 0.0 if t_stat != 0.0 else 1.0
            tcrit = 1.96
        ci_low = float(mean_diff)
        ci_high = float(mean_diff)
        deterministic = True
    else:
        # Compute t and p manually to avoid SciPy's internal precision-loss warning
        t_stat = mean_diff / se
        if scipy_stats is not None:
            df = max(n - 1, 1)
            p = float(2.0 * scipy_stats.t.sf(abs(t_stat), df=df))
            tcrit = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, df=df))
        else:
            # Normal approximation as a fallback
            p = float(
                2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
            )
            tcrit = 1.96
        ci_low = float(mean_diff - tcrit * se)
        ci_high = float(mean_diff + tcrit * se)
        deterministic = False
    return {
        "n": float(n),
        "t_stat": float(t_stat),
        "p": float(p),
        "mean_diff": float(mean_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "deterministic": bool(deterministic),
    }


def _holm_adjust(
    p_values: List[Tuple[Tuple[str, str], float]],
) -> Dict[Tuple[str, str], float]:
    """
    Holm step-down adjustment for a list of (pair, pvalue).
    pair is a (scenario_a, scenario_b) tuple for identification.
    Returns mapping pair -> adjusted_p.
    """
    # Sort ascending by p
    sorted_items = sorted(
        p_values, key=lambda x: (float("inf") if math.isnan(x[1]) else x[1])
    )
    m = len(sorted_items)
    adjusted: Dict[Tuple[str, str], float] = {}
    prev = 0.0
    for i, (pair, p) in enumerate(sorted_items, start=1):
        if math.isnan(p):
            adj = float("nan")
        else:
            adj = min(1.0, (m - i + 1) * p)
            adj = max(prev, adj)
        adjusted[pair] = float(adj)
        prev = 0.0 if math.isnan(adj) else adj
    return adjusted


def _build_insights(analysis_root: Path) -> List[Dict[str, Any]]:
    """
    Compute pairwise paired t-tests across scenarios for selected metrics.
    Returns a list of insight records with keys:
      metric, scen_a, scen_b, n, mean_diff, ci_low, ci_high, p, p_adj, t_stat
    """
    # Gather per-seed metrics for all scenarios
    scenarios = [
        p
        for p in sorted(analysis_root.iterdir())
        if p.is_dir() and not p.name.startswith("_")
    ]
    per_scen = {p.name: _collect_per_seed_metrics(p) for p in scenarios}

    # Metrics to analyze
    metric_labels = {
        "alpha_star": "alpha_star",
        "bac_p999": "BAC p99.9 (delivered/offered)",
        "latency_p99": "Latency p99 (stretch)",
        "USD_per_Gbit_p999": "Cost per Gbps (USD) at p99.9",
        "Watt_per_Gbit_p999": "Power per Gbps (W) at p99.9",
        "capex_total": "Total CapEx (USD)",
    }

    scenario_names = sorted(per_scen.keys())
    pairs: List[Tuple[str, str]] = []
    for i in range(len(scenario_names)):
        for j in range(i + 1, len(scenario_names)):
            pairs.append((scenario_names[i], scenario_names[j]))

    insights: List[Dict[str, Any]] = []

    for metric in metric_labels.keys():
        # Collect raw p-values for Holm adjustment across all pairs for this metric
        pvals_t: List[Tuple[Tuple[str, str], float]] = []
        tmp_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for a, b in pairs:
            seeds_a = set(per_scen[a].keys())
            seeds_b = set(per_scen[b].keys())
            common = sorted(seeds_a & seeds_b)
            if len(common) < 3:
                continue
            va = []
            vb = []
            for s in common:
                ma = per_scen[a][s].get(metric, np.nan)
                mb = per_scen[b][s].get(metric, np.nan)
                if math.isfinite(_safe_float(ma)) and math.isfinite(_safe_float(mb)):
                    va.append(_safe_float(ma))
                    vb.append(_safe_float(mb))
            if len(va) < 3:
                continue
            arr_a = np.array(va, dtype=float)
            arr_b = np.array(vb, dtype=float)
            test = _paired_t_with_ci(arr_a, arr_b)
            test.update({"metric": metric, "scen_a": a, "scen_b": b})
            pvals_t.append(((a, b), test["p"]))

            tmp_results[(a, b)] = test

        if not pvals_t:
            continue

        # Holm adjust
        padj_t = _holm_adjust(pvals_t)
        for pair, test in tmp_results.items():
            test["p_adj"] = float(padj_t.get(pair, float("nan")))
            insights.append(test)

    # Keep only entries with valid stats
    return insights


def _collect_normalized_per_seed(
    analysis_root: Path, baseline_scenario: Optional[str] = None
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return {scenario: {seed: metric->value}} for baseline-normalized metrics per seed.
    Values are ratios for ratio metrics and deltas for drop metrics, computed per seed vs baseline.
    """
    scenarios = [
        p
        for p in sorted(analysis_root.iterdir())
        if p.is_dir() and not p.name.startswith("_")
    ]
    bas = baseline_scenario or os.environ.get("NGRAPH_BASELINE_SCENARIO", "")
    if bas:
        base_name = bas
    else:
        cand = [p.name for p in scenarios if "baseline" in p.name.lower()]
        base_name = cand[0] if cand else (scenarios[0].name if scenarios else "")
    if not base_name:
        return {}
    full = {p.name: _collect_per_seed_full_metrics(p) for p in scenarios}
    if base_name not in full:
        return {}
    base = full[base_name]
    ratio_keys = [
        "bw_p90_pct",
        "bw_p95_pct",
        "bw_p99_pct",
        "bw_p999_pct",
        "auc_norm",
        "lat_fail_p99",
        "USD_per_Gbit_offered",
        "Watt_per_Gbit_offered",
        "USD_per_Gbit_p999",
        "Watt_per_Gbit_p999",
    ]
    delta_keys = ["lat_SLO_1_2_drop", "lat_best_path_drop", "lat_WES_delta"]
    passthrough_keys = ["lat_TD99"]
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for scen, seed_map in full.items():
        if scen == base_name:
            continue
        common = set(seed_map.keys()) & set(base.keys())
        if not common:
            continue
        per_seed: Dict[int, Dict[str, float]] = {}
        for s in common:
            sm = seed_map.get(s, {})
            bm = base.get(s, {})
            rec: Dict[str, float] = {}
            for k in ratio_keys:
                a = (
                    float(sm.get(k, float("nan")))
                    if isinstance(sm.get(k), (int, float))
                    else float("nan")
                )
                b = (
                    float(bm.get(k, float("nan")))
                    if isinstance(bm.get(k), (int, float))
                    else float("nan")
                )
                rec[f"{k}_r"] = (
                    (a / b)
                    if (np.isfinite(a) and np.isfinite(b) and b != 0.0)
                    else float("nan")
                )
            for k in delta_keys:
                a = (
                    float(sm.get(k, float("nan")))
                    if isinstance(sm.get(k), (int, float))
                    else float("nan")
                )
                b = (
                    float(bm.get(k, float("nan")))
                    if isinstance(bm.get(k), (int, float))
                    else float("nan")
                )
                rec[f"{k}_d"] = (
                    (a - b) if (np.isfinite(a) and np.isfinite(b)) else float("nan")
                )
            for k in passthrough_keys:
                v = sm.get(k)
                rec[k] = float(v) if isinstance(v, (int, float)) else float("nan")
            per_seed[int(s)] = rec
        if per_seed:
            out[scen] = per_seed
    return out


def _build_normalized_insights(analysis_root: Path) -> List[Dict[str, Any]]:
    """Paired tests on baseline-normalized metrics per seed (scenario vs 1.0 for ratios; vs 0.0 for deltas)."""
    data = _collect_normalized_per_seed(analysis_root)
    if not data:
        return []
    # Define metric groups
    ratio_metrics = [
        "node_count_r",
        "link_count_r",
        "bw_p90_pct_r",
        "bw_p95_pct_r",
        "bw_p99_pct_r",
        "bw_p999_pct_r",
        "auc_norm_r",
        "lat_fail_p99_r",
        "USD_per_Gbit_offered_r",
        "Watt_per_Gbit_offered_r",
        "USD_per_Gbit_p999_r",
        "Watt_per_Gbit_p999_r",
    ]
    delta_metrics = [
        "lat_SLO_1_2_drop_d",
        "lat_best_path_drop_d",
        "lat_WES_delta_d",
    ]
    results: List[Dict[str, Any]] = []
    for scen, seed_map in data.items():
        rec: Dict[str, Any] = {"scenario": scen}
        # Ratios vs 1.0
        for m in ratio_metrics:
            vals = [v.get(m, float("nan")) for v in seed_map.values()]
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 3:
                # t-test vs 1.0
                t_res = _paired_t_with_ci(arr, np.ones_like(arr))
                rec[f"{m}__n"] = int(arr.size)
                rec[f"{m}__mean"] = float(np.mean(arr))
                rec[f"{m}__p"] = float(t_res.get("p", float("nan")))
        # Deltas vs 0.0
        for m in delta_metrics:
            vals = [v.get(m, float("nan")) for v in seed_map.values()]
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 3:
                # t-test vs 0.0
                t_res = _paired_t_with_ci(arr, np.zeros_like(arr))
                rec[f"{m}__n"] = int(arr.size)
                rec[f"{m}__mean"] = float(np.mean(arr))
                rec[f"{m}__p"] = float(t_res.get("p", float("nan")))
        results.append(rec)
    return results


def _print_normalized_insights(analysis_root: Path, alpha: float = 0.05) -> None:
    """Print baseline-normalized comparisons (per-seed means with n and p)."""
    res = _build_normalized_insights(analysis_root)
    if not res:
        print("\n(no baseline-normalized insights available)")
        return
    df = pd.DataFrame(res).set_index("scenario")

    # Compute Holm-adjusted p-values per metric base across scenarios
    def _holm_adjust_1d(p_series: pd.Series) -> pd.Series:
        pairs = sorted(
            [(idx, float(p)) for idx, p in p_series.items() if np.isfinite(float(p))],
            key=lambda x: x[1],
        )
        m = len(pairs)
        out: Dict[str, float] = {}
        prev = 0.0
        for i, (sc, p) in enumerate(pairs, start=1):
            adj = min(1.0, (m - i + 1) * p)
            adj = max(prev, adj)
            out[str(sc)] = float(adj)
            prev = 0.0 if math.isnan(adj) else adj
        return pd.Series({k: out.get(str(k), float("nan")) for k in p_series.index})

    # Reformat into a compact 2-column per-metric display (mean, n, adj_p)
    display_cols: List[str] = []
    header_map: Dict[str, str] = {}
    for c in sorted(df.columns):
        if c.endswith("__mean"):
            base = c[:-6]
            # n/p columns inferred from base
            header = base
            header_map[base] = header
            display_cols.append(base)
    out_df = pd.DataFrame(index=df.index)
    # Precompute adjusted p per base
    p_adj_map: Dict[str, pd.Series] = {}
    for base in display_cols:
        p_col = f"{base}__p"
        if p_col in df.columns:
            p_adj_map[base] = _holm_adjust_1d(pd.to_numeric(df[p_col], errors="coerce"))
    for base in display_cols:
        mean = df.get(f"{base}__mean")
        n = df.get(f"{base}__n")
        p = p_adj_map.get(base, df.get(f"{base}__p"))
        vals: List[str] = []
        for i in range(df.shape[0]):
            m = mean.iat[i] if mean is not None else float("nan")
            nval = int(n.iat[i]) if n is not None and pd.notna(n.iat[i]) else 0
            pval = p.iat[i] if p is not None else float("nan")
            if not math.isfinite(float(m)):
                vals.append("–")
            else:
                vals.append(f"{m:.3f} (n={nval}, adj_p={pval:.3f})")
        out_df[base] = vals
    print_pretty_table(
        out_df,
        title="All baseline-normalized comparisons vs target (mean, n, adj_p)",
    )


def _print_project_insights(analysis_root: Path, alpha: float = 0.05) -> None:
    insights = _build_insights(analysis_root)
    if not insights:
        print("\n(no project insights available)")
        return

    # Filter significant results by Holm-adjusted p < alpha
    sig = [
        r
        for r in insights
        if (math.isfinite(r.get("p_adj", float("nan"))) and r["p_adj"] < alpha)
    ]
    if not sig:
        print(
            "\nNo statistically significant paired differences at Holm-adjusted alpha = 0.05."
        )
        return

    # Group by metric for organized printing
    header = "Project insights (Holm-adjusted p < 0.05)"
    # Sort for stable output
    metric_order = {
        "alpha_star": 0,
        "bac_p999": 1,
        "latency_p99": 2,
        "USD_per_Gbit_p999": 4,
        "Watt_per_Gbit_p999": 5,
        "capex_total": 6,
    }

    def _metric_sort_key(x: Dict[str, Any]) -> Tuple[int, str, str]:
        return (
            metric_order.get(x.get("metric", "zzz"), 999),
            x.get("scen_a", ""),
            x.get("scen_b", ""),
        )

    ordered = list(sorted(sig, key=_metric_sort_key))
    if Console and RichTable:
        console = Console()
        table = RichTable(title=header)
        table.add_column("Metric")
        table.add_column("A")
        table.add_column("B")
        table.add_column("n", justify="right")
        table.add_column("Δ mean", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("t", justify="right")
        table.add_column("p(adj)", justify="right")
        table.add_column("det", justify="center")

        def _label(m: str) -> str:
            return {
                "alpha_star": "alpha*",
                "bac_p999": "BAC p99.9",
                "latency_p99": "Latency p99",
                "USD_per_Gbit_p999": "USD/Gbps p99.9",
                "Watt_per_Gbit_p999": "Watt/Gbps p99.9",
                "capex_total": "CapEx (USD)",
            }.get(m, m)

        for r in ordered:
            metric = r["metric"]
            scen_a = r["scen_a"]
            scen_b = r["scen_b"]
            n = int(r.get("n", 0))
            mean_diff = _fmt(r.get("mean_diff", float("nan")))
            ci_low = _fmt(r.get("ci_low", float("nan")))
            ci_high = _fmt(r.get("ci_high", float("nan")))
            t_stat = r.get("t_stat", float("nan"))
            p_adj = r.get("p_adj", float("nan"))
            det = "✓" if r.get("deterministic") else ""
            table.add_row(
                _label(metric),
                scen_a,
                scen_b,
                str(n),
                mean_diff,
                f"[{ci_low}, {ci_high}]",
                (f"{t_stat:.3f}" if math.isfinite(float(t_stat)) else "–"),
                (f"{p_adj:.4f}" if math.isfinite(float(p_adj)) else "–"),
                det,
            )
        console.print(table)
    else:
        # Plain text, neat tabular output without wrapping
        def _label(m: str) -> str:
            return {
                "alpha_star": "alpha*",
                "bac_p999": "BAC p99.9",
                "latency_p99": "Latency p99",
                "USD_per_Gbit_p999": "USD/Gbps p99.9",
                "Watt_per_Gbit_p999": "Watt/Gbps p99.9",
                "capex_total": "CapEx (USD)",
            }.get(m, m)

        headers = [
            "Metric",
            "A",
            "B",
            "n",
            "Δ mean",
            "95% CI",
            "t",
            "p(adj)",
            "det",
        ]
        rows = []
        for r in ordered:
            t_stat = r.get("t_stat", float("nan"))
            p_adj = r.get("p_adj", float("nan"))
            rows.append(
                [
                    _label(r["metric"]),
                    r["scen_a"],
                    r["scen_b"],
                    f"{int(r.get('n', 0))}",
                    _fmt(r.get("mean_diff", float("nan"))),
                    f"[{_fmt(r.get('ci_low', float('nan')))}, {_fmt(r.get('ci_high', float('nan')))}]",
                    (f"{t_stat:.3f}" if math.isfinite(float(t_stat)) else "–"),
                    (f"{p_adj:.4f}" if math.isfinite(float(p_adj)) else "–"),
                    ("✓" if r.get("deterministic") else ""),
                ]
            )

        # Compute column widths
        widths = [len(h) for h in headers]
        for r in rows:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(str(cell)))

        print(f"\n{header}")
        # Header
        header_line = "  ".join(
            headers[i].ljust(widths[i])
            if i in (0, 1, 2, 8)
            else headers[i].rjust(widths[i])
            for i in range(len(headers))
        )
        print(header_line)
        print("  ".join(("-" * widths[i]) for i in range(len(widths))))
        # Rows
        for r in rows:
            line = "  ".join(
                str(r[i]).ljust(widths[i])
                if i in (0, 1, 2, 8)
                else str(r[i]).rjust(widths[i])
                for i in range(len(widths))
            )
            print(line)
