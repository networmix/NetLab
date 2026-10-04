from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table as RichTable

from netlab.artifacts import write_csv_atomic

from .comparisons import compare_scenarios
from .paired import holm_series, paired_t
from .seed_data import collect_seed_metrics, select_baseline


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if _is_number(value):
        v = float(value)
        if math.isfinite(v):
            nearest = round(v)
            if abs(v - nearest) <= max(1e-9, 1e-9 * max(abs(v), 1.0)):
                return f"{int(nearest):,}"
        return f"{v:,.{digits}f}"
    return str(value)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (ValueError, TypeError, KeyError, OSError):
        return float("nan")


def _median_ignore_nan(values: List[Any]) -> float:
    arr = np.array([_safe_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmedian(arr))


def build_project_summary_table(analysis_root: Path) -> pd.DataFrame:
    """Build scenario medians from alpha, BAC, latency, cost, and network summaries."""
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
        iters_fail = float("nan")
        iters_total = float("nan")
        unique_patterns = float("nan")
        tm_duration_total_sec = float("nan")
        tm_duration_per_iter_sec = float("nan")

        ap = scen_dir / "alpha_summary.json"
        if ap.exists():
            import json

            a2 = json.loads(ap.read_text(encoding="utf-8"))
            alpha_med = _safe_float(a2.get("median"))

        bp = scen_dir / "bac_summary.json"
        if bp.exists():
            import json

            b = json.loads(bp.read_text(encoding="utf-8"))
            tail = b.get("tail", {}) or {}
            auc_norm = _safe_float(tail.get("auc_norm"))
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
            seeds_count = max(seeds_count, int(df_ns.shape[0]))

        cpp = scen_dir / "costpower_summary.csv"
        iop = scen_dir / "iterops_summary.csv"
        if iop.exists():
            df_io = pd.read_csv(iop)

            def _med_col(name: str, _df_io: pd.DataFrame = df_io) -> float:
                if name not in _df_io.columns:
                    return float("nan")
                series = pd.to_numeric(_df_io[name], errors="coerce")
                vals = np.asarray(series.values, dtype=float)
                return float(np.nanmedian(vals))

            iters_fail = _med_col("iters_fail")
            iters_total = _med_col("iters_total")
            unique_patterns = _med_col("unique_patterns")
            tm_duration_total_sec = _med_col("tm_duration_total_sec")
            tm_duration_per_iter_sec = _med_col("tm_duration_per_iter_sec")
            seeds_count = max(seeds_count, int(df_io.shape[0]))

        if cpp.exists():
            df_cp = pd.read_csv(cpp)
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
            "iters_fail": iters_fail,
            "iters_total": iters_total,
            "unique_patterns": unique_patterns,
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
    cols = [
        "seeds",
        "node_count",
        "link_count",
        "alpha_star",
        "bw_p90",
        "bw_p95",
        "bw_p99",
        "bw_p999",
        "bac_auc",
        "lat_base_p50",
        "lat_fail_p99",
        "lat_TD99",
        "lat_SLO_1_2_drop",
        "lat_best_path_drop",
        "lat_WES_delta",
        "iters_fail",
        "iters_total",
        "unique_patterns",
        "tm_duration_total_sec",
        "tm_duration_per_iter_sec",
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
    console = Console()
    table = RichTable(title=title, show_lines=False, show_header=True, pad_edge=False)
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
        "iters_fail": "failure iterations",
        "iters_total": "total iterations",
        "unique_patterns": "unique patterns",
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
    console.print(
        "\n[dim]- Ratios: higher is better (BW@p, BAC AUC); lower is better (lat_fail_p99, cost/power).\n- Drops/deltas: closer to 0 is better (SLO drop, best-path drop, WES Δ).[/dim]"
    )


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
    base_df = build_baseline_normalized_table(analysis_root)
    if not base_df.empty:
        print_pretty_table(
            base_df, title="Baseline-normalized metrics (scenario / baseline)"
        )
        write_csv_atomic(
            (analysis_root / "project_baseline_normalized.csv"),
            base_df.reset_index(),
        )
        _print_normalized_insights(analysis_root)
    if write_project_csv:
        return save_project_csv_incremental(df)
    return None


def write_normalized_insights_csv(analysis_root: Path) -> Optional[Path]:
    """Write normalized_insights.csv with one row per scenario.

    Each metric has ``__mean``, ``__n``, ``__p``, and Holm-adjusted ``__p_adj``
    columns. Return the path, or None when no comparisons are available.
    """
    res = _build_normalized_insights(analysis_root)
    if not res:
        return None
    df = pd.DataFrame(res).set_index("scenario").sort_index()

    p_adj_cols: Dict[str, pd.Series] = {}
    for c in sorted(df.columns):
        if c.endswith("__p"):
            base = c[:-3]
            p_adj_cols[f"{base}__p_adj"] = holm_series(
                pd.to_numeric(df[c], errors="coerce")
            )
    for name, series in p_adj_cols.items():
        df[name] = series
    out_path = analysis_root / "normalized_insights.csv"
    write_csv_atomic(out_path, df.reset_index())
    return out_path


def write_project_per_seed_abs_csv(analysis_root: Path) -> Optional[Path]:
    """Write absolute metrics and network counts, one row per scenario and seed."""
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
        ns_csv = scen_dir / "network_stats_summary.csv"
        ns_map: Dict[int, Tuple[float, float]] = {}
        if ns_csv.exists():
            df_ns = pd.read_csv(ns_csv)
            for _, r in df_ns.iterrows():
                seed_val = r.get("seed")
                try:
                    s = int(seed_val) if seed_val is not None else None
                except (ValueError, TypeError, KeyError, OSError):
                    s = None
                if s is None:
                    continue
                ns_map[s] = (
                    float(r.get("node_count", float("nan"))),
                    float(r.get("link_count", float("nan"))),
                )
        columns = (
            "auc_norm",
            "bw_p99_pct",
            "lat_fail_p99",
            "USD_per_Gbit_offered",
            "Watt_per_Gbit_offered",
            "USD_per_Gbit_p999",
            "Watt_per_Gbit_p999",
            "capex_total",
        )
        for seed, metrics in collect_seed_metrics(scen_dir).items():
            rec: Dict[str, Any] = {"scenario": scen, "seed": seed}
            if seed in ns_map:
                rec["node_count"], rec["link_count"] = ns_map[seed]
            rec.update({key: metrics[key] for key in columns if key in metrics})
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


def build_baseline_normalized_table(
    analysis_root: Path, baseline_scenario: Optional[str] = None
) -> pd.DataFrame:
    """Normalize on matching seeds, then take each scenario's median.

    Bandwidth, AUC, latency p99, and unit cost/power use scenario/baseline ratios.
    SLO drop, best-path drop, and WES delta use differences. Keep TD99 unchanged.
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
    base_name = select_baseline([p.name for p in scenarios], baseline_scenario)
    scen_to_seed_metrics = {p.name: collect_seed_metrics(p) for p in scenarios}
    if base_name not in scen_to_seed_metrics:
        raise ValueError(f"Baseline scenario not found: {base_name}")
    base = scen_to_seed_metrics[base_name]
    rows: List[Dict[str, Any]] = []

    ratio_keys = [
        "node_count",
        "link_count",
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

        row: Dict[str, Any] = {"scenario": scen_name, "baseline": base_name}

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
    col_order = [
        "baseline",
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
        "lat_SLO_1_2_drop_d",
        "lat_best_path_drop_d",
        "lat_WES_delta_d",
        "lat_TD99",
    ]
    cols = [c for c in col_order if c in df.columns] + [
        c for c in df.columns if c not in col_order
    ]
    return df[cols]


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
    if not scenarios:
        return {}
    base_name = select_baseline([p.name for p in scenarios], baseline_scenario)
    full = {p.name: collect_seed_metrics(p) for p in scenarios}
    if base_name not in full:
        return {}
    base = full[base_name]
    ratio_keys = [
        "node_count",
        "link_count",
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
        for m in ratio_metrics:
            vals = [v.get(m, float("nan")) for v in seed_map.values()]
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 3:
                t_res = paired_t(arr, np.ones_like(arr))
                rec[f"{m}__n"] = int(arr.size)
                rec[f"{m}__mean"] = float(np.mean(arr))
                rec[f"{m}__p"] = float(t_res.get("p", float("nan")))
        for m in delta_metrics:
            vals = [v.get(m, float("nan")) for v in seed_map.values()]
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 3:
                t_res = paired_t(arr, np.zeros_like(arr))
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

    display_cols: List[str] = []
    header_map: Dict[str, str] = {}
    for c in sorted(df.columns):
        if c.endswith("__mean"):
            base = c[:-6]
            header = base
            header_map[base] = header
            display_cols.append(base)
    out_df = pd.DataFrame(index=df.index)
    p_adj_map: Dict[str, pd.Series] = {}
    for base in display_cols:
        p_col = f"{base}__p"
        if p_col in df.columns:
            p_adj_map[base] = holm_series(pd.to_numeric(df[p_col], errors="coerce"))
    for base in display_cols:
        mean = df.get(f"{base}__mean")
        n = df.get(f"{base}__n")
        p = p_adj_map.get(base, df.get(f"{base}__p"))
        vals: List[str] = []
        for i in range(df.shape[0]):
            m_val = float(mean.iloc[i]) if mean is not None else float("nan")
            n_raw = n.iloc[i] if n is not None else None
            nval = int(n_raw) if n_raw is not None and pd.notna(n_raw) else 0
            pval = float(p.iloc[i]) if p is not None else float("nan")
            if not math.isfinite(m_val):
                vals.append("–")
            else:
                vals.append(f"{m_val:.3f} (n={nval}, adj_p={pval:.3f})")
        out_df[base] = vals
    print_pretty_table(
        out_df,
        title="All baseline-normalized comparisons vs target (mean, n, adj_p)",
    )


def _print_project_insights(analysis_root: Path, alpha: float = 0.05) -> None:
    insights = compare_scenarios(analysis_root, alpha=alpha)
    if not insights:
        print("\n(no project insights available)")
        return

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

    header = "Project insights (Holm-adjusted p < 0.05)"
    metric_order = {
        "alpha_star": 0,
        "bw_p999_pct": 1,
        "lat_fail_p99": 2,
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
            "bw_p999_pct": "BAC p99.9",
            "lat_fail_p99": "Latency p99",
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
