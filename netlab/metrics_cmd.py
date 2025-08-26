#!/usr/bin/env python3
"""
netlab.metrics_cmd — Orchestrator for metric computation and reporting.

This module ports the analysis workflow from the standalone script into a library
function callable from the netlab CLI.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import platform
import re
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics.summary as summary_mod
from metrics.aggregate import (
    summarize_across_seeds,
    write_csv_atomic,
    write_json_atomic,
)
from metrics.bac import BacResult, compute_bac, plot_bac
from metrics.costpower import (
    CostPowerResult,
    compute_cost_power,
    plot_cost_power,
)
from metrics.iterops import IterOpsResult, compute_iter_ops
from metrics.latency import LatencyResult, compute_latency_stretch, plot_latency
from metrics.matrixdump import compute_pair_matrices
from metrics.msd import AlphaResult, compute_alpha_star
from metrics.sps import SpsResult, compute_sps


# Local helpers
def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


# ---- Global plotting defaults ----
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (8.0, 5.0)
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


# --------- discovery & grouping helpers ---------

SEED_STEM_RE = re.compile(r"^(?P<stem>.+)__seed(?P<seed>\d+)_scenario$")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_results_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.results.json"))


def parse_seeded_stem(stem: str) -> Tuple[str, Optional[int]]:
    m = SEED_STEM_RE.match(stem)
    if not m:
        return stem, None
    return m.group("stem"), int(m.group("seed"))


def group_by_scenario(files: List[Path]) -> Dict[str, Dict[int, Path]]:
    grouped: Dict[str, Dict[int, Path]] = {}
    for p in files:
        s = p.stem
        if s.endswith(".results"):
            s = s[:-8]
        scenario_stem, seed = parse_seeded_stem(s)
        if seed is None:
            data = load_json(p)
            seed_val = data.get("scenario", {}).get("seed")
            if seed_val is None:
                raise ValueError(f"Missing 'scenario.seed' in results file: {p}")
            try:
                seed = int(seed_val)
            except Exception as e:
                raise ValueError(
                    f"Non-integer 'scenario.seed' in results file {p}: {seed_val}"
                ) from e
        grouped.setdefault(scenario_stem, {})[seed] = p
    return grouped


# --------- main orchestration ---------


@dataclass
class ScenarioOutputs:
    alpha: Dict[int, AlphaResult] = field(default_factory=dict)
    bac_place: Dict[int, BacResult] = field(default_factory=dict)
    bac_maxflow: Dict[int, BacResult] = field(default_factory=dict)
    latency: Dict[int, LatencyResult] = field(default_factory=dict)
    costpower: Dict[int, CostPowerResult] = field(default_factory=dict)
    sps: Dict[int, SpsResult] = field(default_factory=dict)
    iterops: Dict[int, IterOpsResult] = field(default_factory=dict)


def analyze_one_seed(
    results: dict, out_dir: Path, do_plots: bool
) -> Tuple[
    AlphaResult,
    BacResult,
    Optional[BacResult],
    LatencyResult,
    CostPowerResult,
    IterOpsResult,
    Optional[SpsResult],
]:
    def _require_steps(res: dict, require_maxflow: bool) -> None:
        steps = res.get("steps", {})
        required = ["msd_baseline", "tm_placement"] + (
            ["node_to_node_capacity_matrix"] if require_maxflow else []
        )
        for step in required:
            if step not in steps:
                raise ValueError(f"Missing required step in results: {step}")

    def _validate_alpha_and_base_demands(res: dict) -> float:
        msd = res.get("steps", {}).get("msd_baseline", {}).get("data", {}) or {}
        alpha_star = msd.get("alpha_star", None)
        if alpha_star is None:
            raise ValueError("Missing alpha_star in msd_baseline.data")
        try:
            alpha_star = float(alpha_star)
        except Exception as e:
            raise ValueError(f"alpha_star is not a number: {alpha_star}") from e
        if not np.isfinite(alpha_star) or alpha_star <= 0:
            raise ValueError(
                f"alpha_star must be a positive finite number; got {alpha_star}"
            )

        base_demands = msd.get("base_demands", []) or []
        if not isinstance(base_demands, list) or not base_demands:
            raise ValueError("msd_baseline.data.base_demands missing or empty")

        offenders_zero: List[str] = []
        offenders_neg: List[str] = []
        offenders_nan: List[str] = []
        duplicates: List[str] = []
        seen = set()
        base_total = 0.0
        for rec in base_demands:
            src = str(rec.get("source_path", "")).strip()
            dst = str(rec.get("sink_path", "")).strip()
            if not src or not dst:
                raise ValueError("Empty source_path/sink_path in base_demands entry")
            key = (src, dst, rec.get("mode", None), rec.get("priority", None))
            if key in seen:
                duplicates.append(f"{src}→{dst}")
            seen.add(key)
            try:
                dem = float(rec.get("demand", float("nan")))
            except Exception:
                dem = float("nan")
            if not np.isfinite(dem):
                offenders_nan.append(f"{src}→{dst}")
                continue
            if dem < 0.0:
                offenders_neg.append(f"{src}→{dst}")
            elif dem == 0.0:
                offenders_zero.append(f"{src}→{dst}")
            else:
                base_total += dem

        if offenders_nan:
            sample = ", ".join(offenders_nan[:5])
            more = (
                "" if len(offenders_nan) <= 5 else f" (+{len(offenders_nan) - 5} more)"
            )
            raise ValueError(
                f"Invalid traffic matrix: non-numeric demand for {len(offenders_nan)} entr"
                f"{'y' if len(offenders_nan) == 1 else 'ies'}: {sample}{more}"
            )
        if offenders_neg:
            sample = ", ".join(offenders_neg[:5])
            more = (
                "" if len(offenders_neg) <= 5 else f" (+{len(offenders_neg) - 5} more)"
            )
            raise ValueError(
                f"Invalid traffic matrix: negative demand for {len(offenders_neg)} entr"
                f"{'y' if len(offenders_neg) == 1 else 'ies'}: {sample}{more}"
            )
        if offenders_zero:
            sample = ", ".join(offenders_zero[:5])
            more = (
                ""
                if len(offenders_zero) <= 5
                else f" (+{len(offenders_zero) - 5} more)"
            )
            raise ValueError(
                "Invalid traffic matrix: "
                f"{len(offenders_zero)} zero-demand entr{'y' if len(offenders_zero) == 1 else 'ies'} detected: "
                f"{sample}{more}. TM generation must not emit entries with zero demand."
            )
        if duplicates:
            sample = ", ".join(duplicates[:5])
            more = "" if len(duplicates) <= 5 else f" (+{len(duplicates) - 5} more)"
            raise ValueError(
                f"Invalid traffic matrix: duplicate entries detected for {len(duplicates)} pair"
                f"{' ' if len(duplicates) == 1 else 's'}: {sample}{more}"
            )
        return base_total * float(alpha_star)

    def _validate_tm_placement_baseline(
        res: dict, expected_total_at_alpha: Optional[float]
    ) -> None:
        tm_step = res.get("steps", {}).get("tm_placement", {}) or {}
        tm_meta = tm_step.get("metadata", {}) or {}
        if bool(tm_meta.get("baseline")) is not True:
            raise ValueError(
                "tm_placement.metadata.baseline must be true and baseline must be included"
            )

        tm = tm_step.get("data", {}) or {}
        fr = tm.get("flow_results", []) or []
        if not isinstance(fr, list) or not fr:
            raise ValueError("tm_placement.data.flow_results missing or empty")
        first = fr[0]
        if str(first.get("failure_id", "")) != "baseline":
            raise ValueError(
                "tm_placement baseline must be first (flow_results[0].failure_id == 'baseline')"
            )
        baseline = first

        flows = baseline.get("flows", []) or []
        if not isinstance(flows, list) or not flows:
            raise ValueError("tm_placement baseline has no flows")

        def _is_close(a: float, b: float) -> bool:
            if not (np.isfinite(a) and np.isfinite(b)):
                return False
            diff = abs(a - b)
            return diff <= max(1e-6, 1e-3 * max(abs(a), abs(b), 1.0))

        for rec in flows:
            s = rec.get("source", "")
            d = rec.get("destination", "")
            if not s or not d or s == d:
                raise ValueError(
                    "tm_placement baseline contains invalid flow endpoints"
                )
            try:
                dem = float(rec.get("demand", float("nan")))
                pla = float(rec.get("placed", float("nan")))
                drp = float(rec.get("dropped", float("nan")))
            except Exception as e:
                raise ValueError(
                    "tm_placement baseline has non-numeric demand/placed/dropped"
                ) from e
            if not (np.isfinite(dem) and np.isfinite(pla) and np.isfinite(drp)):
                raise ValueError("tm_placement baseline has NaN/Inf values")
            if dem <= 0 or pla < 0 or drp < 0:
                raise ValueError(
                    "tm_placement baseline has non-positive demand or negative placed/dropped"
                )
            if not _is_close(pla + drp, dem):
                raise ValueError(
                    "tm_placement baseline violates placed + dropped ≈ demand"
                )

            cdist = rec.get("cost_distribution", {}) or {}
            if cdist:
                vol_sum = 0.0
                for k, v in cdist.items():
                    try:
                        float(k)
                        vv = float(v)
                    except Exception as e:
                        raise ValueError(
                            "tm_placement baseline has non-numeric cost_distribution"
                        ) from e
                    if vv < 0:
                        raise ValueError(
                            "tm_placement baseline has negative volume in cost_distribution"
                        )
                    vol_sum += vv
                if not _is_close(vol_sum, pla):
                    raise ValueError(
                        "tm_placement baseline cost_distribution volume sum does not equal placed"
                    )

        total_dem = baseline.get("summary", {}).get("total_demand", None)
        if expected_total_at_alpha is not None and total_dem is not None:
            try:
                td = float(total_dem)
            except Exception:
                td = float("nan")
            if np.isfinite(td) and not _is_close(td, float(expected_total_at_alpha)):
                raise ValueError(
                    "tm_placement baseline total_demand does not match base_demands × alpha_star"
                )

    def _validate_maxflow_baseline(res: dict) -> None:
        mf_step = res.get("steps", {}).get("node_to_node_capacity_matrix", {}) or {}
        mf_meta = mf_step.get("metadata", {}) or {}
        if bool(mf_meta.get("baseline")) is not True:
            raise ValueError(
                "node_to_node_capacity_matrix.metadata.baseline must be true and baseline must be included"
            )
        mf = mf_step.get("data", {}) or {}
        fr = mf.get("flow_results", []) or []
        if not isinstance(fr, list) or not fr:
            raise ValueError(
                "node_to_node_capacity_matrix.data.flow_results missing or empty"
            )
        first = fr[0]
        if str(first.get("failure_id", "")) != "baseline":
            raise ValueError(
                "node_to_node_capacity_matrix baseline must be first (flow_results[0].failure_id == 'baseline')"
            )
        for it in fr:
            for rec in it.get("flows", []) or []:
                s = rec.get("source", "")
                d = rec.get("destination", "")
                if not s or not d or s == d:
                    continue
                try:
                    placed = float(rec.get("placed", 0.0))
                except Exception as e:
                    raise ValueError(
                        "maxflow results contain non-numeric placed value"
                    ) from e
                if not np.isfinite(placed) or placed < 0.0:
                    raise ValueError(
                        "maxflow results contain negative or non-finite placed value"
                    )

    require_maxflow = bool(
        os.environ.get("NGRAPH_ENABLE_MAXFLOW", "0").strip()
        not in ("", "0", "false", "False")
    )
    _require_steps(results, require_maxflow)
    expected_total_at_alpha = _validate_alpha_and_base_demands(results)
    _validate_tm_placement_baseline(results, expected_total_at_alpha)
    if require_maxflow:
        _validate_maxflow_baseline(results)

    alpha = compute_alpha_star(results)

    bac_place = compute_bac(results, step_name="tm_placement", mode="auto")
    bac_max = (
        compute_bac(results, step_name="node_to_node_capacity_matrix", mode="auto")
        if require_maxflow
        else None
    )

    latency = compute_latency_stretch(results)
    iterops = compute_iter_ops(results)

    sps_res: Optional[SpsResult] = None
    if require_maxflow:
        sps_res = compute_sps(results)

    offered_alpha_star = None
    if not np.isnan(alpha.base_total_demand) and np.isfinite(alpha.alpha_star):
        offered_alpha_star = float(alpha.base_total_demand * alpha.alpha_star)
    reliable_p999 = bac_place.bw_at_probability_abs.get(99.9, np.nan)
    costpower = compute_cost_power(
        results, offered_at_alpha1=offered_alpha_star, reliable_at_p999=reliable_p999
    )

    if do_plots:
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        plot_bac(
            bac_place,
            overlay=bac_max if bac_max is not None else None,
            save_to=out_dir / "bac.png",
        )
        plot_latency(latency, save_to=out_dir / "latency.png")
        plot_cost_power(costpower, save_to=out_dir / "costpower.png")

    return alpha, bac_place, bac_max, latency, costpower, iterops, sps_res


def run_metrics(
    root: Path,
    only: Optional[str] = None,
    no_plots: bool = False,
    enable_maxflow: bool = False,
) -> None:
    if enable_maxflow:
        os.environ["NGRAPH_ENABLE_MAXFLOW"] = "1"
    out_root = root.parent / f"{root.name}_metrics"
    only_set = set([s.strip() for s in only.split(",") if s.strip()]) if only else None
    do_plots = not bool(no_plots)

    files = find_results_files(root)
    if not files:
        raise FileNotFoundError(f"No *.results.json found under {root}")

    grouped = group_by_scenario(files)
    if only_set:
        grouped = {k: v for k, v in grouped.items() if k in only_set}

    require_maxflow = bool(
        os.environ.get("NGRAPH_ENABLE_MAXFLOW", "0").strip()
        not in ("", "0", "false", "False")
    )
    for scenario_stem, seed_map in grouped.items():
        print(f"\n=== Scenario: {scenario_stem} (seeds={sorted(seed_map)}) ===")
        scenario_out = ScenarioOutputs()

        for seed, path in sorted(seed_map.items()):
            results = load_json(path)
            seed_dir = out_root / scenario_stem / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            alpha, bac_p, bac_m, latency, cp, itops, sps_opt = analyze_one_seed(
                results, seed_dir, do_plots
            )
            scenario_out.alpha[seed] = alpha
            scenario_out.bac_place[seed] = bac_p
            if bac_m is not None:
                scenario_out.bac_maxflow[seed] = bac_m
            scenario_out.latency[seed] = latency
            scenario_out.costpower[seed] = cp
            scenario_out.iterops[seed] = itops

            write_csv_atomic(
                seed_dir / "bac_series.csv",
                bac_p.series.to_frame(name="delivered"),
            )
            write_json_atomic(seed_dir / "bac.json", bac_p.to_jsonable())
            write_json_atomic(seed_dir / "alpha.json", alpha.to_jsonable())
            write_json_atomic(seed_dir / "latency.json", latency.to_jsonable())
            write_json_atomic(seed_dir / "costpower.json", cp.to_jsonable())
            write_json_atomic(seed_dir / "iterops.json", itops.to_jsonable())
            if sps_opt is not None:
                write_json_atomic(seed_dir / "sps.json", sps_opt.to_jsonable())
            # Export per-seed per-iteration latency (long format) with baseline deltas/ratios
            try:
                per_it = getattr(latency, "per_iteration", None) or {}
                if per_it:
                    rows_pi: list[dict] = []
                    base_vals = getattr(latency, "baseline", {}) or {}
                    for metric_key, series in per_it.items():
                        if not isinstance(series, list):
                            continue
                        for idx, val in enumerate(series):
                            try:
                                v = float(val)
                            except Exception:
                                continue
                            if not np.isfinite(v):
                                continue
                            b_raw = base_vals.get(metric_key)
                            b_val = _safe_float(b_raw)
                            delta = v - b_val if (pd.notna(b_val)) else float("nan")
                            ratio = (
                                (v / b_val)
                                if (pd.notna(b_val) and float(b_val) != 0.0)
                                else float("nan")
                            )
                            drop = (b_val - v) if pd.notna(b_val) else float("nan")
                            rows_pi.append(
                                {
                                    "metric": str(metric_key),
                                    "iter": int(idx),
                                    "value": float(v),
                                    "base": float(b_val)
                                    if pd.notna(b_val)
                                    else float("nan"),
                                    "delta": float(delta),
                                    "drop": float(drop),
                                    "ratio": float(ratio),
                                }
                            )
                    if rows_pi:
                        df_pi = pd.DataFrame(rows_pi)[
                            [
                                "metric",
                                "iter",
                                "value",
                                "base",
                                "delta",
                                "drop",
                                "ratio",
                            ]
                        ]
                        write_csv_atomic(
                            seed_dir / "latency_per_iteration_long.csv", df_pi
                        )
            except Exception as e:
                logging.warning(
                    "Failed to write per-iteration latency CSV for %s: %s",
                    seed_dir,
                    e,
                )
            tm_abs, tm_norm, mf_abs, mf_norm = compute_pair_matrices(
                results, include_maxflow=require_maxflow
            )
            if not tm_abs.empty:
                write_csv_atomic(seed_dir / "pairs_tm_abs.csv", tm_abs)
            if not tm_norm.empty:
                write_csv_atomic(seed_dir / "pairs_tm_norm.csv", tm_norm)
            if mf_abs is not None and not mf_abs.empty:
                write_csv_atomic(seed_dir / "pairs_mf_abs.csv", mf_abs)
            if mf_norm is not None and not mf_norm.empty:
                write_csv_atomic(seed_dir / "pairs_mf_norm.csv", mf_norm)

        scen_dir = out_root / scenario_stem
        scen_dir.mkdir(parents=True, exist_ok=True)

        alpha_summary = summarize_across_seeds(
            {k: v.alpha_star for k, v in scenario_out.alpha.items()}, label="alpha_star"
        )
        write_json_atomic(scen_dir / "alpha_summary.json", alpha_summary)

        series_by_seed = {
            seed: br.series for seed, br in scenario_out.bac_place.items()
        }
        bac_summary = summarize_across_seeds(series_by_seed, label="bac_delivered")
        bac_tail = {
            "p50": float(
                np.nanmedian(
                    [
                        v.quantiles_pct.get(0.50, np.nan)
                        for v in scenario_out.bac_place.values()
                    ]
                )
            ),
            "p90": float(
                np.nanmedian(
                    [
                        v.quantiles_pct.get(0.90, np.nan)
                        for v in scenario_out.bac_place.values()
                    ]
                )
            ),
            "p99": float(
                np.nanmedian(
                    [
                        v.quantiles_pct.get(0.99, np.nan)
                        for v in scenario_out.bac_place.values()
                    ]
                )
            ),
            "p999": float(
                np.nanmedian(
                    [
                        v.quantiles_pct.get(0.999, np.nan)
                        for v in scenario_out.bac_place.values()
                    ]
                )
            ),
            "p9999": float(
                np.nanmedian(
                    [
                        v.quantiles_pct.get(0.9999, np.nan)
                        for v in scenario_out.bac_place.values()
                    ]
                )
            ),
            "auc_norm": float(
                np.nanmedian(
                    [v.auc_normalized for v in scenario_out.bac_place.values()]
                )
            ),
        }

        def _bw_med(pct: float, _scenario_out: ScenarioOutputs = scenario_out) -> float:
            vals = []
            for v in _scenario_out.bac_place.values():
                raw = v.bw_at_probability_pct.get(pct)
                vals.append(_safe_float(raw))
            return float(np.nanmedian(vals)) if vals else float("nan")

        bac_tail.update(
            {
                "bw_p90_pct": _bw_med(90.0),
                "bw_p95_pct": _bw_med(95.0),
                "bw_p99_pct": _bw_med(99.0),
                "bw_p999_pct": _bw_med(99.9),
                "bw_p9999_pct": _bw_med(99.99),
            }
        )
        write_json_atomic(
            scen_dir / "bac_summary.json", {"per_seed": bac_summary, "tail": bac_tail}
        )

        pooled_samples: list[float] = []
        per_seed_samples: dict[int, list[float]] = {}
        for seed, br in scenario_out.bac_place.items():
            offered = float(br.offered)
            s = np.asarray(br.series.astype(float).values, dtype=float)
            if np.isfinite(offered) and offered > 0.0 and s.size > 0:
                norm = np.minimum(s / offered, 1.0) * 100.0
                vals = [float(x) for x in norm if np.isfinite(x)]
                if vals:
                    pooled_samples.extend(vals)
                    per_seed_samples[seed] = vals

        pooled_tail = {}
        grid = np.linspace(0.0, 100.0, 401)
        pooled_grid_x = []
        pooled_grid_a = []
        iqr_q25 = []
        iqr_q75 = []
        if pooled_samples:
            xs = np.sort(np.asarray(pooled_samples, dtype=float))
            cdf = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
            avail = 1.0 - cdf
            pooled_grid_x = xs.tolist()
            pooled_grid_a = avail.tolist()
            for p in (90.0, 95.0, 99.0, 99.9, 99.99):
                q = max(0.0, 1.0 - (p / 100.0))
                thr = float(np.quantile(xs, q, method="lower"))
                pooled_tail[f"bw_p{str(p).rstrip('0').rstrip('.')}__pct"] = thr / 100.0
            pooled_tail["auc_norm"] = float(np.mean(xs / 100.0))

            if len(per_seed_samples) >= 3:
                mat = []
                for vals in per_seed_samples.values():
                    sv = np.sort(np.asarray(vals, dtype=float))
                    cdf_s = np.arange(1, sv.size + 1, dtype=float) / float(sv.size)
                    a_s = 1.0 - cdf_s
                    a_on_grid = np.interp(grid, sv, a_s, left=a_s[0], right=a_s[-1])
                    mat.append(a_on_grid)
                mat = np.asarray(mat, dtype=float)
                iqr_q25 = np.nanpercentile(mat, 25, axis=0).tolist()
                iqr_q75 = np.nanpercentile(mat, 75, axis=0).tolist()

        pooled_payload = {
            "pooled_tail": pooled_tail,
            "pooled_grid": {"x_pct": pooled_grid_x, "availability": pooled_grid_a},
        }
        if iqr_q25 and iqr_q75:
            pooled_payload["pooled_iqr"] = {
                "x_pct": grid.tolist(),
                "a_q25": iqr_q25,
                "a_q75": iqr_q75,
            }
        path = scen_dir / "bac_summary.json"
        cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        cur.update(pooled_payload)
        write_json_atomic(path, cur)

        rows = []
        for seed, s in scenario_out.latency.items():
            b = s.baseline or {}
            f = s.failures or {}
            d = s.derived or {}
            rows.append(
                {
                    "seed": seed,
                    "base_p50": _safe_float(b.get("p50")),
                    "fail_p99": _safe_float(f.get("p99")),
                    "TD99": _safe_float(d.get("TD99")),
                    "SLO_1_2_drop": _safe_float(d.get("SLO_1_2_drop")),
                    "best_path_share_drop": _safe_float(d.get("best_path_share_drop")),
                    "WES_delta": _safe_float(d.get("WES_delta")),
                }
            )
        if rows:
            lat_sum = pd.DataFrame(rows).set_index("seed").sort_index()
            write_csv_atomic(scen_dir / "latency_summary.csv", lat_sum)

        # Persist scenario-level pooled latency exceedance (p95 and p99)
        try:

            def _availability_curve(
                samples: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray]:
                xs = np.asarray(samples, dtype=float)
                xs = xs[np.isfinite(xs)]
                if xs.size == 0:
                    return np.array([], dtype=float), np.array([], dtype=float)
                xs_sorted = np.sort(xs)
                cdf = np.arange(1, xs_sorted.size + 1, dtype=float) / float(
                    xs_sorted.size
                )
                avail = 1.0 - cdf
                return xs_sorted, avail

            for metric in ("p95", "p99"):
                pooled: list[float] = []
                seed_curves: list[np.ndarray] = []
                grid = np.linspace(1.0, 5.0, 401)
                for _seed, s in sorted(scenario_out.latency.items()):
                    per_it = getattr(s, "per_iteration", None) or {}
                    series = per_it.get(metric)
                    if not isinstance(series, list) or not series:
                        continue
                    vals: list[float] = []
                    for v in series:
                        try:
                            vv = float(v)
                        except Exception:
                            continue
                        if np.isfinite(vv):
                            vals.append(vv)
                    if not vals:
                        continue
                    pooled.extend(vals)
                    xs, a = _availability_curve(np.asarray(vals, dtype=float))
                    if xs.size > 0 and a.size > 0:
                        agrid = np.interp(grid, xs, a, left=a[0], right=a[-1])
                        seed_curves.append(agrid)

                if not pooled:
                    continue

                x_sorted, a_sorted = _availability_curve(
                    np.asarray(pooled, dtype=float)
                )
                if x_sorted.size > 0 and a_sorted.size > 0:
                    df_exc = pd.DataFrame(
                        {
                            "x": x_sorted.astype(float),
                            "availability": a_sorted.astype(float),
                        }
                    )
                    write_csv_atomic(
                        scen_dir / f"latency_pooled_exceedance_{metric}.csv", df_exc
                    )

                if len(seed_curves) >= 3:
                    mat = np.vstack(seed_curves)
                    q25 = np.nanpercentile(mat, 25, axis=0)
                    q75 = np.nanpercentile(mat, 75, axis=0)
                    df_iqr = pd.DataFrame(
                        {
                            "x": grid.astype(float),
                            "a_q25": q25.astype(float),
                            "a_q75": q75.astype(float),
                        }
                    )
                    write_csv_atomic(
                        scen_dir / f"latency_pooled_iqr_{metric}.csv", df_iqr
                    )
        except Exception as e:
            logging.warning(
                "Failed to write pooled latency exceedance CSVs for %s: %s",
                scen_dir,
                e,
            )

        io_rows = []
        for seed, it in scenario_out.iterops.items():
            ser = it.flat_series()
            rec: Dict[str, float] = {"seed": float(seed)}
            for k, v in ser.items():
                rec[str(k)] = float(v) if pd.notna(v) else float("nan")
            io_rows.append(rec)
        if io_rows:
            io_df = pd.DataFrame(io_rows).set_index("seed").sort_index()
            write_csv_atomic(scen_dir / "iterops_summary.csv", io_df)

        ns_rows = []
        for seed, path in sorted(seed_map.items()):
            res2 = load_json(path)
            ns = res2.get("steps", {}).get("network_statistics", {}).get("data", {})
            if not ns:
                raise ValueError(f"Missing network_statistics.data in {path}")
            if ns.get("node_count") is None or ns.get("link_count") is None:
                raise ValueError(
                    f"Incomplete network_statistics (node/link counts) in {path}"
                )
            ns_rows.append(
                {
                    "seed": seed,
                    "node_count": int(ns.get("node_count")),
                    "link_count": int(ns.get("link_count")),
                }
            )
        ns_df = pd.DataFrame(ns_rows).set_index("seed").sort_index()
        write_csv_atomic(scen_dir / "network_stats_summary.csv", ns_df)

        provenance = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "scenarios_root": os.path.relpath(root, start=Path.cwd()),
            "output_root": os.path.relpath(out_root, start=Path.cwd()),
        }
        try:
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
            provenance["git_commit"] = commit
        except Exception as e:
            logging.warning("Failed to retrieve git commit: %s", e)
            provenance["git_commit_error"] = str(e)
        write_json_atomic(scen_dir / "provenance.json", provenance)

        cp_df = pd.DataFrame(
            {
                seed: scenario_out.costpower[seed].flat_series()
                for seed in sorted(scenario_out.costpower)
            }
        ).T
        write_csv_atomic(scen_dir / "costpower_summary.csv", cp_df)

        # Scenario-level (inter-seed) figures saved inside each scenario folder
        if do_plots:
            from metrics.plot_cross_seed_bac import (
                plot_cross_seed_bac as _plot_cross_seed_bac,
            )
            from metrics.plot_cross_seed_iterops import (
                plot_cross_seed_iterops as _plot_cross_seed_iterops,
            )
            from metrics.plot_cross_seed_latency import (
                plot_cross_seed_latency as _plot_cross_seed_latency,
            )

            # Cross-seed BAC for this scenario
            scen_bac_png = scen_dir / "BAC.png"
            res_bac = _plot_cross_seed_bac(
                out_root, only=[scenario_stem], save_to=scen_bac_png
            )
            if res_bac is None:
                raise ValueError(
                    f"No BAC data found to plot for scenario '{scenario_stem}'"
                )

            # Cross-seed latency availability (p99) for this scenario
            scen_lat_png = scen_dir / "Latency_p99.png"
            res_lat = _plot_cross_seed_latency(
                out_root, metric="p99", only=[scenario_stem], save_to=scen_lat_png
            )
            if res_lat is None:
                raise ValueError(
                    f"No latency data found to plot for scenario '{scenario_stem}'"
                )

            # Cross-seed iterops summary for this scenario (single-scenario bars)
            scen_iterops_png = scen_dir / "IterationOps.png"
            res_iter = _plot_cross_seed_iterops(
                out_root, only=[scenario_stem], save_to=scen_iterops_png
            )
            if res_iter is None:
                raise ValueError(
                    f"No iterops data found to plot for scenario '{scenario_stem}'"
                )

    # Build project-level tables, write CSVs, then print summary by reading those CSVs
    df = summary_mod.build_project_summary_table(out_root)
    if df.empty:
        print("(no scenarios summarized)")
        return
    project_csv = out_root / "project.csv"
    write_csv_atomic(project_csv, df.reset_index())
    base_df = summary_mod.build_baseline_normalized_table(out_root)
    if not base_df.empty:
        write_csv_atomic(
            (out_root / "project_baseline_normalized.csv"),
            base_df.reset_index(),
        )
        # Also persist baseline-normalized insights (mean, n, p per metric)
        summary_mod.write_normalized_insights_csv(out_root)
        # And per-seed normalized values for distribution plots
        summary_mod.write_normalized_per_seed_csv(out_root)
    # Per-seed absolute values for distribution plots
    summary_mod.write_project_per_seed_abs_csv(out_root)

    # Print from CSVs to standardize output regardless of invocation
    summary_txt = out_root / "summary.txt"
    buf = io.StringIO()
    with redirect_stdout(buf):
        import pandas as _pd

        df_proj = _pd.read_csv(project_csv).set_index("scenario")
        summary_mod.print_pretty_table(df_proj, title="Consolidated project metrics")
        print("\n\n")
        base_csv = out_root / "project_baseline_normalized.csv"
        if base_csv.exists():
            df_norm = _pd.read_csv(base_csv).set_index("scenario")
            summary_mod.print_pretty_table(
                df_norm,
                title="Baseline-normalized metrics (scenario / baseline)",
            )
            print("\n\n")
            # Baseline-normalized paired comparisons vs target (means with n and p)
            summary_mod._print_normalized_insights(out_root)
            # Ensure CSV is written for machine consumption
            summary_mod.write_normalized_insights_csv(out_root)
            print("\n\n")
    text = buf.getvalue()
    summary_txt.write_text(text, encoding="utf-8")
    print(f"Wrote text summary: {summary_txt}")
    print(text, end="")
    print(f"Wrote project CSV: {project_csv}")

    # Create and save comprehensive provenance information for metrics run
    metrics_provenance = _create_metrics_provenance(root, out_root, files, only)

    # Add scenarios and seeds analyzed
    for scenario_stem, seed_map in grouped.items():
        metrics_provenance["scenarios_analyzed"].append(scenario_stem)
        metrics_provenance["seeds_analyzed"][scenario_stem] = sorted(seed_map.keys())

    provenance_path = out_root / "provenance.json"
    write_json_atomic(provenance_path, metrics_provenance)
    print(f"📋 Metrics provenance saved to: {provenance_path}")


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
            # Baseline-normalized paired comparisons vs target (means with n and p)
            summary_mod._print_normalized_insights(out_root)
        else:
            raise FileNotFoundError(f"Missing normalized table: {norm_csv}")

    # Baseline-normalized t-tests are printed above; project-level A vs B tests are available via CLI 'test'.

    # Cross-seed figures (publishable) when plotting is enabled
    if plots:
        # Prefer cross-seed pooled BAC overlay for the summary figure
        from metrics.plot_bac_delta_vs_baseline import (
            plot_bac_delta_vs_baseline as _plot_bac_delta_vs_baseline,
        )
        from metrics.plot_cross_seed_bac import (
            plot_cross_seed_bac as _plot_cross_seed_bac,
        )
        from metrics.plot_cross_seed_iterops import (
            plot_cross_seed_iterops as _plot_cross_seed_iterops,
        )
        from metrics.plot_cross_seed_latency import (
            plot_cross_seed_latency as _plot_cross_seed_latency,
        )
        from metrics.plot_significance_heatmap import (
            plot_significance_heatmap as _plot_significance_heatmap,
        )

        fig_dir = out_root
        fig_dir.mkdir(parents=True, exist_ok=True)

        # BAC figure (PNG)
        out_bac_png = fig_dir / "BAC.png"
        bac_path = _plot_cross_seed_bac(out_root, save_to=out_bac_png)
        if bac_path is None:
            raise ValueError("No BAC data found to plot")
        print(f"Saved BAC summary figure: {bac_path}")

        # Latency availability figure (p99; PNG)
        out_lat_png = fig_dir / "Latency_p99.png"
        lat_path = _plot_cross_seed_latency(out_root, metric="p99", save_to=out_lat_png)
        if lat_path is None:
            raise ValueError("No latency data found to plot")
        print(f"Saved latency summary figure: {lat_path}")

        # Iteration operations figure (multi-panel)
        out_iterops_png = fig_dir / "IterationOps.png"
        iterops_path = _plot_cross_seed_iterops(out_root, save_to=out_iterops_png)
        if iterops_path is None:
            raise ValueError("No iterops data found to plot")
        print(f"Saved iteration-ops figure: {iterops_path}")

        # BAC Δ-availability vs baseline (80–100%) with top-left legend
        out_delta_png = fig_dir / "BAC_delta_vs_baseline.png"
        delta_path = _plot_bac_delta_vs_baseline(
            out_root,
            baseline="baseline_SingleRouter",
            grid_min=80.0,
            grid_max=100.0,
            legend_loc="upper left",
            save_to=out_delta_png,
        )
        if delta_path is None:
            raise ValueError("No data to plot for BAC Δ-availability")
        print(f"Saved BAC Δ-availability figure: {delta_path}")

        # Significance heatmap (normalized effects with p<0.05 markers)
        out_heatmap_png = fig_dir / "effects_heatmap.png"
        heatmap_path = _plot_significance_heatmap(out_root, save_to=out_heatmap_png)
        if heatmap_path is None:
            print("(no insights to plot for significance heatmap)")
        else:
            print(f"Saved effects heatmap: {heatmap_path}")

        # Project-level distribution plots for key scalar metrics
        # Load project-level tables from CSVs for uniformity
        import pandas as _pd

        df_proj = _pd.read_csv(project_csv).set_index("scenario")

        def _plot_dist_abs(column: str, title: str, ylabel: str, fname: str) -> None:
            if df_proj.empty or column not in df_proj.columns:
                return
            import seaborn as sns

            plt.figure(figsize=(8.5, 5.2))
            # Per-seed absolute CSV
            per_seed_csv = out_root / "project_per_seed_abs.csv"
            if not per_seed_csv.exists():
                return
            df_ps = _pd.read_csv(per_seed_csv)
            if column not in df_proj.columns:
                return
            # Determine ordered scenarios from aggregate medians
            data = (
                df_proj[[column]]
                .copy()
                .reset_index()
                .rename(columns={"index": "scenario"})
            )
            data = data.sort_values(by=column, ascending=False)
            order = data["scenario"].tolist()
            # Build jitter points from per-seed values if present
            # Map column names in per-seed CSV
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
            # Overlay median point dots in the same order
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

            plt.figure(figsize=(8.5, 5.2))
            per_seed_norm_csv = out_root / "project_baseline_normalized_per_seed.csv"
            if not per_seed_norm_csv.exists():
                return
            df_psn = _pd.read_csv(per_seed_norm_csv)
            # Determine ordered scenarios from normalized medians
            data = (
                base_df_local[[column]]
                .copy()
                .reset_index()
                .rename(columns={"index": "scenario"})
            )
            data = data.sort_values(by=column, ascending=False)
            order = data["scenario"].tolist()
            # Jitter per-seed normalized values
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
            # Overlay scenario median in the same order
            sns.pointplot(
                data=data,
                x="scenario",
                y=column,
                order=order,
                linestyle="none",
                color="C0",
                errorbar=None,
            )
            # Reference line
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

        # Absolute plots (prefixed with abs_)
        _plot_dist_abs("node_count", "Node count", "nodes", "abs_nodes.png")
        _plot_dist_abs("link_count", "Link count", "links", "abs_links.png")
        _plot_dist_abs(
            "bac_auc",
            title="BAC AUC (median across seeds)",
            ylabel="AUC (0..1)",
            fname="abs_AUC.png",
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

        # Normalized plots vs baseline (prefixed with norm_)
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


def _create_metrics_provenance(
    root: Path, out_root: Path, files: List[Path], only: Optional[str] = None
) -> Dict[str, Any]:
    """Create comprehensive provenance information for a metrics run."""
    cwd = Path.cwd()
    provenance: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "command": "metrics",
        "source_root": os.path.relpath(root, start=cwd),
        "output_root": os.path.relpath(out_root, start=cwd),
        "source_files": {},
        "scenarios_analyzed": [],
        "seeds_analyzed": {},
    }

    # Get git commit if available
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        provenance["git_commit"] = commit
    except Exception as e:
        logging.warning("Failed to retrieve git commit: %s", e)
        provenance["git_commit_error"] = str(e)

    # Add source file information with hashes
    for file_path in files:
        try:
            file_content = file_path.read_bytes()
            file_hash = hashlib.sha256(file_content).hexdigest()
            rel_path = os.path.relpath(file_path, start=cwd)
            provenance["source_files"][rel_path] = {
                "path": rel_path,
                "sha256": file_hash,
                "size_bytes": len(file_content),
            }
        except Exception as e:
            logging.warning("Failed to hash source file %s: %s", file_path, e)
            rel_path = os.path.relpath(file_path, start=cwd)
            provenance["source_files"][rel_path] = {
                "path": rel_path,
                "hash_error": str(e),
            }

    # Add analysis scope information
    if only:
        provenance["only_scenarios"] = [s.strip() for s in only.split(",") if s.strip()]

    return provenance
