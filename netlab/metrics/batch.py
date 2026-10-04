#!/usr/bin/env python3
"""Aggregate selected simulations and publish a complete metrics report."""

from __future__ import annotations

import io
import json
import logging
import os
import platform
import re
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import netlab.metrics.summary as summary_mod
from netlab.artifacts import (
    package_versions,
    sha256_file,
    write_csv_atomic,
)
from netlab.metrics.aggregate import (
    summarize_across_seeds,
)
from netlab.metrics.bac import BacResult
from netlab.metrics.costpower import (
    CostPowerResult,
)
from netlab.metrics.iterops import IterOpsResult
from netlab.metrics.latency import LatencyResult
from netlab.metrics.matrixdump import compute_pair_matrices
from netlab.metrics.msd import AlphaResult
from netlab.metrics.sps import SpsResult

from .analysis import analyze_one_seed
from .common import write_metric_json as write_json_atomic
from .distributions import availability_curve, curve_on_grid, threshold_at_probability


def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (ValueError, TypeError, KeyError, OSError):
        return float("nan")


SEED_STEM_RE = re.compile(r"^(?P<stem>.+)__seed(?P<seed>-?\d+)_scenario$")


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
        data = load_json(p)
        recorded_seed = data.get("scenario", {}).get("seed")
        if recorded_seed is not None and (
            type(recorded_seed) is not int
            or (seed is not None and seed != recorded_seed)
        ):
            raise ValueError(
                f"Scenario seed disagrees with filename or is not an integer: {p}"
            )
        if seed is None:
            seed_val = data.get("scenario", {}).get("seed")
            if seed_val is None:
                raise ValueError(f"Missing 'scenario.seed' in results file: {p}")
            try:
                if type(seed_val) is not int:
                    raise ValueError("seed must be an integer")
                seed = seed_val
            except (ValueError, TypeError, KeyError, OSError) as e:
                raise ValueError(
                    f"Non-integer 'scenario.seed' in results file {p}: {seed_val}"
                ) from e
        scenario_files = grouped.setdefault(scenario_stem, {})
        if seed in scenario_files:
            raise ValueError(
                f"Duplicate scenario/seed {scenario_stem}/{seed}: {scenario_files[seed]} and {p}"
            )
        scenario_files[seed] = p
    return grouped


@dataclass
class ScenarioOutputs:
    alpha: Dict[int, AlphaResult] = field(default_factory=dict)
    bac_place: Dict[int, BacResult] = field(default_factory=dict)
    bac_maxflow: Dict[int, BacResult] = field(default_factory=dict)
    latency: Dict[int, LatencyResult] = field(default_factory=dict)
    costpower: Dict[int, CostPowerResult] = field(default_factory=dict)
    sps: Dict[int, SpsResult] = field(default_factory=dict)
    iterops: Dict[int, IterOpsResult] = field(default_factory=dict)


def run_metrics(
    root: Path,
    only: Optional[str] = None,
    no_plots: bool = False,
    enable_maxflow: bool = False,
) -> None:
    """Publish a complete report tree only after every selected input succeeds.

    Each invocation replaces the generated report, including with ``only``.
    A failed analysis leaves the previous report and its provenance intact.
    """
    root = root.resolve()
    out_root = root.parent / f"{root.name}_metrics"
    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}-metrics-", dir=root.parent
    ) as temp:
        staged = Path(temp) / out_root.name
        log = io.StringIO()
        with redirect_stdout(log):
            _run_metrics(root, staged, only, no_plots, enable_maxflow)
        # Provenance names the published location, not the temporary staging area.
        for path in staged.rglob("provenance.json"):
            payload = load_json(path)
            payload["output_root"] = os.path.relpath(out_root, start=Path.cwd())
            write_json_atomic(path, payload)
        backup = Path(temp) / "previous"
        if out_root.exists():
            out_root.rename(backup)
        try:
            staged.rename(out_root)
        except BaseException:
            if backup.exists():
                backup.rename(out_root)
            raise
        print(log.getvalue().replace(str(staged), str(out_root)), end="")


def _run_metrics(
    root: Path,
    out_root: Path,
    only: str | None,
    no_plots: bool,
    enable_maxflow: bool,
) -> None:
    only_set = set([s.strip() for s in only.split(",") if s.strip()]) if only else None
    do_plots = not bool(no_plots)

    files = find_results_files(root)
    if not files:
        raise FileNotFoundError(f"No *.results.json found under {root}")

    grouped = group_by_scenario(files)
    if only_set:
        grouped = {k: v for k, v in grouped.items() if k in only_set}
        if missing := only_set - set(grouped):
            raise ValueError(f"Requested scenarios not found: {sorted(missing)}")

    input_hashes = {
        p: sha256_file(p) for mapping in grouped.values() for p in mapping.values()
    }
    require_maxflow = enable_maxflow
    for scenario_stem, seed_map in grouped.items():
        print(f"\n=== Scenario: {scenario_stem} (seeds={sorted(seed_map)}) ===")
        scenario_out = ScenarioOutputs()

        for seed, path in sorted(seed_map.items()):
            results = load_json(path)
            seed_dir = out_root / scenario_stem / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            alpha, bac_p, bac_m, latency, cp, itops, sps_opt = analyze_one_seed(
                results, seed_dir, do_plots, enable_maxflow=enable_maxflow
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
            try:
                per_it = latency.per_iteration or {}
                if per_it:
                    rows_pi: list[dict] = []
                    base_vals = latency.baseline or {}
                    for metric_key, series in per_it.items():
                        if not isinstance(series, list):
                            continue
                        for idx, val in enumerate(series):
                            try:
                                v = float(val)
                            except (ValueError, TypeError, KeyError, OSError):
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
            except (ValueError, TypeError, KeyError, OSError) as e:
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
            {k: v.alpha_star for k, v in scenario_out.alpha.items()}
        )
        write_json_atomic(scen_dir / "alpha_summary.json", alpha_summary)

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
            scen_dir / "bac_summary.json",
            {"seed_count": len(scenario_out.bac_place), "tail": bac_tail},
        )

        pooled_samples: list[float] = []
        per_seed_samples: dict[int, list[float]] = {}
        for seed, br in scenario_out.bac_place.items():
            offered = float(br.offered)
            s = np.asarray(br.series.astype(float).values, dtype=float)
            if np.isfinite(offered) and offered > 0.0 and s.size > 0:
                norm = (s / offered) * 100.0
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
            samples = np.asarray(pooled_samples, dtype=float)
            xs, avail = availability_curve(samples)
            pooled_grid_x = xs.tolist()
            pooled_grid_a = avail.tolist()
            for p in (90.0, 95.0, 99.0, 99.9, 99.99):
                thr = threshold_at_probability(samples, p)
                pooled_tail[f"bw_p{str(p).rstrip('0').rstrip('.')}__pct"] = thr / 100.0
            pooled_tail["auc_norm"] = float(np.mean(np.minimum(samples / 100.0, 1.0)))

            if len(per_seed_samples) >= 3:
                mat = []
                for vals in per_seed_samples.values():
                    sv = np.sort(np.asarray(vals, dtype=float))
                    a_on_grid = curve_on_grid(*availability_curve(sv), grid)
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

        try:
            for metric in ("p95", "p99"):
                pooled: list[float] = []
                seed_curves: list[np.ndarray] = []
                grid = np.linspace(1.0, 5.0, 401)
                for _seed, s in sorted(scenario_out.latency.items()):
                    per_it = s.per_iteration or {}
                    series = per_it.get(metric)
                    if not isinstance(series, list) or not series:
                        continue
                    vals: list[float] = []
                    for v in series:
                        try:
                            vv = float(v)
                        except (ValueError, TypeError, KeyError, OSError):
                            continue
                        if np.isfinite(vv):
                            vals.append(vv)
                    if not vals:
                        continue
                    pooled.extend(vals)
                    xs, a = availability_curve(np.asarray(vals, dtype=float))
                    if xs.size > 0 and a.size > 0:
                        agrid = curve_on_grid(xs, a, grid)
                        seed_curves.append(agrid)

                if not pooled:
                    continue

                x_sorted, a_sorted = availability_curve(np.asarray(pooled, dtype=float))
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
        except (ValueError, TypeError, KeyError, OSError) as e:
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
            if any(
                type(ns[key]) is not int or ns[key] < 0
                for key in ("node_count", "link_count")
            ):
                raise ValueError(f"Network counts must be nonnegative integers: {path}")
            ns_rows.append(
                {
                    "seed": seed,
                    "node_count": int(ns.get("node_count")),
                    "link_count": int(ns.get("link_count")),
                }
            )
        ns_df = pd.DataFrame(ns_rows).set_index("seed").sort_index()
        write_csv_atomic(scen_dir / "network_stats_summary.csv", ns_df)

        provenance: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "scenarios_root": os.path.relpath(root, start=Path.cwd()),
            "output_root": os.path.relpath(out_root, start=Path.cwd()),
        }
        provenance["packages"] = _metric_package_versions()

        write_json_atomic(scen_dir / "provenance.json", provenance)

        cp_df = pd.DataFrame(
            {
                seed: scenario_out.costpower[seed].flat_series()
                for seed in sorted(scenario_out.costpower)
            }
        ).T
        write_csv_atomic(scen_dir / "costpower_summary.csv", cp_df)

        if do_plots:
            from netlab.metrics.plot_cross_seed_bac import (
                plot_cross_seed_bac as _plot_cross_seed_bac,
            )
            from netlab.metrics.plot_cross_seed_iterops import (
                plot_cross_seed_iterops as _plot_cross_seed_iterops,
            )
            from netlab.metrics.plot_cross_seed_latency import (
                plot_cross_seed_latency as _plot_cross_seed_latency,
            )

            scen_bac_png = scen_dir / "BAC.png"
            res_bac = _plot_cross_seed_bac(
                out_root, only=[scenario_stem], save_to=scen_bac_png
            )
            if res_bac is None:
                print(f"{scenario_stem}: normalized BAC unavailable (zero baseline)")

            scen_lat_png = scen_dir / "Latency_p99.png"
            res_lat = _plot_cross_seed_latency(
                out_root, metric="p99", only=[scenario_stem], save_to=scen_lat_png
            )
            if res_lat is None:
                print(f"{scenario_stem}: latency unavailable (no referenced delivery)")

            scen_iterops_png = scen_dir / "IterationOps.png"
            res_iter = _plot_cross_seed_iterops(
                out_root, only=[scenario_stem], save_to=scen_iterops_png
            )
            if res_iter is None:
                raise ValueError(
                    f"No iterops data found to plot for scenario '{scenario_stem}'"
                )

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
        summary_mod.write_normalized_insights_csv(out_root)
        summary_mod.write_normalized_per_seed_csv(out_root)
    summary_mod.write_project_per_seed_abs_csv(out_root)

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
            summary_mod._print_normalized_insights(out_root)
            summary_mod.write_normalized_insights_csv(out_root)
            print("\n\n")
    text = buf.getvalue()
    summary_txt.write_text(text, encoding="utf-8")
    print(f"Wrote text summary: {summary_txt}")
    print(text, end="")
    print(f"Wrote project CSV: {project_csv}")

    if any(sha256_file(path) != digest for path, digest in input_hashes.items()):
        raise ValueError(
            "Source results changed during metric analysis; rerun on stable inputs"
        )
    selected_files = list(input_hashes)
    metrics_provenance = _create_metrics_provenance(
        root, out_root, selected_files, only
    )
    metrics_provenance["settings"] = {
        "enable_maxflow": enable_maxflow,
        "plots": do_plots,
    }

    for scenario_stem, seed_map in grouped.items():
        metrics_provenance["scenarios_analyzed"].append(scenario_stem)
        metrics_provenance["seeds_analyzed"][scenario_stem] = sorted(seed_map.keys())

    provenance_path = out_root / "provenance.json"
    write_json_atomic(provenance_path, metrics_provenance)
    print(f"📋 Metrics provenance saved to: {provenance_path}")


def _create_metrics_provenance(
    root: Path, out_root: Path, files: List[Path], only: Optional[str] = None
) -> Dict[str, Any]:
    """Record source files, hashes, and settings for a metrics run."""
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

    provenance["packages"] = _metric_package_versions()

    for file_path in files:
        try:
            file_hash = sha256_file(file_path)
            rel_path = os.path.relpath(file_path, start=cwd)
            provenance["source_files"][rel_path] = {
                "path": rel_path,
                "sha256": file_hash,
                "size_bytes": file_path.stat().st_size,
            }
        except (ValueError, TypeError, KeyError, OSError) as e:
            logging.warning("Failed to hash source file %s: %s", file_path, e)
            rel_path = os.path.relpath(file_path, start=cwd)
            provenance["source_files"][rel_path] = {
                "path": rel_path,
                "hash_error": str(e),
            }

    if only:
        provenance["only_scenarios"] = [s.strip() for s in only.split(",") if s.strip()]

    return provenance


def _metric_package_versions() -> dict:
    packages = package_versions()
    for name in ("numpy", "pandas", "scipy", "matplotlib", "seaborn"):
        packages[name] = {"version": version(name)}
    return packages
