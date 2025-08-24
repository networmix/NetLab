from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from metrics.summary import (
    build_baseline_normalized_table,
    build_project_summary_table,
)
from netlab.metrics_cmd import run_metrics


def _copy_results_json_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*.results.json"):
        rel = p.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def _allclose_series(
    a: pd.Series, b: pd.Series, rtol: float = 1e-9, atol: float = 1e-9
) -> bool:
    av = pd.to_numeric(a, errors="coerce").astype(float).values
    bv = pd.to_numeric(b, errors="coerce").astype(float).values
    if av.shape != bv.shape:
        return False
    # Use lists to satisfy conservative numpy type stubs
    av_list = [float(x) for x in av]
    bv_list = [float(x) for x in bv]
    return np.allclose(av_list, bv_list, rtol=rtol, atol=atol, equal_nan=True)


def _assert_project_csv_equal(a_path: Path, b_path: Path) -> None:
    a = pd.read_csv(a_path).set_index("scenario").sort_index()
    b = pd.read_csv(b_path).set_index("scenario").sort_index()
    assert list(a.index) == list(b.index)
    assert set(a.columns) == set(b.columns)
    # Reindex columns to common ordering
    cols = sorted(a.columns)
    a = a[cols]
    b = b[cols]
    for col in cols:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(
            b[col]
        ):
            assert _allclose_series(a[col], b[col])
        else:
            pd.testing.assert_series_equal(
                a[col].astype(str), b[col].astype(str), check_names=False
            )


def test_run_metrics_matches_reference_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_scen = repo_root / "scenarios"
    dst_scen = tmp_path / "scenarios"
    _copy_results_json_tree(src_scen, dst_scen)

    # Run end-to-end analysis on the copied scenarios tree (no maxflow; matches reference)
    run_metrics(root=dst_scen, no_plots=True, enable_maxflow=False)

    out_root = tmp_path / "scenarios_metrics"
    ref_root = repo_root / "scenarios_metrics"

    # Compare project-level CSVs (values should match exactly)
    _assert_project_csv_equal(out_root / "project.csv", ref_root / "project.csv")
    _assert_project_csv_equal(
        out_root / "project_baseline_normalized.csv",
        ref_root / "project_baseline_normalized.csv",
    )

    # Spot-check a few per-seed JSON outputs for equality
    ref_alpha = json.loads(
        (ref_root / "small_baseline/seed11/alpha.json").read_text(encoding="utf-8")
    )
    out_alpha = json.loads(
        (out_root / "small_baseline/seed11/alpha.json").read_text(encoding="utf-8")
    )
    assert out_alpha.get("alpha_star") == ref_alpha.get("alpha_star")

    ref_bac = json.loads(
        (ref_root / "small_clos/seed12/bac.json").read_text(encoding="utf-8")
    )
    out_bac = json.loads(
        (out_root / "small_clos/seed12/bac.json").read_text(encoding="utf-8")
    )
    # Key summary fields must match
    for k in ("auc_normalized",):
        assert float(out_bac.get(k)) == float(ref_bac.get(k))
    # Normalized BW at probability entries must match
    for p in ("90.0", "95.0", "99.0", "99.9"):
        assert float(out_bac["bw_at_probability_pct"][p]) == float(
            ref_bac["bw_at_probability_pct"][p]
        )


def test_summary_tables_match_reference() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ref_root = repo_root / "scenarios_metrics"

    # Build tables from the reference metrics dir and compare to the saved CSVs
    df_proj = build_project_summary_table(ref_root).sort_index()
    df_proj_ref = (
        pd.read_csv(ref_root / "project.csv").set_index("scenario").sort_index()
    )
    assert list(df_proj.index) == list(df_proj_ref.index)
    for col in df_proj_ref.columns:
        if col not in df_proj.columns:
            # Allow extra columns in CSV (future extensions), but ensure core columns are present
            continue
        s1 = df_proj[col]
        s2 = df_proj_ref[col]
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            assert _allclose_series(s1, s2)
        else:
            pd.testing.assert_series_equal(
                s1.astype(str), s2.astype(str), check_names=False
            )


def _flatten_json_numeric(d: object, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_json_numeric(v, key))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            key = f"{prefix}[{i}]"
            out.update(_flatten_json_numeric(v, key))
    else:
        try:
            if isinstance(d, (int, float)) and not (
                isinstance(d, float) and np.isnan(d)
            ):
                out[prefix] = float(d)
        except Exception:
            pass
    return out


def _compare_json_numeric_close(
    a: dict, b: dict, rtol: float = 1e-9, atol: float = 1e-9
) -> None:
    fa = _flatten_json_numeric(a)
    fb = _flatten_json_numeric(b)
    assert set(fa.keys()) == set(fb.keys())
    for k in fa:
        av = float(fa[k])
        bv = float(fb[k])
        assert np.isclose(av, bv, rtol=rtol, atol=atol)


def _assert_csv_equal(a: Path, b: Path) -> None:
    da = pd.read_csv(a)
    db = pd.read_csv(b)
    assert list(da.columns) == list(db.columns)
    assert da.shape == db.shape
    for col in da.columns:
        if pd.api.types.is_numeric_dtype(da[col]) and pd.api.types.is_numeric_dtype(
            db[col]
        ):
            assert _allclose_series(da[col], db[col])
        else:
            pd.testing.assert_series_equal(
                da[col].astype(str), db[col].astype(str), check_names=False
            )


def test_all_scenario_metrics_match_reference(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_scen = repo_root / "scenarios"
    dst_scen = tmp_path / "scenarios"
    _copy_results_json_tree(src_scen, dst_scen)

    # Generate outputs
    run_metrics(root=dst_scen, no_plots=True, enable_maxflow=False)

    out_root = tmp_path / "scenarios_metrics"
    ref_root = repo_root / "scenarios_metrics"

    # For each scenario dir in reference, compare expected per-scenario files and per-seed files
    for scen_dir in sorted(
        [p for p in ref_root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    ):
        scen = scen_dir.name
        out_scen = out_root / scen
        assert out_scen.exists()

        # Scenario-level CSV/JSON summaries
        for fname in (
            "alpha_summary.json",
            "bac_summary.json",
            "latency_summary.csv",
            "iterops_summary.csv",
            "network_stats_summary.csv",
            "costpower_summary.csv",
        ):
            ref_p = scen_dir / fname
            out_p = out_scen / fname
            assert ref_p.exists(), f"missing reference {ref_p}"
            assert out_p.exists(), f"missing output {out_p}"
            if fname.endswith(".json"):
                ref_j = json.loads(ref_p.read_text(encoding="utf-8"))
                out_j = json.loads(out_p.read_text(encoding="utf-8"))
                _compare_json_numeric_close(out_j, ref_j)
            else:
                _assert_csv_equal(out_p, ref_p)

        # Per-seed JSON/CSV
        for seed_dir in sorted(
            [p for p in scen_dir.iterdir() if p.is_dir() and p.name.startswith("seed")]
        ):
            out_seed = out_scen / seed_dir.name
            assert out_seed.exists()
            for fname in (
                "alpha.json",
                "bac.json",
                "latency.json",
                "costpower.json",
                "iterops.json",
            ):
                ref_p = seed_dir / fname
                out_p = out_seed / fname
                if ref_p.exists():
                    assert out_p.exists(), f"missing output {out_p}"
                    ref_j = json.loads(ref_p.read_text(encoding="utf-8"))
                    out_j = json.loads(out_p.read_text(encoding="utf-8"))
                    _compare_json_numeric_close(out_j, ref_j)
            for csv_name in ("pairs_tm_abs.csv", "pairs_tm_norm.csv"):
                ref_csv = seed_dir / csv_name
                out_csv = out_seed / csv_name
                if ref_csv.exists():
                    assert out_csv.exists(), f"missing output {out_csv}"
                    _assert_csv_equal(out_csv, ref_csv)
    df_norm = build_baseline_normalized_table(ref_root).sort_index()
    df_norm_ref = (
        pd.read_csv(ref_root / "project_baseline_normalized.csv")
        .set_index("scenario")
        .sort_index()
    )
    assert list(df_norm.index) == list(df_norm_ref.index)
    for col in df_norm_ref.columns:
        if col not in df_norm.columns:
            continue
        s1 = df_norm[col]
        s2 = df_norm_ref[col]
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            assert _allclose_series(s1, s2)
        else:
            pd.testing.assert_series_equal(
                s1.astype(str), s2.astype(str), check_names=False
            )
