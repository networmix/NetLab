from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from metrics.summary import (
    _build_normalized_insights,
    _holm_adjust,
    _paired_t_with_ci,
    _print_normalized_insights,
    build_baseline_normalized_table,
    build_project_summary_table,
    print_pretty_table,
    save_project_csv_incremental,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_build_project_summary_table_matches_csv() -> None:
    root = _repo_root() / "scenarios_metrics"
    df = build_project_summary_table(root).sort_index()
    ref = pd.read_csv(root / "project.csv").set_index("scenario").sort_index()
    # Column set may evolve; require at least the expected ones present
    for col in ("seeds", "node_count", "link_count", "alpha_star", "bac_auc"):
        assert col in df.columns
    # Index equality and numeric similarity on overlapping columns
    assert list(df.index) == list(ref.index)
    common = [
        c
        for c in ref.columns
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in common:
        assert np.allclose(
            pd.to_numeric(df[c], errors="coerce"),
            pd.to_numeric(ref[c], errors="coerce"),
            equal_nan=True,
        )


def test_build_baseline_normalized_table_has_expected_columns() -> None:
    root = _repo_root() / "scenarios_metrics"
    df = build_baseline_normalized_table(root).sort_index()
    assert "baseline" in df.columns
    # A few key normalized metrics should be present
    for c in ("bw_p99_pct_r", "auc_norm_r", "USD_per_Gbit_offered_r"):
        assert c in df.columns


def test_print_pretty_table_does_not_crash(capsys) -> None:
    root = _repo_root() / "scenarios_metrics"
    df = build_project_summary_table(root)
    print_pretty_table(df, title="Consolidated project metrics")
    out = capsys.readouterr().out
    # It should print some header or rows
    assert "project" in out or "scenario" in out or "metrics" in out


def test_print_pretty_table_fallback_no_rich(monkeypatch, capsys) -> None:
    import metrics.summary as summ

    # Force fallback path by nulling Console/RichTable
    monkeypatch.setattr(summ, "Console", None)
    monkeypatch.setattr(summ, "RichTable", None)
    root = _repo_root() / "scenarios_metrics"
    df = build_project_summary_table(root)
    print_pretty_table(df, title="Consolidated project metrics")
    out = capsys.readouterr().out
    assert "scenario" in out


def test_save_project_csv_incremental(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "scenario": ["a", "b"],
            "seeds": [1, 2],
            "node_count": [10.0, 20.0],
        }
    ).set_index("scenario")
    p1 = save_project_csv_incremental(df, cwd=tmp_path)
    p2 = save_project_csv_incremental(df, cwd=tmp_path)
    assert p1.exists() and p2.exists() and p1 != p2


def test_build_normalized_insights_shape() -> None:
    root = _repo_root() / "scenarios_metrics"
    data = _build_normalized_insights(root)
    # Returns a list of dicts with scenario key
    assert isinstance(data, list)
    if data:
        assert "scenario" in data[0]


def test_holm_and_paired_t_helpers() -> None:
    # _holm_adjust monotonic step-down behavior
    pairs = [(("a", "b"), 0.01), (("a", "c"), 0.02), (("b", "c"), 0.50)]
    adjusted = _holm_adjust(pairs)
    assert adjusted[("a", "b")] <= adjusted[("a", "c")] <= adjusted[("b", "c")]

    # _paired_t_with_ci degenerate case (zero variance) and regular case
    import numpy as _np

    a = _np.array([1.0, 1.0, 1.0, 1.0])
    b = _np.array([0.0, 0.0, 0.0, 0.0])
    import math as _math

    res_det = _paired_t_with_ci(a, b)
    assert res_det.get("deterministic") is True
    # Degenerate case may yield infinite t; accept inf
    t_det = res_det.get("t_stat", _np.nan)
    assert _math.isinf(t_det) or _np.isfinite(t_det)

    a2 = _np.array([1.0, 2.0, 3.0, 4.0])
    b2 = _np.array([0.5, 2.0, 1.0, 3.5])
    res = _paired_t_with_ci(a2, b2)
    assert res.get("deterministic") is False
    assert _np.isfinite(res.get("t_stat", _np.nan))


def test_normalized_table_respects_env_baseline(monkeypatch) -> None:
    root = _repo_root() / "scenarios_metrics"
    # Force baseline to a non-default scenario and verify ratios/deltas for baseline row
    monkeypatch.setenv("NGRAPH_BASELINE_SCENARIO", "small_clos")
    df = build_baseline_normalized_table(root)
    assert "small_clos" in df.index
    row = df.loc["small_clos"]
    # Baseline row expected defaults: ratios 1.0, deltas 0.0
    for c in list(row.index):
        if c.endswith("_r"):
            val = row.get(c)
            if pd.notna(val):
                assert float(val) == 1.0
        if c.endswith("_d"):
            val = row.get(c)
            if pd.notna(val):
                assert float(val) == 0.0


def test_normalized_table_heuristic_baseline_when_missing_baseline_dir(
    tmp_path: Path,
) -> None:
    # Create a temp analysis root with only two non-baseline scenarios
    repo_root = _repo_root() / "scenarios_metrics"
    for scen in ("small_clos", "small_dragonfly"):
        src = repo_root / scen
        dst = tmp_path / scen
        # Copy directory tree
        for p in src.rglob("*"):
            rel = p.relative_to(src)
            target = dst / rel
            if p.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(p.read_bytes())
    df = build_baseline_normalized_table(tmp_path)
    # Heuristic picks first alphabetical (small_clos)
    assert "small_clos" in df.index
    row = df.loc["small_clos"]
    for c in list(row.index):
        if c.endswith("_r"):
            val = row.get(c)
            if pd.notna(val):
                assert float(val) == 1.0
        if c.endswith("_d"):
            val = row.get(c)
            if pd.notna(val):
                assert float(val) == 0.0


def test_print_normalized_insights_smoke(capsys) -> None:
    root = _repo_root() / "scenarios_metrics"
    _print_normalized_insights(root)
    out = capsys.readouterr().out
    # Either table header or explicit absence message
    # Function prints to stdout; accept no output (no scenarios) or header/message
    assert (
        out == ""
        or ("All baseline-normalized comparisons" in out)
        or ("no baseline-normalized insights available" in out)
    )
