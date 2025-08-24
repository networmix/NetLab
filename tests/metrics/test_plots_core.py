from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from metrics.bac import compute_bac, plot_bac
from metrics.plot_bac_delta_vs_baseline import (
    _pooled_availability_curve,
    plot_bac_delta_vs_baseline,
)
from metrics.plot_cross_seed_bac import (
    _availability_curve_from_samples,
    _load_seed_bac,
    plot_cross_seed_bac,
)
from metrics.plot_cross_seed_iterops import (
    _load_iterops_median_per_iter,
    plot_cross_seed_iterops,
)
from metrics.plot_cross_seed_latency import (
    _availability_curve as _lat_availability_curve,
)
from metrics.plot_cross_seed_latency import (
    _load_seed_latency,
    plot_cross_seed_latency,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_cross_seed_bac_core_curves_match_pooled_grid() -> None:
    root = _repo_root() / "scenarios_metrics"
    for scen_dir in sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    ):
        # Load pooled grid from reference (written by metrics_cmd)
        import json

        bjs = json.loads((scen_dir / "bac_summary.json").read_text(encoding="utf-8"))
        pooled = bjs.get("pooled_grid", {}) or {}
        x_ref = np.asarray(pooled.get("x_pct", []), dtype=float)
        a_ref = np.asarray(pooled.get("availability", []), dtype=float)
        assert x_ref.size > 0 and a_ref.size == x_ref.size

        # Recompute from seed bac.json via plotting helper
        pooled_samples: list[float] = []
        for seed_dir in sorted([p for p in scen_dir.glob("seed*") if p.is_dir()]):
            samples_pct, _offered = _load_seed_bac(seed_dir)
            if samples_pct.size:
                pooled_samples.extend(samples_pct.tolist())
        xs, avail = _availability_curve_from_samples(
            np.asarray(pooled_samples, dtype=float)
        )
        assert xs.size == x_ref.size
        assert np.allclose(xs, x_ref, rtol=0.0, atol=0.0)
        assert np.allclose(avail, a_ref, rtol=0.0, atol=0.0)


def test_bac_delta_vs_baseline_core_matches_reference() -> None:
    root = _repo_root() / "scenarios_metrics"
    baseline = "small_baseline"
    base_x, base_a = _pooled_availability_curve(root / baseline)
    assert base_x.size > 0

    # Reference via bac_summary.json pooled grids
    import json

    base_js = json.loads(
        (root / baseline / "bac_summary.json").read_text(encoding="utf-8")
    )
    base_grid = np.asarray(base_js["pooled_grid"]["x_pct"], dtype=float)
    base_av = np.asarray(base_js["pooled_grid"]["availability"], dtype=float)
    # Ensure helper and reference are identical (sanity)
    assert np.allclose(base_x, base_grid)
    assert np.allclose(base_a, base_av)

    grid = np.linspace(80.0, 100.0, 1 + int(4 * (100.0 - 80.0)))
    base_on_grid = np.interp(grid, base_x, base_a, left=base_a[0], right=base_a[-1])

    for scen_dir in sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    ):
        if scen_dir.name == baseline:
            continue
        sx, sa = _pooled_availability_curve(scen_dir)
        if sx.size == 0:
            continue
        s_on_grid = np.interp(grid, sx, sa, left=sa[0], right=sa[-1])
        delta = s_on_grid - base_on_grid

        # Reference delta computed from bac_summary.json pooled grids
        sjs = json.loads((scen_dir / "bac_summary.json").read_text(encoding="utf-8"))
        sx_ref = np.asarray(sjs["pooled_grid"]["x_pct"], dtype=float)
        sa_ref = np.asarray(sjs["pooled_grid"]["availability"], dtype=float)
        s_on_grid_ref = np.interp(
            grid, sx_ref, sa_ref, left=sa_ref[0], right=sa_ref[-1]
        )
        delta_ref = s_on_grid_ref - base_on_grid
        assert np.allclose(delta, delta_ref, rtol=1e-12, atol=1e-12)


def test_iterops_medians_match_project_csv() -> None:
    root = _repo_root() / "scenarios_metrics"
    proj = pd.read_csv(root / "project.csv").set_index("scenario")
    for scen_dir in sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    ):
        series = _load_iterops_median_per_iter(scen_dir)
        assert series is not None
        row = proj.loc[scen_dir.name]
        # Columns in project.csv must match helper outputs
        s_spf = float(series.get("spf_calls_total_per_iter", float("nan")))
        s_flows = float(series.get("flows_created_total_per_iter", float("nan")))
        s_reopt = float(series.get("reopt_calls_total_per_iter", float("nan")))
        # Row is a Series; index access returns scalar
        r_spf = float(row["spf_calls_per_iter"])  # type: ignore[index]
        r_flows = float(row["flows_created_per_iter"])  # type: ignore[index]
        r_reopt = float(row["reopt_calls_per_iter"])  # type: ignore[index]
        assert np.isfinite(s_spf) and np.isfinite(r_spf) and np.isclose(s_spf, r_spf)
        assert (
            np.isfinite(s_flows)
            and np.isfinite(r_flows)
            and np.isclose(s_flows, r_flows)
        )
        assert (
            np.isfinite(s_reopt)
            and np.isfinite(r_reopt)
            and np.isclose(s_reopt, r_reopt)
        )


def test_cross_seed_latency_core_shapes() -> None:
    root = _repo_root() / "scenarios_metrics"
    for scen_dir in sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    ):
        pooled: list[float] = []
        for seed_dir in sorted([p for p in scen_dir.glob("seed*") if p.is_dir()]):
            arr = _load_seed_latency(seed_dir, metric="p99")
            pooled.extend(arr.tolist())
        if not pooled:
            continue
        xs, avail = _lat_availability_curve(np.asarray(pooled, dtype=float))
        assert xs.size > 0 and avail.size == xs.size
        # Monotonicity and range checks
        assert np.all(np.diff(xs) >= 0.0)
        assert np.all((avail >= 0.0) & (avail <= 1.0))
        assert np.all(np.diff(avail) <= 1e-12)  # non-increasing


def test_plot_functions_save_files(tmp_path: Path) -> None:
    root = _repo_root() / "scenarios_metrics"
    # Save BAC figure
    bac_png = tmp_path / "BAC.png"
    out = plot_cross_seed_bac(
        root, only=["small_baseline", "small_clos"], save_to=bac_png
    )
    assert out is not None and out.exists()

    # Save latency figure
    lat_png = tmp_path / "Latency_p99.png"
    out2 = plot_cross_seed_latency(
        root, metric="p99", only=["small_baseline", "small_clos"], save_to=lat_png
    )
    assert out2 is not None and out2.exists()

    # Save iterops figure
    iterops_png = tmp_path / "IterationOps.png"
    out3 = plot_cross_seed_iterops(
        root, only=["small_baseline", "small_clos"], save_to=iterops_png
    )
    assert out3 is not None and out3.exists()

    # Save BAC delta vs baseline
    delta_png = tmp_path / "BAC_delta_vs_baseline.png"
    out4 = plot_bac_delta_vs_baseline(
        root, baseline="small_baseline", save_to=delta_png
    )
    assert out4 is not None and out4.exists()


def test_plot_bac_overlay_saves(tmp_path: Path) -> None:
    # Synthetic small BAC results to exercise overlay path
    results = {
        "workflow": {"tm": {"step_type": "TrafficMatrixPlacement"}},
        "steps": {
            "tm": {
                "metadata": {"baseline": True},
                "data": {
                    "flow_results": [
                        {
                            "failure_id": "baseline",
                            "flows": [
                                {
                                    "source": "A",
                                    "destination": "B",
                                    "placed": 100.0,
                                    "demand": 100.0,
                                }
                            ],
                        },
                        {
                            "failure_id": "f1",
                            "flows": [
                                {
                                    "source": "A",
                                    "destination": "B",
                                    "placed": 80.0,
                                    "demand": 100.0,
                                }
                            ],
                        },
                    ]
                },
            }
        },
    }
    bac_primary = compute_bac(results, step_name="tm", mode="auto")
    # Make an overlay with different values (pretend a different step)
    results2 = {
        "workflow": {"tm": {"step_type": "MaxFlow"}},
        "steps": {
            "tm": {
                "metadata": {"baseline": True},
                "data": {
                    "flow_results": [
                        {
                            "failure_id": "baseline",
                            "flows": [
                                {
                                    "source": "A",
                                    "destination": "B",
                                    "placed": 120.0,
                                    "demand": 120.0,
                                }
                            ],
                        },
                        {
                            "failure_id": "f1",
                            "flows": [
                                {
                                    "source": "A",
                                    "destination": "B",
                                    "placed": 100.0,
                                    "demand": 120.0,
                                }
                            ],
                        },
                    ]
                },
            }
        },
    }
    bac_overlay = compute_bac(results2, step_name="tm", mode="auto")
    outp = tmp_path / "bac_overlay.png"
    plot_bac(bac_primary, overlay=bac_overlay, save_to=outp)
    assert outp.exists()
