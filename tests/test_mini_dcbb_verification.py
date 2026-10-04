"""Run the mini DC-BB scenario and check its metrics against hand calculations.

The 10-node topology has asymmetric parallel paths.

Topology: 2 sites, 2 planes, dual LH paths (path_a: 100km/100cap,
path_b: 200km/50cap). Flow policy: TE_WCMP_UNLIM.

See tests/data/mini_dcbb.yaml for full topology documentation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from netlab.metrics.bac import compute_bac
from netlab.metrics.latency import compute_latency_stretch
from netlab.metrics.msd import compute_alpha_star


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> dict:
    output = tmp_path_factory.mktemp("mini-dcbb")
    scenario = Path(__file__).parent / "data" / "mini_dcbb.yaml"
    run = subprocess.run(
        [sys.executable, "-m", "ngraph", "run", str(scenario), "-o", str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    with (output / "mini_dcbb.results.json").open() as f:
        return json.load(f)


def test_alpha_star(results: dict) -> None:
    """alpha_star = total_cross_site_capacity / demand = 300/100 = 3.0."""
    alpha = compute_alpha_star(results)
    assert alpha.alpha_star == 3.0
    assert alpha.source == "msd_baseline"


def test_bac_lh_path_offered(results: dict) -> None:
    """Baseline delivers 600 (2 flows × 300 each)."""
    bac = compute_bac(results, step_name="tm_lh_path")
    assert bac.offered == 600.0


def test_bac_lh_path_series_length(results: dict) -> None:
    """1 baseline + 10 failure iterations (expanded by occurrence_count)."""
    bac = compute_bac(results, step_name="tm_lh_path")
    assert len(bac.series) == 11


def test_bac_lh_path_auc(results: dict) -> None:
    """AUC = (1.0 + 5×(200/600) + 5×(400/600)) / 11 = 6/11."""
    bac = compute_bac(results, step_name="tm_lh_path")
    assert np.isclose(bac.auc_normalized, 6.0 / 11.0)


def test_bac_lh_path_per_flow_keys(results: dict) -> None:
    """Two directions produce two per-flow BAC entries."""
    bac = compute_bac(results, step_name="tm_lh_path")
    assert len(bac.per_flow) == 2
    labels = set(bac.per_flow.keys())
    assert "abc1/rsw>xyz1/rsw" in labels
    assert "xyz1/rsw>abc1/rsw" in labels


def test_bac_lh_path_per_flow_symmetric(results: dict) -> None:
    """Symmetric topology: both directions have identical AUC.

    Per-flow offered=300 (one flow's baseline placed).
    Per-flow series: [300, 100×5, 200×5].
    AUC = (1 + 5×(1/3) + 5×(2/3)) / 11 = 6/11.
    """
    bac = compute_bac(results, step_name="tm_lh_path")
    for _label, pf in bac.per_flow.items():
        assert pf.offered == 300.0
        assert len(pf.series) == 11
        assert np.isclose(pf.auc_normalized, 6.0 / 11.0)


def test_bac_1x_bb_series_length(results: dict) -> None:
    """1 baseline + 20 failure iterations."""
    bac = compute_bac(results, step_name="tm_1x_bb")
    assert len(bac.series) == 21


def test_bac_1x_bb_auc(results: dict) -> None:
    """AUC = (1.0 + 20×0.5) / 21 = 11/21."""
    bac = compute_bac(results, step_name="tm_1x_bb")
    assert np.isclose(bac.auc_normalized, 11.0 / 21.0)


@pytest.fixture
def latency_lh(results: dict) -> dict:
    """Map tm_lh_path to the latency reader's tm_placement step."""
    return {"steps": {"tm_placement": results["steps"]["tm_lh_path"]}}


def test_latency_lh_baseline_p50(latency_lh: dict) -> None:
    """Baseline: shortest cost is 112, p50 stretch = 1.0."""
    lat = compute_latency_stretch(latency_lh)
    assert np.isclose(lat.baseline["p50"], 1.0)


def test_latency_lh_baseline_wes(latency_lh: dict) -> None:
    """Baseline WES = (200×0 + 100×(212/112-1)) / 300."""
    lat = compute_latency_stretch(latency_lh)
    expected = 100.0 * (212.0 / 112.0 - 1.0) / 300.0
    assert np.isclose(lat.baseline["WES"], expected)


def test_latency_lh_baseline_best_path_share(latency_lh: dict) -> None:
    """200 of 300 volume travels at baseline min cost → 2/3."""
    lat = compute_latency_stretch(latency_lh)
    assert np.isclose(lat.baseline["best_path_share"], 2.0 / 3.0)


def test_latency_lh_failures_p50(latency_lh: dict) -> None:
    """Median p50 across 5×(stretch=212/112) + 5×(stretch=1.0).

    Sorted: [1.0]*5 + [1.893]*5. Median = avg of 5th and 6th = (1+212/112)/2.
    """
    lat = compute_latency_stretch(latency_lh)
    expected = (1.0 + 212.0 / 112.0) / 2.0
    assert np.isclose(lat.failures["p50"], expected)


def test_latency_lh_failures_wes(latency_lh: dict) -> None:
    """Median WES across 5×0.893 + 5×0.0 = avg of 5th and 6th sorted."""
    lat = compute_latency_stretch(latency_lh)
    expected = (0.0 + (212.0 / 112.0 - 1.0)) / 2.0
    assert np.isclose(lat.failures["WES"], expected)


@pytest.fixture
def latency_bb(results: dict) -> dict:
    return {"steps": {"tm_placement": results["steps"]["tm_1x_bb"]}}


def test_latency_bb_failures_p50(latency_bb: dict) -> None:
    """All 20 iterations: cdist={112:100, 212:50}. p50 at cumwt 0.5 → 1.0."""
    lat = compute_latency_stretch(latency_bb)
    assert np.isclose(lat.failures["p50"], 1.0)


def test_latency_bb_failures_wes(latency_bb: dict) -> None:
    """WES = 50×(212/112-1) / 150."""
    lat = compute_latency_stretch(latency_bb)
    expected = 50.0 * (212.0 / 112.0 - 1.0) / 150.0
    assert np.isclose(lat.failures["WES"], expected)
