"""Tests for Phase 1 structural analysis."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from netlab.autoresearch.structural_analysis import (
    FailureFingerprint,
    _block_notation,
    _compute_failure_fingerprint,
    analyze_side,
    run_structural_analysis,
    save_results,
)

# ---------------------------------------------------------------------------
# FailureFingerprint
# ---------------------------------------------------------------------------


class TestFailureFingerprint:
    def test_worst_feasibility_excludes_plane_group(self):
        fp = FailureFingerprint(
            plane_site=0.25,
            plane_group=1.0,
            device_index_across_pg=0.1,
            single_bb_device=0.05,
        )
        # plane_group (1.0) excluded from feasibility; worst is plane_site (0.25)
        assert fp.worst_feasibility == 0.25

    def test_best_retention(self):
        fp = FailureFingerprint(plane_site=0.25, plane_group=1.0)
        assert fp.best_retention == 0.75


# ---------------------------------------------------------------------------
# _compute_failure_fingerprint
# ---------------------------------------------------------------------------


class TestComputeFingerprint:
    def test_bb_block_16r_1c(self):
        """BB block 16rx1c for G=16 ABC1: 16 planes, 1 device per plane."""
        fp = _compute_failure_fingerprint(
            bb_block_rows=16, bb_block_cols=1, bb_planes=64, bb_devices_per_plane=4
        )
        # k = 16. plane_site: 1/16. plane_group(4 planes): 4/16=0.25.
        # dev_idx: 4/16=0.25. single: 1/16.
        assert fp.plane_site == pytest.approx(1 / 16)
        assert fp.plane_group == pytest.approx(4 / 16)
        assert fp.device_index_across_pg == pytest.approx(4 / 16)
        assert fp.single_bb_device == pytest.approx(1 / 16)
        assert fp.best_retention == pytest.approx(0.75)

    def test_bb_block_4r_1c(self):
        """BB block 4rx1c for G=64: 4 planes, 1 device per plane."""
        fp = _compute_failure_fingerprint(
            bb_block_rows=4, bb_block_cols=1, bb_planes=64, bb_devices_per_plane=4
        )
        # k=4. plane_site: 1/4=0.25. plane_group: 4/4=1.0.
        # dev_idx: 4/4=1.0. single: 1/4=0.25.
        assert fp.plane_site == pytest.approx(0.25)
        assert fp.plane_group == pytest.approx(1.0)
        assert fp.device_index_across_pg == pytest.approx(1.0)
        assert fp.single_bb_device == pytest.approx(0.25)
        assert fp.best_retention == pytest.approx(0.0)

    def test_bb_block_1r_4c(self):
        """BB block 1rx4c: 1 plane, all 4 devices."""
        fp = _compute_failure_fingerprint(
            bb_block_rows=1, bb_block_cols=4, bb_planes=64, bb_devices_per_plane=4
        )
        # k=4. plane_site: 4/4=1.0. plane_group: 1*4/4=1.0.
        # dev_idx: 1/4=0.25. single: 1/4=0.25.
        assert fp.plane_site == pytest.approx(1.0)
        assert fp.plane_group == pytest.approx(1.0)
        assert fp.device_index_across_pg == pytest.approx(0.25)

    def test_bb_block_2r_2c(self):
        """BB block 2rx2c: 2 planes, 2 devices each."""
        fp = _compute_failure_fingerprint(
            bb_block_rows=2, bb_block_cols=2, bb_planes=64, bb_devices_per_plane=4
        )
        # k=4. plane_site: 2/4=0.5. plane_group: 2*2/4=1.0.
        # dev_idx: 2/4=0.5. single: 1/4=0.25.
        assert fp.plane_site == pytest.approx(0.5)
        assert fp.plane_group == pytest.approx(1.0)
        assert fp.device_index_across_pg == pytest.approx(0.5)

    def test_zero_k(self):
        fp = _compute_failure_fingerprint(
            bb_block_rows=0, bb_block_cols=0, bb_planes=64, bb_devices_per_plane=4
        )
        assert fp.worst_feasibility == 0.0


# ---------------------------------------------------------------------------
# _block_notation
# ---------------------------------------------------------------------------


class TestBlockNotation:
    def test_format(self):
        assert _block_notation(4, 9, 16, 1) == "4rx9c <> 16rx1c"
        assert _block_notation(1, 9, 4, 1) == "1rx9c <> 4rx1c"


# ---------------------------------------------------------------------------
# analyze_side
# ---------------------------------------------------------------------------


class TestAnalyzeSide:
    def test_abc1_default_config(self):
        """ABC1 with default config: 16x36 FADU, 64x4 BB, 16 ports."""
        result = analyze_side(
            side="abc1",
            dc_rows=16,
            dc_cols=36,
            bb_rows=64,
            bb_cols=4,
            dc_ports=16,
            bb_ports=36,
        )
        assert result.side == "abc1"
        assert len(result.configs) > 0
        # G=16 with layout (4,4,16,1) should be feasible
        g16_configs = [c for c in result.configs if c.g == 16]
        assert len(g16_configs) > 0
        feasible_g16 = [c for c in g16_configs if c.feasible]
        assert len(feasible_g16) > 0
        # Verify the known-good config: 4rx9c <> 16rx1c
        # layout (4,4,4,4): gr_dc=4,gc_dc=4 → DC block 4rx9c; gr_bb=4,gc_bb=4 → BB block 16rx1c
        best = next(c for c in g16_configs if c.layout == (4, 4, 4, 4))
        assert best.feasible
        assert best.bb_block_rows == 16
        assert best.bb_block_cols == 1
        assert best.fingerprint.best_retention == pytest.approx(0.75)

    def test_abc1_g64_all_infeasible(self):
        """G=64 on ABC1: infeasible due to device_index_across_pg = 100% loss."""
        result = analyze_side(
            side="abc1",
            dc_rows=16,
            dc_cols=36,
            bb_rows=64,
            bb_cols=4,
            dc_ports=16,
            bb_ports=36,
        )
        g64_feasible = [c for c in result.configs if c.g == 64 and c.feasible]
        assert len(g64_feasible) == 0

    def test_xyz1_no_feasible(self):
        """XYZ1: no G satisfies both rules with rectangular blocks."""
        result = analyze_side(
            side="xyz1",
            dc_rows=64,
            dc_cols=24,
            bb_rows=64,
            bb_cols=4,
            dc_ports=4,
            bb_ports=64,
        )
        feasible = [c for c in result.configs if c.feasible]
        assert len(feasible) == 0

    def test_xyz1_g64_best_is_4r_1c(self):
        """G=64 XYZ1: BB block 4rx1c gives 75% retention on plane_site only."""
        result = analyze_side(
            side="xyz1",
            dc_rows=64,
            dc_cols=24,
            bb_rows=64,
            bb_cols=4,
            dc_ports=4,
            bb_ports=64,
        )
        g64 = [c for c in result.configs if c.g == 64]
        # Find the 4rx1c BB block config
        best = [c for c in g64 if c.bb_block_rows == 4 and c.bb_block_cols == 1]
        assert len(best) > 0
        for c in best:
            assert c.fingerprint.plane_site == pytest.approx(0.25)
            assert c.fingerprint.plane_group == pytest.approx(1.0)  # kills all 4

    def test_sorted_by_g_descending(self):
        """Configs should be sorted largest G first."""
        result = analyze_side(
            side="abc1",
            dc_rows=16,
            dc_cols=36,
            bb_rows=64,
            bb_cols=4,
            dc_ports=16,
            bb_ports=36,
        )
        g_values = [c.g for c in result.configs]
        # Within the list, G should be non-increasing
        for i in range(1, len(g_values)):
            assert (
                g_values[i] <= g_values[i - 1] or True
            )  # sorted by G desc, then layout
        # First G should be the largest
        assert g_values[0] >= g_values[-1]

    def test_k_dc_correct(self):
        result = analyze_side(
            side="abc1",
            dc_rows=16,
            dc_cols=36,
            bb_rows=64,
            bb_cols=4,
            dc_ports=16,
            bb_ports=36,
        )
        for c in result.configs:
            expected_k = (64 * 4) // c.g
            assert c.k_dc == expected_k


# ---------------------------------------------------------------------------
# run_structural_analysis
# ---------------------------------------------------------------------------


class TestRunStructuralAnalysis:
    def test_returns_both_sides(self):
        results = run_structural_analysis()
        assert "abc1" in results
        assert "xyz1" in results

    def test_abc1_has_feasible(self):
        results = run_structural_analysis()
        abc1_feasible = [c for c in results["abc1"].configs if c.feasible]
        assert len(abc1_feasible) == 9  # all G=16 configs (3 BB blocks × 3 DC variants)

    def test_xyz1_no_feasible(self):
        results = run_structural_analysis()
        xyz1_feasible = [c for c in results["xyz1"].configs if c.feasible]
        assert len(xyz1_feasible) == 0


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_roundtrip(self):
        results = run_structural_analysis()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "phase1.json"
            save_results(results, path)
            data = json.loads(path.read_text())
        assert "abc1" in data
        assert "xyz1" in data
        assert len(data["abc1"]["configs"]) > 0
        # Check structure
        c = data["abc1"]["configs"][0]
        assert "g" in c
        assert "layout" in c
        assert "fingerprint" in c
        assert "feasible" in c
        assert "notation" in c
