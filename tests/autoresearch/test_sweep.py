"""Tests for Phase 2 sweep module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from netlab.autoresearch.structural_analysis import run_structural_analysis
from netlab.autoresearch.sweep import (
    ResultEntry,
    SweepConfig,
    _dedup_configs,
    print_results,
)


class TestResultEntry:
    def test_to_dict(self):
        e = ResultEntry(
            g_abc1=16,
            g_xyz1=64,
            layout_abc1="4r9c-16r1c",
            layout_xyz1="4r6c-4r1c",
            alpha_star=9.21,
            bac_combined=0.86,
            bac_modes={
                "lh_path": {"auc": 1.0, "pct": [1.0] * 100},
                "1x_bb": {"auc": 0.85, "pct": [0.75] * 50 + [1.0] * 50},
            },
            status="success",
        )
        d = e.to_dict()
        assert d["g_abc1"] == 16
        assert d["bac_combined"] == 0.86
        assert d["bac_modes"]["lh_path"]["auc"] == 1.0
        assert len(d["bac_modes"]["1x_bb"]["pct"]) == 100

    def test_from_dict_roundtrip(self):
        e = ResultEntry(
            g_abc1=64,
            g_xyz1=64,
            bac_modes={"lh_path": {"auc": 1.0, "pct": [1.0] * 100}},
            status="success",
        )
        d = e.to_dict()
        e2 = ResultEntry.from_dict(d)
        assert e2.bac_modes["lh_path"]["auc"] == 1.0
        assert len(e2.bac_modes["lh_path"]["pct"]) == 100

    def test_default_status(self):
        e = ResultEntry()
        assert e.status == "pending"


class TestDedup:
    def test_dedup_abc1(self):
        results = run_structural_analysis()
        deduped = _dedup_configs(results["abc1"].configs)
        assert len(deduped) == 9

    def test_dedup_xyz1(self):
        results = run_structural_analysis()
        deduped = _dedup_configs(results["xyz1"].configs)
        assert len(deduped) == 6


class TestSweepConfig:
    def test_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            sc = SweepConfig(output_dir=Path(td))
            assert sc.failure_iterations == 200
            assert sc.timeout_s == 300


class TestPrintResults:
    def test_prints_without_error(self, capsys):
        entries = [
            ResultEntry(
                g_abc1=16,
                g_xyz1=64,
                layout_abc1="4r9c-16r1c",
                layout_xyz1="4r6c-4r1c",
                alpha_star=9.21,
                bac_combined=0.95,
                bac_modes={"lh_path": 1.0, "1x_bb": 0.85},
                status="success",
                duration_s=45.0,
            ),
            ResultEntry(
                g_abc1=64,
                g_xyz1=64,
                layout_abc1="1r9c-4r1c",
                layout_xyz1="4r6c-4r1c",
                status="crash",
                error="inspect failed",
            ),
        ]
        print_results(entries)
        captured = capsys.readouterr()
        assert "1 success / 2 total" in captured.out
        assert "Failed: 1" in captured.out
