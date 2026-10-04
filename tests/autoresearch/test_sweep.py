"""Tests for DC-BB simulation sweeps."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from netlab.autoresearch.structural_analysis import run_structural_analysis
from netlab.autoresearch.sweep import (
    ResultEntry,
    SweepConfig,
    _dedup_configs,
    _extract_step_metrics,
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


def test_step_metrics_preserve_destinations_and_weights():
    results = {
        "steps": {
            "tm": {
                "data": {
                    "baseline": {
                        "flows": [
                            {
                                "source": "A",
                                "destination": "B",
                                "demand": 100.0,
                                "placed": 100.0,
                            },
                            {
                                "source": "A",
                                "destination": "C",
                                "demand": 200.0,
                                "placed": 200.0,
                            },
                        ]
                    },
                    "flow_results": [
                        {
                            "occurrence_count": 2,
                            "failure_state": {},
                            "flows": [
                                {"source": "A", "destination": "B", "placed": 0.0},
                                {"source": "A", "destination": "C", "placed": 100.0},
                            ],
                        }
                    ],
                }
            }
        }
    }
    metrics = _extract_step_metrics(results, "tm")
    assert metrics["auc"] == pytest.approx(5 / 9, abs=1e-6)
    assert set(metrics["flow_bac"]) == {"A>B", "A>C"}
    assert metrics["flow_bac"]["A>B"]["auc"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["flow_bac"]["A>C"]["auc"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["flow_bac"]["A>B"]["pct"][49] == 0.0
    assert metrics["flow_bac"]["A>C"]["pct"][49] == 0.5
    assert metrics["failure_stats"]["event_count"] == 2


def test_sweep_retries_failed_entries_and_refuses_changed_inputs(tmp_path):
    import json

    from netlab.autoresearch.sweep import _load_completed, _prepare_sweep

    config = SweepConfig(tmp_path)
    _prepare_sweep(config, "abc1")
    records = tmp_path / "results.jsonl"
    records.write_text(
        "\n".join(
            json.dumps({"status": status, "result_dir": status})
            for status in ["success", "error", "timeout"]
        )
    )
    assert _load_completed(records) == {"success"}
    _prepare_sweep(config, "abc1")
    config.seed += 1
    with pytest.raises(ValueError, match="inputs or dependencies changed"):
        _prepare_sweep(config, "abc1")
