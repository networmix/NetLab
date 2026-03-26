"""Tests for the metrics-first analysis loop."""

from __future__ import annotations

import json
from pathlib import Path

from netlab.autoresearch.analysis_loop import run_analysis_loop
from netlab.autoresearch.backend import MockBackend
from netlab.autoresearch.metrics_report import build_metrics_report

MINI_DCBB_RESULTS = (
    Path(__file__).parent.parent
    / "data"
    / "mini_dcbb_output"
    / "mini_dcbb.results.json"
)


def _load_results() -> dict:
    with MINI_DCBB_RESULTS.open() as f:
        return json.load(f)


class TestMetricsReport:
    def test_contains_alpha(self) -> None:
        results = _load_results()
        report = build_metrics_report(results)
        assert "alpha_star: 3.0" in report

    def test_contains_bac_auc(self) -> None:
        results = _load_results()
        report = build_metrics_report(results)
        assert "AUC: 0.5455" in report

    def test_contains_per_direction(self) -> None:
        results = _load_results()
        report = build_metrics_report(results)
        assert "abc1/rsw>xyz1/rsw" in report

    def test_contains_latency(self) -> None:
        results = _load_results()
        report = build_metrics_report(results)
        assert "baseline p50: 1.0" in report

    def test_contains_failure_patterns(self) -> None:
        results = _load_results()
        report = build_metrics_report(results)
        assert "unique patterns: 2" in report  # tm_lh_path
        assert "unique patterns: 4" in report  # tm_1x_bb


class TestAnalysisLoop:
    def test_produces_interpretation_and_next_hypothesis(self) -> None:
        """LLM receives verified metrics and produces interpretation + next hypothesis."""
        results = _load_results()
        backend = MockBackend(
            [
                "The topology has 2 planes with asymmetric LH paths. BAC AUC of 0.55 for lh_path reflects the 50/50 chance of losing path_a (severe) vs path_b (mild).",
                "Test a 3-plane topology with equal capacity paths to see if BAC improves.",
            ]
        )
        result = run_analysis_loop(
            results=results,
            hypothesis="Test dual LH paths with 2:1 capacity ratio",
            backend=backend,
        )
        assert result.complete
        assert "2 planes" in result.interpretation
        assert result.next_hypothesis != ""
        assert "alpha_star: 3.0" in result.metrics_report

    def test_retries_on_empty_response(self) -> None:
        """Empty LLM response triggers retry."""
        results = _load_results()
        backend = MockBackend(
            [
                "",  # empty
                "The topology shows uniform 50% degradation under BB node failure.",
                "Try adding cross-plane redundancy.",
            ]
        )
        result = run_analysis_loop(
            results=results,
            hypothesis="Test",
            backend=backend,
        )
        assert result.complete
        assert "50%" in result.interpretation

    def test_metrics_report_is_always_present(self) -> None:
        """Even on LLM failure, the metrics report is generated."""
        results = _load_results()
        backend = MockBackend(["", "", ""])  # all empty
        result = run_analysis_loop(
            results=results,
            hypothesis="Test",
            backend=backend,
            max_iterations=2,
        )
        assert not result.complete
        assert "alpha_star: 3.0" in result.metrics_report
