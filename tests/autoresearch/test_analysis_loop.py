"""Tests for the analysis loop."""

from __future__ import annotations

import json
from pathlib import Path

from netlab.autoresearch.analysis_loop import (
    _build_results_summary,
    _parse_findings,
    run_analysis_loop,
)
from netlab.autoresearch.backend import MockBackend

MINI_DCBB_RESULTS = (
    Path(__file__).parent.parent
    / "data"
    / "mini_dcbb_output"
    / "mini_dcbb.results.json"
)


def _load_results() -> dict:
    with MINI_DCBB_RESULTS.open() as f:
        return json.load(f)


class TestParseFindings:
    def test_single_finding(self) -> None:
        response = (
            "CLAIM: Alpha star is 3.0\n"
            "EVIDENCE: steps.msd_baseline.data.alpha_star = 3.0\n"
            "DISPROOF: Alpha would differ if cross-site capacity changed\n"
        )
        findings = _parse_findings(response)
        assert len(findings) == 1
        assert findings[0]["claim"] == "Alpha star is 3.0"
        assert "alpha_star = 3.0" in findings[0]["evidence"]

    def test_multiple_findings(self) -> None:
        response = (
            "CLAIM: First claim\n"
            "EVIDENCE: steps.a.b = 1.0\n"
            "DISPROOF: X\n"
            "\n"
            "CLAIM: Second claim\n"
            "EVIDENCE: steps.c.d = 2.0\n"
            "DISPROOF: Y\n"
        )
        findings = _parse_findings(response)
        assert len(findings) == 2

    def test_multi_line_evidence(self) -> None:
        response = (
            "CLAIM: Multi-evidence\n"
            "EVIDENCE: steps.a.b = 1.0\n"
            "steps.c.d = 2.0\n"
            "DISPROOF: Z\n"
        )
        findings = _parse_findings(response)
        assert len(findings) == 1
        assert "steps.c.d = 2.0" in findings[0]["evidence"]

    def test_no_findings(self) -> None:
        response = "I think the results look interesting but I need more data."
        findings = _parse_findings(response)
        assert len(findings) == 0


class TestBuildResultsSummary:
    def test_includes_alpha(self) -> None:
        results = _load_results()
        summary = _build_results_summary(results)
        assert "alpha_star=3.0" in summary

    def test_includes_step_names(self) -> None:
        results = _load_results()
        summary = _build_results_summary(results)
        assert "tm_lh_path" in summary
        assert "tm_1x_bb" in summary


class TestAnalysisLoop:
    def test_completes_with_verified_findings(self) -> None:
        """LLM produces correctly cited findings on first try."""
        results = _load_results()
        response = (
            "CLAIM: Maximum demand multiplier is 3.0\n"
            "EVIDENCE: steps.msd_baseline.data.alpha_star = 3.0\n"
            "DISPROOF: Would differ if link capacities changed\n"
        )
        backend = MockBackend([response])
        result = run_analysis_loop(
            results=results,
            hypothesis="The topology supports 3x demand scaling",
            backend=backend,
        )
        assert result.complete
        assert len(result.findings) == 1
        assert result.findings[0].verification.all_verified

    def test_retries_on_mismatch(self) -> None:
        """LLM cites wrong number first, then corrects."""
        results = _load_results()
        bad_response = (
            "CLAIM: Alpha is 5.0\n"
            "EVIDENCE: steps.msd_baseline.data.alpha_star = 5.0\n"
            "DISPROOF: X\n"
        )
        good_response = (
            "CLAIM: Alpha is 3.0\n"
            "EVIDENCE: steps.msd_baseline.data.alpha_star = 3.0\n"
            "DISPROOF: X\n"
        )
        backend = MockBackend([bad_response, good_response])
        result = run_analysis_loop(
            results=results,
            hypothesis="Test",
            backend=backend,
        )
        assert result.complete
        assert result.iterations_used == 2

    def test_fails_on_budget_exhaustion(self) -> None:
        """LLM never produces valid findings."""
        results = _load_results()
        bad_response = (
            "CLAIM: Wrong\n"
            "EVIDENCE: steps.msd_baseline.data.alpha_star = 999.0\n"
            "DISPROOF: X\n"
        )
        backend = MockBackend([bad_response] * 5)
        result = run_analysis_loop(
            results=results,
            hypothesis="Test",
            backend=backend,
            max_iterations=3,
        )
        assert not result.complete
        assert result.iterations_used == 3

    def test_retries_on_bad_format(self) -> None:
        """LLM returns unstructured text first, then proper format."""
        results = _load_results()
        unstructured = "The results show interesting patterns in the BAC values."
        structured = (
            "CLAIM: Alpha is 3.0\n"
            "EVIDENCE: steps.msd_baseline.data.alpha_star = 3.0\n"
            "DISPROOF: Different capacity would change it\n"
        )
        backend = MockBackend([unstructured, structured])
        result = run_analysis_loop(
            results=results,
            hypothesis="Test",
            backend=backend,
        )
        assert result.complete
        assert result.iterations_used == 2
