"""Tests for the citation verifier."""

from __future__ import annotations

from netlab.autoresearch.citation_verifier import (
    Claim,
    extract_claims_from_text,
    resolve_path,
    verify_claim,
    verify_claims,
)

SAMPLE_DATA = {
    "steps": {
        "msd_baseline": {"data": {"alpha_star": 3.0}},
        "tm_lh_path": {
            "data": {
                "baseline": {"summary": {"total_placed": 600.0, "overall_ratio": 1.0}},
                "flow_results": [
                    {"occurrence_count": 5, "summary": {"overall_ratio": 0.3333}},
                    {"occurrence_count": 5, "summary": {"overall_ratio": 0.6667}},
                ],
            }
        },
    }
}


class TestResolvePath:
    def test_simple_path(self) -> None:
        assert resolve_path(SAMPLE_DATA, "steps.msd_baseline.data.alpha_star") == 3.0

    def test_nested_path(self) -> None:
        assert (
            resolve_path(
                SAMPLE_DATA, "steps.tm_lh_path.data.baseline.summary.total_placed"
            )
            == 600.0
        )

    def test_list_index(self) -> None:
        assert (
            resolve_path(
                SAMPLE_DATA, "steps.tm_lh_path.data.flow_results.0.occurrence_count"
            )
            == 5
        )

    def test_missing_path(self) -> None:
        assert resolve_path(SAMPLE_DATA, "steps.nonexistent.data") is None

    def test_empty_path(self) -> None:
        assert resolve_path(SAMPLE_DATA, "") is None


class TestVerifyClaim:
    def test_correct_claim(self) -> None:
        claim = Claim(
            text="alpha = 3.0",
            path="steps.msd_baseline.data.alpha_star",
            claimed_value=3.0,
        )
        verify_claim(claim, SAMPLE_DATA)
        assert claim.verified is True
        assert claim.actual_value == 3.0

    def test_incorrect_claim(self) -> None:
        claim = Claim(
            text="alpha = 5.0",
            path="steps.msd_baseline.data.alpha_star",
            claimed_value=5.0,
        )
        verify_claim(claim, SAMPLE_DATA)
        assert claim.verified is False
        assert claim.actual_value == 3.0

    def test_missing_path(self) -> None:
        claim = Claim(text="x = 1.0", path="steps.nonexistent", claimed_value=1.0)
        verify_claim(claim, SAMPLE_DATA)
        assert claim.verified is None

    def test_tolerance(self) -> None:
        claim = Claim(
            text="ratio = 0.3333",
            path="steps.tm_lh_path.data.flow_results.0.summary.overall_ratio",
            claimed_value=0.3333,
            tolerance=0.001,
        )
        verify_claim(claim, SAMPLE_DATA)
        assert claim.verified is True


class TestVerifyClaims:
    def test_all_verified(self) -> None:
        claims = [
            Claim(
                text="a", path="steps.msd_baseline.data.alpha_star", claimed_value=3.0
            ),
            Claim(
                text="b",
                path="steps.tm_lh_path.data.baseline.summary.total_placed",
                claimed_value=600.0,
            ),
        ]
        result = verify_claims(claims, SAMPLE_DATA)
        assert result.all_verified
        assert len(result.mismatches) == 0

    def test_one_mismatch(self) -> None:
        claims = [
            Claim(
                text="a", path="steps.msd_baseline.data.alpha_star", claimed_value=3.0
            ),
            Claim(
                text="b", path="steps.msd_baseline.data.alpha_star", claimed_value=999.0
            ),
        ]
        result = verify_claims(claims, SAMPLE_DATA)
        assert not result.all_verified
        assert len(result.mismatches) == 1


class TestExtractClaims:
    def test_extract_from_evidence_text(self) -> None:
        text = "steps.msd_baseline.data.alpha_star = 3.0\nsteps.tm_lh_path.data.baseline.summary.total_placed = 600.0"
        claims = extract_claims_from_text(text)
        assert len(claims) == 2
        assert claims[0].path == "steps.msd_baseline.data.alpha_star"
        assert claims[0].claimed_value == 3.0

    def test_colon_separator(self) -> None:
        text = "steps.msd_baseline.data.alpha_star: 3.0"
        claims = extract_claims_from_text(text)
        assert len(claims) == 1

    def test_no_claims(self) -> None:
        text = "The results look good overall."
        claims = extract_claims_from_text(text)
        assert len(claims) == 0
