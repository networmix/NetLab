"""Citation verifier for LLM analysis claims.

Extracts numeric claims from LLM text (e.g., "BAC AUC is 0.5455")
and verifies them against actual values in the results data.
Prevents hallucinated numbers from entering the knowledge base.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Claim:
    """A numeric claim extracted from LLM text."""

    text: str  # original text fragment containing the claim
    path: str  # dot-path into results (e.g., "steps.msd_baseline.data.alpha_star")
    claimed_value: float
    actual_value: float | None = None
    verified: bool | None = None  # None = not checked, True = matches, False = mismatch
    tolerance: float = 1e-4

    @property
    def status(self) -> str:
        if self.verified is None:
            return "unchecked"
        return "verified" if self.verified else "MISMATCH"


@dataclass
class VerificationResult:
    """Result of verifying all claims in a text."""

    claims: list[Claim] = field(default_factory=list)

    @property
    def all_verified(self) -> bool:
        return all(c.verified is True for c in self.claims) and len(self.claims) > 0

    @property
    def mismatches(self) -> list[Claim]:
        return [c for c in self.claims if c.verified is False]

    def summary(self) -> str:
        n = len(self.claims)
        ok = sum(1 for c in self.claims if c.verified is True)
        bad = sum(1 for c in self.claims if c.verified is False)
        unk = sum(1 for c in self.claims if c.verified is None)
        return f"{ok}/{n} verified, {bad} mismatches, {unk} unchecked"


def resolve_path(data: dict, path: str) -> Any:
    """Navigate a dot-separated path into a nested dict.

    Supports dict keys and integer list indices.
    Returns None if any segment is missing.

    Examples::

        resolve_path(data, "steps.msd_baseline.data.alpha_star")
        resolve_path(data, "steps.tm_lh_path.data.flow_results.0.summary.total_placed")
    """
    current: Any = data
    for segment in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(segment)
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def verify_claim(claim: Claim, data: dict) -> Claim:
    """Verify a single claim against the data.

    Sets claim.actual_value and claim.verified.
    """
    actual = resolve_path(data, claim.path)
    if actual is None:
        claim.actual_value = None
        claim.verified = None  # path not found — can't verify
        return claim

    try:
        actual_float = float(actual)
    except (TypeError, ValueError):
        claim.actual_value = None
        claim.verified = None
        return claim

    claim.actual_value = actual_float
    claim.verified = abs(actual_float - claim.claimed_value) <= claim.tolerance
    return claim


def verify_claims(claims: list[Claim], data: dict) -> VerificationResult:
    """Verify a list of claims against the results data."""
    for claim in claims:
        verify_claim(claim, data)
    return VerificationResult(claims=claims)


# --- Claim extraction from structured LLM output ---

# Pattern: "path = value" or "path: value" in LLM-structured output.
# Handles optional backtick quoting (e.g., `steps.msd.data.alpha_star` = 3.0).
_CLAIM_PATTERN = re.compile(
    r"`?(?P<path>[\w.]+(?:\.[\w.]+)+)`?\s*[=:]\s*(?P<value>-?[\d.]+)"
)


def extract_claims_from_text(text: str) -> list[Claim]:
    """Extract numeric claims from LLM text.

    Looks for patterns like:
      steps.msd_baseline.data.alpha_star = 3.0
      steps.tm_lh_path.data.baseline.summary.total_placed: 600.0

    Returns list of Claim objects with path and claimed_value set.
    """
    claims: list[Claim] = []
    for match in _CLAIM_PATTERN.finditer(text):
        path = match.group("path")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        # Only accept paths that look like results paths (start with "steps." or similar)
        if "." in path:
            claims.append(
                Claim(
                    text=match.group(0),
                    path=path,
                    claimed_value=value,
                )
            )
    return claims
