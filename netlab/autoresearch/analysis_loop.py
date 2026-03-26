"""Inner Loop 2: Results analysis and explanation.

Iterates: LLM forms claims about simulation results → extracts
specific numbers → citation verifier checks → adversarial questioning
→ revise until explanation is complete.

Stateless and in-memory. All state is passed through function arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backend import LLMBackend
from .citation_verifier import (
    VerificationResult,
    extract_claims_from_text,
    verify_claims,
)


@dataclass
class Finding:
    """A single verified finding from the analysis loop."""

    claim: str
    evidence: str  # specific numbers cited
    verification: VerificationResult
    adversarial_check: str  # what would disprove this


@dataclass
class AnalysisResult:
    """Output of the analysis loop."""

    findings: list[Finding] = field(default_factory=list)
    iterations_used: int = 0
    complete: bool = False

    def summary(self) -> str:
        n = len(self.findings)
        verified = sum(1 for f in self.findings if f.verification.all_verified)
        return f"{n} findings ({verified} fully verified), {self.iterations_used} iterations"


_ANALYSIS_SYSTEM_PROMPT = """\
You are analyzing network simulation results. Your job is to explain
WHY the results look the way they do, grounded in specific numbers.

Rules:
1. Every claim must cite specific values from the data using dot-path
   notation: steps.step_name.data.field.subfield = value
2. After each claim, state what would disprove it.
3. Be precise. "BAC drops significantly" is not acceptable.
   "steps.tm_lh_path.data.flow_results.0.summary.overall_ratio = 0.3333" is.
"""

_ANALYSIS_PROMPT_TEMPLATE = """\
Hypothesis being tested:
{hypothesis}

Simulation results summary:
{results_summary}

{feedback}

Provide your analysis as a series of findings. For each finding:
1. State the claim
2. Cite specific values using dot-path notation (path = value)
3. State what would disprove this claim

Format each finding as:

CLAIM: <your claim>
EVIDENCE: <dot.path = value, one per line>
DISPROOF: <what would disprove this>
"""

_ADVERSARIAL_PROMPT = """\
You previously found:
{finding_text}

The citations were verified against the actual data:
{verification_summary}

Now critically question this finding:
1. Is there a simpler explanation?
2. Could this be a coincidence or artifact?
3. What additional evidence would strengthen or weaken this claim?

If the finding stands, respond with "CONFIRMED".
If it needs revision, provide the revised finding in the same format.
"""


def _parse_findings(response: str) -> list[dict[str, str]]:
    """Parse LLM response into finding dicts."""
    findings: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("CLAIM:"):
            if current.get("claim"):
                findings.append(current)
            current = {"claim": line[6:].strip()}
        elif line.startswith("EVIDENCE:"):
            current["evidence"] = line[9:].strip()
        elif line.startswith("DISPROOF:"):
            current["disproof"] = line[9:].strip()
        elif current.get("evidence") is not None and "=" in line and "." in line:
            # Continuation of evidence lines
            current["evidence"] += "\n" + line

    if current.get("claim"):
        findings.append(current)

    return findings


def _build_results_summary(results: dict) -> str:
    """Build a concise summary of simulation results for the LLM."""
    lines: list[str] = []
    steps = results.get("steps", {})

    for step_name, step_data in steps.items():
        data = step_data.get("data", {})

        baseline = data.get("baseline", {})
        flow_results = data.get("flow_results", [])

        if baseline and isinstance(baseline, dict):
            summary = baseline.get("summary", {})
            lines.append(
                f"{step_name}: baseline ratio={summary.get('overall_ratio', 'N/A')}, "
                f"placed={summary.get('total_placed', 'N/A')}, "
                f"demand={summary.get('total_demand', 'N/A')}"
            )

        if flow_results:
            n_patterns = len(flow_results)
            total_iters = sum(fr.get("occurrence_count", 1) for fr in flow_results)
            lines.append(
                f"  {n_patterns} unique failure patterns, {total_iters} iterations"
            )
            for fr in flow_results[:3]:
                s = fr.get("summary", {})
                lines.append(
                    f"  pattern (count={fr.get('occurrence_count', 1)}): "
                    f"ratio={s.get('overall_ratio', 'N/A')}, "
                    f"placed={s.get('total_placed', 'N/A')}"
                )

        # MSD alpha
        alpha = data.get("alpha_star")
        if alpha is not None:
            lines.append(f"{step_name}: alpha_star={alpha}")

    return "\n".join(lines)


def run_analysis_loop(
    results: dict,
    hypothesis: str,
    backend: LLMBackend,
    max_iterations: int = 10,
) -> AnalysisResult:
    """Run the analysis loop on simulation results.

    Iterates: LLM forms claims → extract citations → verify → adversarial check
    until findings are complete or budget is exhausted.

    Args:
        results: ngraph simulation results dict.
        hypothesis: The hypothesis being tested (natural language).
        backend: LLM backend for analysis.
        max_iterations: Maximum analysis iterations.

    Returns:
        AnalysisResult with verified findings.
    """
    results_summary = _build_results_summary(results)
    all_findings: list[Finding] = []
    feedback = ""

    for iteration in range(max_iterations):
        # Ask LLM to analyze
        prompt = _ANALYSIS_PROMPT_TEMPLATE.format(
            hypothesis=hypothesis,
            results_summary=results_summary,
            feedback=feedback,
        )
        response = backend.generate(prompt, system=_ANALYSIS_SYSTEM_PROMPT)

        # Parse findings
        raw_findings = _parse_findings(response)
        if not raw_findings:
            feedback = "Your previous response did not contain any findings in the expected format. Please use CLAIM: / EVIDENCE: / DISPROOF: format."
            continue

        # Verify citations for each finding
        new_findings: list[Finding] = []
        for raw in raw_findings:
            evidence_text = raw.get("evidence", "")
            claims = extract_claims_from_text(evidence_text)
            verification = verify_claims(claims, results)

            finding = Finding(
                claim=raw.get("claim", ""),
                evidence=evidence_text,
                verification=verification,
                adversarial_check=raw.get("disproof", ""),
            )
            new_findings.append(finding)

        all_findings.extend(new_findings)

        # Check if any claims failed verification
        mismatches = [f for f in new_findings if f.verification.mismatches]
        if mismatches:
            mismatch_details = []
            for f in mismatches:
                for c in f.verification.mismatches:
                    mismatch_details.append(
                        f"  {c.path}: claimed {c.claimed_value}, actual {c.actual_value}"
                    )
            feedback = (
                "Some of your cited values do not match the actual data:\n"
                + "\n".join(mismatch_details)
                + "\nPlease recheck and revise your analysis."
            )
            continue

        # All findings verified — analysis is complete
        return AnalysisResult(
            findings=all_findings,
            iterations_used=iteration + 1,
            complete=True,
        )

    # Budget exhausted
    return AnalysisResult(
        findings=all_findings,
        iterations_used=max_iterations,
        complete=False,
    )
