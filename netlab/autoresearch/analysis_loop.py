"""Inner Loop 2: Results analysis and interpretation.

The metrics pipeline extracts verified numbers. The LLM interprets them.
Clear separation: facts are machine-generated, explanations are LLM-generated.

Flow:
  results JSON → metrics_report (verified numbers) → LLM interprets → findings
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import LLMBackend
from .metrics_report import build_metrics_report


@dataclass
class AnalysisResult:
    """Output of the analysis loop."""

    metrics_report: str  # machine-generated, verified
    interpretation: str  # LLM-generated explanation
    next_hypothesis: str  # LLM-generated suggestion for next experiment
    iterations_used: int = 0
    complete: bool = False

    def summary(self) -> str:
        lines = self.interpretation.strip().splitlines()
        n_lines = len(lines)
        preview = lines[0][:80] if lines else "(empty)"
        return f"{n_lines} lines, {self.iterations_used} iterations: {preview}..."


_ANALYSIS_SYSTEM_PROMPT = """\
You are a network reliability engineer analyzing simulation results.

You will receive a METRICS REPORT containing verified numbers from the
simulation. These numbers are machine-computed and correct — do not
question or re-derive them.

Your job: explain WHY the results look the way they do. Connect the
numbers to the topology structure. Identify what matters and what doesn't.

Be direct. No filler. Every sentence should convey an insight about the
topology's behavior under failure.
"""

_ANALYSIS_PROMPT = """\
Hypothesis being tested:
{hypothesis}

{metrics_report}

Explain these results. For each failure mode:
1. Why does BAC have this value? What structural property causes it?
2. Does latency degrade under failure, or just bandwidth?
3. Are both directions affected equally?

Then summarize: what are the key design strengths and weaknesses of this topology?
"""

_NEXT_HYPOTHESIS_PROMPT = """\
Based on this analysis:

{interpretation}

Original hypothesis:
{hypothesis}

Metrics summary:
{metrics_summary}

Propose the next topology experiment. Consider:
- What structural variation might improve resilience?
- What tradeoff hasn't been explored (cost vs redundancy, latency vs bandwidth)?
- What aspect of the results is surprising or unexplained?

Respond with a clear, actionable topology description that can be
directly used to generate an ngraph scenario. Be specific about
node counts, link capacities, and failure modes to test.
"""


def run_analysis_loop(
    results: dict,
    hypothesis: str,
    backend: LLMBackend,
    max_iterations: int = 3,
) -> AnalysisResult:
    """Analyze simulation results using verified metrics + LLM interpretation.

    1. Compute metrics programmatically (trustworthy)
    2. Ask LLM to interpret the metrics (where it adds value)
    3. Ask LLM to propose the next hypothesis (closes the outer loop)

    Args:
        results: ngraph simulation results dict.
        hypothesis: The hypothesis being tested.
        backend: LLM backend for interpretation.
        max_iterations: Max retries if LLM produces empty response.

    Returns:
        AnalysisResult with verified metrics, interpretation, and next hypothesis.
    """
    # Step 1: compute verified metrics (no LLM involved)
    metrics_report = build_metrics_report(results)

    # Step 2: ask LLM to interpret
    interpretation = ""
    for _attempt in range(max_iterations):
        prompt = _ANALYSIS_PROMPT.format(
            hypothesis=hypothesis,
            metrics_report=metrics_report,
        )
        response = backend.generate(prompt, system=_ANALYSIS_SYSTEM_PROMPT)
        interpretation = response.strip()
        if interpretation:
            break

    if not interpretation:
        return AnalysisResult(
            metrics_report=metrics_report,
            interpretation="(LLM produced no interpretation)",
            next_hypothesis="",
            iterations_used=max_iterations,
            complete=False,
        )

    # Step 3: ask LLM to propose next hypothesis
    # Use a brief metrics summary (first 20 lines) to avoid token bloat
    metrics_lines = metrics_report.splitlines()
    metrics_summary = "\n".join(metrics_lines[:20])

    next_prompt = _NEXT_HYPOTHESIS_PROMPT.format(
        interpretation=interpretation,
        hypothesis=hypothesis,
        metrics_summary=metrics_summary,
    )
    next_hypothesis = backend.generate(
        next_prompt, system=_ANALYSIS_SYSTEM_PROMPT
    ).strip()

    return AnalysisResult(
        metrics_report=metrics_report,
        interpretation=interpretation,
        next_hypothesis=next_hypothesis,
        iterations_used=1,
        complete=True,
    )
