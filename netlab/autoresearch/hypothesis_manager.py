"""Run hypothesis cycles and persist scenarios, metrics, interpretations, and knowledge."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import yaml

from netlab.artifacts import write_text_atomic

from .analysis_loop import AnalysisResult, run_analysis_loop
from .backend import LLMBackend
from .generation_loop import GenerationResult, run_generation_loop


@dataclass
class HypothesisCycle:
    """Record of one complete research cycle."""

    cycle_id: int
    hypothesis: str
    hypothesis_hash: str
    status: str
    generation: GenerationResult | None = None
    simulation_path: str | None = None
    analysis: AnalysisResult | None = None
    error: str | None = None
    duration_s: float = 0.0
    timestamp: str = ""


@dataclass
class CycleLogEntry:
    """Minimal entry for the append-only cycle log."""

    cycle_id: int
    hypothesis_hash: str
    status: str
    error: str | None = None
    generation_iterations: int = 0
    analysis_iterations: int = 0
    findings_count: int = 0
    duration_s: float = 0.0
    timestamp: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hypothesis_hash(text: str) -> str:
    return sha256(text.strip().encode()).hexdigest()[:16]


class HypothesisManager:
    """Run research cycles in ``cycles/<id>/`` and record them in cycle_log.jsonl.

    A generation failure permits another attempt at the same hypothesis.
    """

    def __init__(
        self,
        project_dir: Path,
        backend: LLMBackend,
    ) -> None:
        self._project_dir = project_dir
        self._backend = backend
        self._cycles_dir = project_dir / "cycles"
        self._cycles_dir.mkdir(parents=True, exist_ok=True)

        self._log_path = project_dir / "cycle_log.jsonl"

    def _next_cycle_id(self) -> int:
        """Determine next cycle ID from existing directories."""
        existing = [
            int(d.name)
            for d in self._cycles_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
        return max(existing, default=0) + 1

    def _append_log(self, entry: CycleLogEntry) -> None:
        """Append a cycle summary to the log."""
        line = json.dumps(asdict(entry)) + "\n"
        previous = self._log_path.read_text() if self._log_path.exists() else ""
        write_text_atomic(self._log_path, previous + line)

    def run_cycle(self, hypothesis: str) -> HypothesisCycle:
        """Generate and simulate a hypothesis, analyze the results, and save the cycle."""
        t0 = time.time()
        cycle_id = self._next_cycle_id()
        h_hash = _hypothesis_hash(hypothesis)
        cycle_dir = self._cycles_dir / f"{cycle_id:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        write_text_atomic(
            cycle_dir / "hypothesis.yml",
            yaml.dump(
                {"hypothesis": hypothesis, "hash": h_hash}, default_flow_style=False
            ),
        )

        gen_result = run_generation_loop(
            idea=hypothesis,
            backend=self._backend,
            work_dir=cycle_dir,
        )

        if not gen_result.success:
            cycle = HypothesisCycle(
                cycle_id=cycle_id,
                hypothesis=hypothesis,
                hypothesis_hash=h_hash,
                status="generation_failed",
                generation=gen_result,
                error=gen_result.error,
                duration_s=round(time.time() - t0, 1),
                timestamp=_now_iso(),
            )
            self._append_log(
                CycleLogEntry(
                    cycle_id=cycle_id,
                    hypothesis_hash=h_hash,
                    status="generation_failed",
                    error=gen_result.error,
                    generation_iterations=gen_result.iterations_used,
                    timestamp=_now_iso(),
                    duration_s=cycle.duration_s,
                )
            )
            return cycle

        scenario_path = cycle_dir / "scenario.yml"
        write_text_atomic(scenario_path, gen_result.scenario_yaml)

        results_data = gen_result.results_data
        assert results_data is not None

        analysis = run_analysis_loop(
            results=results_data,
            hypothesis=hypothesis,
            backend=self._backend,
        )

        write_text_atomic(cycle_dir / "metrics_report.md", analysis.metrics_report)

        write_text_atomic(cycle_dir / "interpretation.md", analysis.interpretation)

        if analysis.next_hypothesis:
            write_text_atomic(
                cycle_dir / "next_hypothesis.md", analysis.next_hypothesis
            )

        status = "analyzed" if analysis.complete else "analysis_incomplete"
        write_text_atomic(
            cycle_dir / "status.yml",
            yaml.dump(
                {
                    "status": status,
                    "analysis_iterations": analysis.iterations_used,
                    "hypothesis_hash": h_hash,
                },
                default_flow_style=False,
            ),
        )

        cycle = HypothesisCycle(
            cycle_id=cycle_id,
            hypothesis=hypothesis,
            hypothesis_hash=h_hash,
            status=status,
            generation=gen_result,
            simulation_path=str(gen_result.results_path)
            if gen_result.results_path
            else None,
            analysis=analysis,
            duration_s=round(time.time() - t0, 1),
            timestamp=_now_iso(),
        )

        self._append_log(
            CycleLogEntry(
                cycle_id=cycle_id,
                hypothesis_hash=h_hash,
                status=status,
                generation_iterations=gen_result.iterations_used,
                analysis_iterations=analysis.iterations_used,
                findings_count=0,
                timestamp=_now_iso(),
                duration_s=cycle.duration_s,
            )
        )

        return cycle
