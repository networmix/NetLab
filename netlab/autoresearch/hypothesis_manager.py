"""Outer Loop: Hypothesis cycle with persistence.

Orchestrates the full research cycle:
  hypothesis → generate scenario → simulate → analyze → update knowledge

Per-cycle artifacts are stored in directories. Cross-cycle state
(cycle log, knowledge, dead ends) persists across sessions.
Only this loop needs persistence — inner loops are stateless.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from .analysis_loop import AnalysisResult, run_analysis_loop
from .backend import LLMBackend
from .generation_loop import GenerationResult, run_generation_loop


@dataclass
class HypothesisCycle:
    """Record of one complete research cycle."""

    cycle_id: int
    hypothesis: str
    hypothesis_hash: str
    status: str  # "generated" | "simulated" | "analyzed" | "failed"
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
    """Manages the outer research loop with persistent state.

    State layout::

        project_dir/
          cycles/
            001/
              hypothesis.yml
              scenario.yml
              findings.md
              status.yml
            002/
              ...
          cycle_log.jsonl
          knowledge.md
          dead_ends.jsonl
    """

    def __init__(
        self,
        project_dir: Path,
        backend: LLMBackend,
        ngraph_bin: str | None = None,
        simulation_timeout_s: int = 600,
    ) -> None:
        self._project_dir = project_dir
        self._backend = backend
        self._ngraph_bin = ngraph_bin or shutil.which("ngraph") or "ngraph"
        self._simulation_timeout_s = simulation_timeout_s

        # Ensure directories exist
        self._cycles_dir = project_dir / "cycles"
        self._cycles_dir.mkdir(parents=True, exist_ok=True)

        # Cross-cycle state files
        self._log_path = project_dir / "cycle_log.jsonl"
        self._knowledge_path = project_dir / "knowledge.md"
        self._dead_ends_path = project_dir / "dead_ends.jsonl"

    def _next_cycle_id(self) -> int:
        """Determine next cycle ID from existing directories."""
        existing = [
            int(d.name)
            for d in self._cycles_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
        return max(existing, default=0) + 1

    def _load_dead_end_hashes(self) -> set[str]:
        """Load hypothesis hashes from dead ends log."""
        hashes: set[str] = set()
        if self._dead_ends_path.exists():
            for line in self._dead_ends_path.read_text().splitlines():
                if line.strip():
                    try:
                        entry = json.loads(line)
                        hashes.add(entry.get("hypothesis_hash", ""))
                    except json.JSONDecodeError:
                        continue
        return hashes

    def _load_knowledge(self) -> str:
        """Load current knowledge document."""
        if self._knowledge_path.exists():
            return self._knowledge_path.read_text()
        return ""

    def _save_knowledge(self, content: str) -> None:
        """Save updated knowledge document (atomic write)."""
        tmp = self._knowledge_path.with_suffix(".md.tmp")
        tmp.write_text(content)
        tmp.replace(self._knowledge_path)

    def _append_log(self, entry: CycleLogEntry) -> None:
        """Append to cycle log (atomic)."""
        line = json.dumps(asdict(entry)) + "\n"
        with self._log_path.open("a") as f:
            f.write(line)

    def _append_dead_end(
        self, hypothesis_hash: str, reason: str, cycle_id: int
    ) -> None:
        """Record a dead end."""
        entry = {
            "hypothesis_hash": hypothesis_hash,
            "reason": reason,
            "cycle_id": cycle_id,
            "timestamp": _now_iso(),
        }
        with self._dead_ends_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def run_cycle(self, hypothesis: str) -> HypothesisCycle:
        """Run one complete research cycle for a hypothesis.

        Steps:
          1. Generate and validate scenario (Inner Loop 1)
          2. Run ngraph simulation
          3. Analyze results (Inner Loop 2)
          4. Update knowledge

        Args:
            hypothesis: Natural language description of the connectivity
                idea to test.

        Returns:
            HypothesisCycle with full results.
        """
        t0 = time.time()
        cycle_id = self._next_cycle_id()
        h_hash = _hypothesis_hash(hypothesis)
        cycle_dir = self._cycles_dir / f"{cycle_id:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # Check for dead end
        if h_hash in self._load_dead_end_hashes():
            cycle = HypothesisCycle(
                cycle_id=cycle_id,
                hypothesis=hypothesis,
                hypothesis_hash=h_hash,
                status="skipped",
                error="Hypothesis previously identified as dead end",
                timestamp=_now_iso(),
            )
            self._append_log(
                CycleLogEntry(
                    cycle_id=cycle_id,
                    hypothesis_hash=h_hash,
                    status="skipped",
                    error=cycle.error,
                    timestamp=_now_iso(),
                )
            )
            return cycle

        # Save hypothesis
        (cycle_dir / "hypothesis.yml").write_text(
            yaml.dump(
                {"hypothesis": hypothesis, "hash": h_hash}, default_flow_style=False
            )
        )

        # Step 1: Generate scenario
        gen_result = run_generation_loop(
            idea=hypothesis,
            backend=self._backend,
            ngraph_bin=self._ngraph_bin,
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
            self._append_dead_end(
                h_hash, f"Generation failed: {gen_result.error}", cycle_id
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

        # Persist validated scenario
        scenario_path = cycle_dir / "scenario.yml"
        if gen_result.scenario_path and gen_result.scenario_path != scenario_path:
            shutil.copy2(gen_result.scenario_path, scenario_path)

        # Step 2: Simulate
        results_dir = cycle_dir / "results"
        results_dir.mkdir(exist_ok=True)
        try:
            sim_result = subprocess.run(
                [self._ngraph_bin, "run", str(scenario_path), "-o", str(results_dir)],
                capture_output=True,
                text=True,
                timeout=self._simulation_timeout_s,
            )
            if sim_result.returncode != 0:
                error_msg = f"ngraph run failed: {sim_result.stderr[-300:]}"
                cycle = HypothesisCycle(
                    cycle_id=cycle_id,
                    hypothesis=hypothesis,
                    hypothesis_hash=h_hash,
                    status="simulation_failed",
                    generation=gen_result,
                    error=error_msg,
                    duration_s=round(time.time() - t0, 1),
                    timestamp=_now_iso(),
                )
                self._append_log(
                    CycleLogEntry(
                        cycle_id=cycle_id,
                        hypothesis_hash=h_hash,
                        status="simulation_failed",
                        error=error_msg,
                        generation_iterations=gen_result.iterations_used,
                        timestamp=_now_iso(),
                        duration_s=cycle.duration_s,
                    )
                )
                return cycle
        except subprocess.TimeoutExpired:
            cycle = HypothesisCycle(
                cycle_id=cycle_id,
                hypothesis=hypothesis,
                hypothesis_hash=h_hash,
                status="simulation_timeout",
                generation=gen_result,
                error=f"Simulation timed out after {self._simulation_timeout_s}s",
                duration_s=round(time.time() - t0, 1),
                timestamp=_now_iso(),
            )
            self._append_log(
                CycleLogEntry(
                    cycle_id=cycle_id,
                    hypothesis_hash=h_hash,
                    status="simulation_timeout",
                    generation_iterations=gen_result.iterations_used,
                    timestamp=_now_iso(),
                    duration_s=cycle.duration_s,
                )
            )
            return cycle

        # Load results
        results_files = list(results_dir.glob("*.results.json"))
        if not results_files:
            error_msg = "No results.json produced by ngraph"
            cycle = HypothesisCycle(
                cycle_id=cycle_id,
                hypothesis=hypothesis,
                hypothesis_hash=h_hash,
                status="simulation_failed",
                generation=gen_result,
                error=error_msg,
                duration_s=round(time.time() - t0, 1),
                timestamp=_now_iso(),
            )
            self._append_log(
                CycleLogEntry(
                    cycle_id=cycle_id,
                    hypothesis_hash=h_hash,
                    status="simulation_failed",
                    error=error_msg,
                    timestamp=_now_iso(),
                    duration_s=cycle.duration_s,
                )
            )
            return cycle

        with results_files[0].open() as f:
            results_data: dict[str, Any] = json.load(f)

        # Step 3: Analyze
        analysis = run_analysis_loop(
            results=results_data,
            hypothesis=hypothesis,
            backend=self._backend,
        )

        # Save metrics report (machine-generated, verified)
        (cycle_dir / "metrics_report.md").write_text(analysis.metrics_report)

        # Save LLM interpretation
        (cycle_dir / "interpretation.md").write_text(analysis.interpretation)

        # Save next hypothesis suggestion
        if analysis.next_hypothesis:
            (cycle_dir / "next_hypothesis.md").write_text(analysis.next_hypothesis)

        # Save status
        status = "analyzed" if analysis.complete else "analysis_incomplete"
        (cycle_dir / "status.yml").write_text(
            yaml.dump(
                {
                    "status": status,
                    "analysis_iterations": analysis.iterations_used,
                    "hypothesis_hash": h_hash,
                },
                default_flow_style=False,
            )
        )

        cycle = HypothesisCycle(
            cycle_id=cycle_id,
            hypothesis=hypothesis,
            hypothesis_hash=h_hash,
            status=status,
            generation=gen_result,
            simulation_path=str(results_files[0]),
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
