"""Tests for the hypothesis manager (outer loop)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from netlab.autoresearch.backend import MockBackend
from netlab.autoresearch.hypothesis_manager import HypothesisManager


def _find_ngraph() -> str:
    import shutil

    path = shutil.which("ngraph")
    if path is None:
        venv_bin = Path(sys.executable).parent / "ngraph"
        if venv_bin.exists():
            path = str(venv_bin)
    if path is None:
        pytest.skip("ngraph binary not found")
    return path


# Valid scenario that passes generation loop
VALID_SCENARIO = """\
seed: 42
network:
  nodes:
    A: {}
    B: {}
  links:
    - source: A
      target: B
      capacity: 100
      cost: 1
demands:
  tm:
    - source: ^A$
      target: ^B$
      volume: 10
      mode: combine
      flow_policy: SHORTEST_PATHS_ECMP
workflow:
  - type: MaximumSupportedDemand
    name: msd_baseline
    demand_set: tm
    resolution: 0.1
"""

# MockBackend needs 3 responses:
# 1. Generation: scenario YAML
# 2. Analysis: interpretation
# 3. Analysis: next hypothesis
INTERPRETATION = "Alpha is 10.0 because cap=100 and demand=10. The topology is simple and fully connected."
NEXT_HYPOTHESIS = (
    "Try adding a second parallel link with different cost to see latency effects."
)


class TestHypothesisManager:
    def test_full_cycle(self, tmp_path: Path) -> None:
        """Run a complete hypothesis cycle: generate → simulate → analyze."""
        backend = MockBackend([VALID_SCENARIO, INTERPRETATION, NEXT_HYPOTHESIS])
        manager = HypothesisManager(
            project_dir=tmp_path,
            backend=backend,
            ngraph_bin=_find_ngraph(),
            simulation_timeout_s=60,
        )

        cycle = manager.run_cycle("Two nodes connected by a 100 Gbps link")

        assert cycle.status == "analyzed"
        assert cycle.generation is not None
        assert cycle.generation.success
        assert cycle.analysis is not None
        assert cycle.analysis.complete

        # Check persistence
        assert (tmp_path / "cycles" / "001" / "hypothesis.yml").exists()
        assert (tmp_path / "cycles" / "001" / "scenario.yml").exists()
        assert (tmp_path / "cycles" / "001" / "metrics_report.md").exists()
        assert (tmp_path / "cycles" / "001" / "interpretation.md").exists()
        assert (tmp_path / "cycles" / "001" / "next_hypothesis.md").exists()
        assert (tmp_path / "cycles" / "001" / "status.yml").exists()
        assert (tmp_path / "cycle_log.jsonl").exists()

        # Metrics report contains verified numbers
        report = (tmp_path / "cycles" / "001" / "metrics_report.md").read_text()
        assert "alpha_star" in report

        # Next hypothesis is persisted
        next_h = (tmp_path / "cycles" / "001" / "next_hypothesis.md").read_text()
        assert "parallel link" in next_h

    def test_generation_failure_records_dead_end(self, tmp_path: Path) -> None:
        """Failed generation marks hypothesis as dead end."""
        backend = MockBackend(["not valid yaml {{"] * 20)
        manager = HypothesisManager(
            project_dir=tmp_path,
            backend=backend,
            ngraph_bin=_find_ngraph(),
        )

        cycle = manager.run_cycle("Impossible topology idea")
        assert cycle.status == "generation_failed"
        assert (tmp_path / "dead_ends.jsonl").exists()

        # Second attempt with same hypothesis is skipped
        backend2 = MockBackend([VALID_SCENARIO, INTERPRETATION, NEXT_HYPOTHESIS])
        manager2 = HypothesisManager(
            project_dir=tmp_path,
            backend=backend2,
            ngraph_bin=_find_ngraph(),
        )
        cycle2 = manager2.run_cycle("Impossible topology idea")
        assert cycle2.status == "skipped"

    def test_cycle_id_increments(self, tmp_path: Path) -> None:
        """Each cycle gets a unique incrementing ID."""
        backend = MockBackend(
            [
                VALID_SCENARIO,
                INTERPRETATION,
                NEXT_HYPOTHESIS,
                VALID_SCENARIO,
                INTERPRETATION,
                NEXT_HYPOTHESIS,
            ]
        )
        manager = HypothesisManager(
            project_dir=tmp_path,
            backend=backend,
            ngraph_bin=_find_ngraph(),
        )

        c1 = manager.run_cycle("First hypothesis")
        c2 = manager.run_cycle("Second hypothesis")

        assert c1.cycle_id == 1
        assert c2.cycle_id == 2
        assert (tmp_path / "cycles" / "001").exists()
        assert (tmp_path / "cycles" / "002").exists()
