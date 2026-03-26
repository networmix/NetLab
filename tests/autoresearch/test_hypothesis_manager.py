"""Tests for the hypothesis manager (outer loop)."""

from __future__ import annotations

import json
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


# Valid scenario that the generation loop will accept
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

# Analysis response with correct citation (cap=100, demand=10 → alpha=10)
ANALYSIS_RESPONSE = """\
CLAIM: Alpha star is 10.0 for this simple 2-node topology
EVIDENCE: steps.msd_baseline.data.alpha_star = 10.0
DISPROOF: Would differ if link capacity or demand volume changed
"""


class TestHypothesisManager:
    def test_full_cycle(self, tmp_path: Path) -> None:
        """Run a complete hypothesis cycle: generate → simulate → analyze."""
        # MockBackend: first call = generation, second = analysis
        backend = MockBackend([VALID_SCENARIO, ANALYSIS_RESPONSE])
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
        assert len(cycle.analysis.findings) > 0

        # Check persistence
        assert (tmp_path / "cycles" / "001" / "hypothesis.yml").exists()
        assert (tmp_path / "cycles" / "001" / "scenario.yml").exists()
        assert (tmp_path / "cycles" / "001" / "findings.md").exists()
        assert (tmp_path / "cycles" / "001" / "status.yml").exists()
        assert (tmp_path / "cycle_log.jsonl").exists()

        # Cycle log has one entry
        log_lines = (tmp_path / "cycle_log.jsonl").read_text().strip().splitlines()
        assert len(log_lines) == 1
        entry = json.loads(log_lines[0])
        assert entry["cycle_id"] == 1
        assert entry["status"] == "analyzed"

    def test_generation_failure_records_dead_end(self, tmp_path: Path) -> None:
        """Failed generation marks hypothesis as dead end."""
        # Return garbage that won't pass inspect
        backend = MockBackend(["not valid yaml {{"] * 20)
        manager = HypothesisManager(
            project_dir=tmp_path,
            backend=backend,
            ngraph_bin=_find_ngraph(),
        )

        cycle = manager.run_cycle("Impossible topology idea")

        assert cycle.status == "generation_failed"
        assert (tmp_path / "dead_ends.jsonl").exists()

        # Second attempt with same hypothesis should be skipped
        backend2 = MockBackend([VALID_SCENARIO, ANALYSIS_RESPONSE])
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
                ANALYSIS_RESPONSE,
                VALID_SCENARIO,
                ANALYSIS_RESPONSE,
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
