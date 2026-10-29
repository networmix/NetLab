"""Tests for the scenario generation loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from netlab.autoresearch.backend import MockBackend
from netlab.autoresearch.generation_loop import (
    _extract_yaml,
    run_generation_loop,
)
from netlab.simulation import inspect_scenario

# A minimal valid scenario that ngraph will accept
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
      preset: SHORTEST_PATHS_ECMP
workflow:
  - type: MaximumSupportedDemand
    name: msd_baseline
    demand_set: tm
    alpha_tolerance: 0.1
"""

# Unknown link targets are refused during model loading.
BROKEN_SCENARIO = """\
seed: 42
network:
  nodes:
    A: {}
  links:
    - source: A
      target: NONEXISTENT
      capacity: 100
      cost: 1
"""

# Invalid YAML, rejected before model expansion.
INVALID_YAML = "not: [valid: yaml: {{{"


class TestExtractYaml:
    def test_fenced_block(self) -> None:
        response = "Here is the scenario:\n```yaml\nseed: 42\nnetwork:\n  nodes: {}\n```\nDone."
        assert _extract_yaml(response).startswith("seed: 42")

    def test_raw_yaml(self) -> None:
        response = "seed: 42\nnetwork:\n  nodes:\n    A: {}"
        assert _extract_yaml(response).startswith("seed: 42")

    def test_leading_text_stripped(self) -> None:
        response = "Sure, here you go:\n\nseed: 42\nnetwork:\n  nodes: {}"
        assert _extract_yaml(response).startswith("seed: 42")


class TestInspectScenario:
    def test_valid_scenario(self, tmp_path: Path) -> None:
        scenario_path = tmp_path / "scenario.yml"
        scenario_path.write_text(VALID_SCENARIO)
        result = inspect_scenario(scenario_path.read_text())
        assert result.node_count == 2
        assert result.link_count == 1

    def test_unknown_link_target_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match=r"network.links\[0\].*NONEXISTENT"):
            inspect_scenario(BROKEN_SCENARIO)
        assert inspect_scenario(VALID_SCENARIO).link_count == 1


class TestGenerationLoop:
    def test_succeeds_on_first_try(self, tmp_path: Path) -> None:
        """LLM produces valid YAML on the first attempt."""
        backend = MockBackend([VALID_SCENARIO])
        result = run_generation_loop(
            idea="Two nodes A and B connected by a single link",
            backend=backend,
            work_dir=tmp_path,
        )
        assert result.success
        assert result.iterations_used == 1
        assert result.inspect is not None
        assert result.inspect.node_count == 2

    def test_succeeds_after_revision(self, tmp_path: Path) -> None:
        """Unknown target is refused, then valid on second attempt."""
        backend = MockBackend([BROKEN_SCENARIO, VALID_SCENARIO])
        result = run_generation_loop(
            idea="Two connected nodes",
            backend=backend,
            work_dir=tmp_path,
        )
        assert result.success
        assert result.iterations_used == 2

    def test_fails_after_budget(self, tmp_path: Path) -> None:
        """LLM never produces viable scenario within budget."""
        backend = MockBackend([BROKEN_SCENARIO] * 5)
        result = run_generation_loop(
            idea="Two connected nodes",
            backend=backend,
            max_iterations=3,
            work_dir=tmp_path,
        )
        assert not result.success
        assert result.iterations_used == 3

    def test_handles_non_yaml_response(self, tmp_path: Path) -> None:
        """LLM returns garbage, then valid YAML."""
        backend = MockBackend(["I don't understand the question", VALID_SCENARIO])
        result = run_generation_loop(
            idea="Two connected nodes",
            backend=backend,
            work_dir=tmp_path,
        )
        assert result.success
        assert result.iterations_used == 2
