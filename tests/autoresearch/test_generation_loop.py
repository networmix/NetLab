"""Tests for the scenario generation loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from netlab.autoresearch.backend import MockBackend
from netlab.autoresearch.generation_loop import (
    _extract_yaml,
    inspect_scenario,
    run_generation_loop,
)

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
      flow_policy: SHORTEST_PATHS_ECMP
workflow:
  - type: MaximumSupportedDemand
    name: msd_baseline
    demand_set: tm
    resolution: 0.1
"""

# Syntactically valid but structurally broken — ngraph accepts it
# but inspect shows 0 links (the target doesn't exist, link is dropped).
# The generation loop should reject this because links=0 is not viable.
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

# Actually invalid YAML that ngraph will reject
INVALID_YAML = "not: [valid: yaml: {{{"


def _find_ngraph() -> str:
    import shutil
    import sys

    # Check PATH first, then the current Python's venv bin directory
    path = shutil.which("ngraph")
    if path is None:
        venv_bin = Path(sys.executable).parent / "ngraph"
        if venv_bin.exists():
            path = str(venv_bin)
    if path is None:
        pytest.skip("ngraph binary not found")
    return path


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
        result = inspect_scenario(scenario_path, _find_ngraph())
        assert result.success
        assert result.node_count == 2
        assert result.link_count == 1

    def test_broken_scenario_passes_inspect_but_has_no_links(
        self, tmp_path: Path
    ) -> None:
        """ngraph silently drops links to nonexistent nodes."""
        scenario_path = tmp_path / "scenario.yml"
        scenario_path.write_text(BROKEN_SCENARIO)
        result = inspect_scenario(scenario_path, _find_ngraph())
        # inspect succeeds (valid YAML) but link count is 0
        assert result.success
        assert result.link_count == 0


class TestGenerationLoop:
    def test_succeeds_on_first_try(self, tmp_path: Path) -> None:
        """LLM produces valid YAML on the first attempt."""
        backend = MockBackend([VALID_SCENARIO])
        result = run_generation_loop(
            idea="Two nodes A and B connected by a single link",
            backend=backend,
            ngraph_bin=_find_ngraph(),
            work_dir=tmp_path,
        )
        assert result.success
        assert result.iterations_used == 1
        assert result.inspect is not None
        assert result.inspect.node_count == 2

    def test_succeeds_after_revision(self, tmp_path: Path) -> None:
        """Broken scenario (0 links) fails viability, then valid on second attempt."""
        backend = MockBackend([BROKEN_SCENARIO, VALID_SCENARIO])
        result = run_generation_loop(
            idea="Two connected nodes",
            backend=backend,
            ngraph_bin=_find_ngraph(),
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
            ngraph_bin=_find_ngraph(),
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
            ngraph_bin=_find_ngraph(),
            work_dir=tmp_path,
        )
        assert result.success
        assert result.iterations_used == 2
