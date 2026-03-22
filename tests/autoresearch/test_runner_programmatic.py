"""Tests for AutoResearchRunner programmatic generation mode."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from netlab.autoresearch.backend import MockBackend
from netlab.autoresearch.experiment_log import ExperimentLog
from netlab.autoresearch.runner import AutoResearchRunner, RunConfig

# ---------------------------------------------------------------------------
# Helpers: create a programmatic-mode test project directory
# ---------------------------------------------------------------------------

HYPOTHESIS_TEMPLATE_YAML = textwrap.dedent("""\
    params:
      g_abc1:
        type: enum
        values: ["16", "32", "64"]
        default: "64"
        description: "Mesh group count for ABC1"
      g_xyz1:
        type: enum
        values: ["64", "128", "256"]
        default: "64"
        description: "Mesh group count for XYZ1"
""")

OBJECTIVE_YAML = textwrap.dedent("""\
    direction: maximize
    primary_metric: alpha_star
    metrics:
      alpha_star:
        path: "steps.msd_baseline.data.alpha_star"
""")

CONFIG_YAML = textwrap.dedent("""\
    generation_mode: programmatic
    generator_module: netlab.autoresearch.scenario_generator
    generator_function: generate_scenario
    config_class: netlab.autoresearch.scenario_generator.DcBbScenarioConfig
""")

PROGRAM_MD = "Optimize mesh group count for maximum throughput."


def make_programmatic_project(tmp_path: Path) -> Path:
    """Create a programmatic-mode test project directory."""
    proj = tmp_path / "project"
    proj.mkdir()
    (proj / "program.md").write_text(PROGRAM_MD)
    (proj / "hypothesis_template.yml").write_text(HYPOTHESIS_TEMPLATE_YAML)
    (proj / "objective.yml").write_text(OBJECTIVE_YAML)
    (proj / "config.yml").write_text(CONFIG_YAML)
    (proj / "results").mkdir()
    return proj


def _mock_response(g_abc1: int, g_xyz1: int) -> str:
    """Build a mock LLM response with the given G values."""
    return textwrap.dedent(f"""\
        Trying g_abc1={g_abc1}, g_xyz1={g_xyz1}.

        ```yaml
        params:
          g_abc1: {g_abc1}
          g_xyz1: {g_xyz1}
        ```
    """)


# ---------------------------------------------------------------------------
# Init validation tests
# ---------------------------------------------------------------------------


class TestProgrammaticInit:
    def test_init_loads_generator(self, tmp_path: Path) -> None:
        """Programmatic mode runner loads the generator function."""
        proj = make_programmatic_project(tmp_path)

        backend = MockBackend([])
        config = RunConfig(project_dir=proj, backend=backend, max_experiments=0)
        runner = AutoResearchRunner(config)

        assert runner._generation_mode == "programmatic"
        assert runner._generator_fn is not None
        assert runner._config_class_ref is not None
        assert runner._merger is None

    def test_no_config_yml_defaults_to_template(self, tmp_path: Path) -> None:
        """Without config.yml, runner defaults to template mode."""
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "program.md").write_text(PROGRAM_MD)
        (proj / "hypothesis_template.yml").write_text(
            textwrap.dedent("""\
                params:
                  link_capacity:
                    type: float
                    range: [0.5, 10.0]
                    default: 2.0
                    description: "Link capacity"
            """)
        )
        (proj / "objective.yml").write_text(OBJECTIVE_YAML)

        # Need a base_scenario.yml for template mode
        (proj / "base_scenario.yml").write_text(
            textwrap.dedent("""\
                seed: 42
                network:
                  nodes:
                    N1: {}
                    N2: {}
                  links:
                    - source: N1
                      target: N2
                      capacity: ${{link_capacity}}
                      cost: 1.0
                workflow:
                  - type: MaximumSupportedDemand
                    name: msd_baseline
                    demand_set: baseline
            """)
        )
        (proj / "results").mkdir()

        backend = MockBackend([])
        config = RunConfig(project_dir=proj, backend=backend, max_experiments=0)
        runner = AutoResearchRunner(config)

        assert runner._generation_mode == "template"
        assert runner._merger is not None
        assert runner._generator_fn is None

    def test_unknown_mode_raises(self, tmp_path: Path) -> None:
        """Unknown generation_mode raises ValueError."""
        proj = make_programmatic_project(tmp_path)
        (proj / "config.yml").write_text("generation_mode: bogus\n")

        backend = MockBackend([])
        config = RunConfig(project_dir=proj, backend=backend, max_experiments=0)

        with pytest.raises(ValueError, match="Unknown generation_mode"):
            AutoResearchRunner(config)

    def test_no_base_scenario_needed(self, tmp_path: Path) -> None:
        """Programmatic mode does not require base_scenario.yml."""
        proj = make_programmatic_project(tmp_path)
        # Verify base_scenario.yml does NOT exist
        assert not (proj / "base_scenario.yml").exists()

        backend = MockBackend([])
        config = RunConfig(project_dir=proj, backend=backend, max_experiments=0)
        runner = AutoResearchRunner(config)

        assert runner._generation_mode == "programmatic"


# ---------------------------------------------------------------------------
# Happy path: full loop with real ngraph (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(600)
class TestProgrammaticHappyPath:
    def test_generates_and_runs_scenario(self, tmp_path: Path) -> None:
        """Programmatic mode: generate a DC-BB scenario, run ngraph, get results.

        Uses default G values (64, 64) which are compatible with the
        default layout (16,4,16,4).
        """
        proj = make_programmatic_project(tmp_path)

        responses = [_mock_response(64, 64)]
        backend = MockBackend(responses)
        config = RunConfig(
            project_dir=proj, backend=backend, max_experiments=1, seed=42
        )
        runner = AutoResearchRunner(config)
        runner.run()

        assert runner.status == "completed"
        assert runner.ngraph_call_count == 1

        log = ExperimentLog(proj, direction="maximize")
        entries = log.load()
        assert len(entries) == 1
        assert entries[0].status == "success"
        assert entries[0].metrics is not None
        assert "alpha_star" in entries[0].metrics
        assert entries[0].objective_score is not None

        # Check experiment directory
        exp_dir = proj / "results" / "exp_001"
        assert exp_dir.exists()
        assert (exp_dir / "scenario.yml").exists()
        assert (exp_dir / "scenario.results.json").exists()

        # Verify the generated scenario has the expected structure
        scenario = yaml.safe_load((exp_dir / "scenario.yml").read_text())
        assert "network" in scenario
        assert "nodes" in scenario["network"]
        assert "links" in scenario["network"]
        assert "workflow" in scenario

        # Check best_hypothesis.yml
        best_path = proj / "best_hypothesis.yml"
        assert best_path.exists()
        best_data = yaml.safe_load(best_path.read_text())
        assert best_data["params"]["g_abc1"] == 64
        assert best_data["params"]["g_xyz1"] == 64


# ---------------------------------------------------------------------------
# Generation error: invalid G value produces generation_error
# ---------------------------------------------------------------------------


class TestProgrammaticGenerationError:
    def test_invalid_g_produces_generation_error(self, tmp_path: Path) -> None:
        """g_abc1=32 with default layout -> ValueError from generate_scenario
        -> logged as generation_error."""
        proj = make_programmatic_project(tmp_path)

        # g_abc1=32 is invalid with default layout_abc1=(16,4,16,4) because
        # 16*4=64 != 32
        responses = [_mock_response(32, 64)]
        backend = MockBackend(responses)
        config = RunConfig(
            project_dir=proj, backend=backend, max_experiments=1, seed=42
        )
        runner = AutoResearchRunner(config)
        runner.run()

        assert runner.status == "completed"
        assert runner.ngraph_call_count == 0  # never reached ngraph

        log = ExperimentLog(proj, direction="maximize")
        entries = log.load()
        assert len(entries) == 1
        assert entries[0].status == "generation_error"
        assert entries[0].error_detail is not None
        assert (
            "layout" in entries[0].error_detail.lower()
            or "valid" in entries[0].error_detail.lower()
        )


# ---------------------------------------------------------------------------
# Deduplication works in programmatic mode
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(600)
class TestProgrammaticDedup:
    def test_same_params_cached(self, tmp_path: Path) -> None:
        """Same G params twice -> second is cached, only 1 ngraph call."""
        proj = make_programmatic_project(tmp_path)

        responses = [
            _mock_response(64, 64),
            _mock_response(64, 64),  # duplicate
        ]
        backend = MockBackend(responses)
        config = RunConfig(
            project_dir=proj, backend=backend, max_experiments=2, seed=42
        )
        runner = AutoResearchRunner(config)
        runner.run()

        assert runner.status == "completed"
        assert runner.ngraph_call_count == 1

        log = ExperimentLog(proj, direction="maximize")
        entries = log.load()
        assert len(entries) == 2
        assert entries[0].status == "success"
        assert entries[1].status == "cached"
        assert entries[0].params_hash == entries[1].params_hash
        assert entries[1].metrics == entries[0].metrics


# ---------------------------------------------------------------------------
# Mixed success and generation error
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(600)
class TestProgrammaticMixed:
    def test_success_then_error_then_cached(self, tmp_path: Path) -> None:
        """First succeeds, second fails generation, third is a dup of first."""
        proj = make_programmatic_project(tmp_path)

        responses = [
            _mock_response(64, 64),  # valid -> success
            _mock_response(32, 64),  # invalid G -> generation_error
            _mock_response(64, 64),  # duplicate -> cached
        ]
        backend = MockBackend(responses)
        config = RunConfig(
            project_dir=proj, backend=backend, max_experiments=3, seed=42
        )
        runner = AutoResearchRunner(config)
        runner.run()

        assert runner.status == "completed"
        assert runner.ngraph_call_count == 1

        log = ExperimentLog(proj, direction="maximize")
        entries = log.load()
        assert len(entries) == 3
        assert entries[0].status == "success"
        assert entries[1].status == "generation_error"
        assert entries[2].status == "cached"
