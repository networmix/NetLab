"""End-to-end execution test: generate scenario, write YAML, run ngraph.

Generates a DC-BB scenario with reduced parameters for CI speed,
writes it to YAML, runs ``ngraph inspect`` and ``ngraph run``, and verifies
the results contain the expected workflow steps and metrics.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from netlab.autoresearch.scenario_generator import generate_scenario
from netlab.results import feasible_alpha, require_step, sampled_cases
from netlab.simulation import inspect_scenario
from tests.autoresearch.dcbb_samples import SmallDcBbScenarioConfig


@pytest.mark.slow
class TestE2EExecution:
    """Generate a DC-BB scenario, run ngraph, verify results."""

    @pytest.fixture(scope="class")
    @classmethod
    def scenario_output(cls, tmp_path_factory):
        """Generate scenario, write YAML, run ngraph inspect + run.

        Returns (results_dict, results_path, run_stdout, run_stderr).
        """
        tmp_path = tmp_path_factory.mktemp("e2e")

        scenario = generate_scenario(SmallDcBbScenarioConfig())
        inspection = inspect_scenario(yaml.safe_dump(scenario))
        assert (inspection.node_count, inspection.link_count) == (19, 28)

        scenario_path = tmp_path / "scenario.yml"
        with open(scenario_path, "w") as f:
            yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)

        # Verify ngraph inspect passes first
        inspect_result = subprocess.run(
            [sys.executable, "-m", "ngraph", "inspect", str(scenario_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert inspect_result.returncode == 0, (
            f"ngraph inspect failed (rc={inspect_result.returncode}):\n"
            f"stderr: {inspect_result.stderr[-1000:]}"
        )

        # Run ngraph run
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        run_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ngraph",
                "run",
                str(scenario_path),
                "-o",
                str(results_dir),
            ],
            capture_output=True,
            text=True,
            timeout=540,  # 9 minutes
        )
        assert run_result.returncode == 0, (
            f"ngraph run failed (rc={run_result.returncode}):\n"
            f"stderr: {run_result.stderr[-1000:]}"
        )

        # Load results
        results_file = results_dir / "scenario.results.json"
        assert results_file.exists(), (
            f"Results file not found at {results_file}. "
            f"Contents of {results_dir}: {list(results_dir.iterdir())}"
        )

        with open(results_file) as f:
            results = json.load(f)

        return results, results_file, run_result.stdout, run_result.stderr

    @pytest.mark.timeout(600)
    def test_results_has_steps(self, scenario_output):
        results, *_ = scenario_output
        assert "steps" in results, (
            f"No 'steps' key in results. Keys: {list(results.keys())}"
        )

    @pytest.mark.timeout(600)
    def test_msd_baseline_present(self, scenario_output):
        results, *_ = scenario_output
        assert (
            require_step(results, "msd_baseline")["step_type"]
            == "MaximumSupportedDemand"
        )

    @pytest.mark.timeout(600)
    def test_alpha_star_positive(self, scenario_output):
        results, *_ = scenario_output
        assert feasible_alpha(require_step(results, "msd_baseline")) > 0

    @pytest.mark.timeout(600)
    def test_tm_combined_present(self, scenario_output):
        results, *_ = scenario_output
        assert (
            require_step(results, "tm_combined")["step_type"]
            == "TrafficMatrixPlacement"
        )

    @pytest.mark.timeout(600)
    def test_tm_combined_has_samples(self, scenario_output):
        results, *_ = scenario_output
        assert sampled_cases(require_step(results, "tm_combined"))

    @pytest.mark.timeout(600)
    def test_step_metadata(self, scenario_output):
        results, *_ = scenario_output
        for name in ("msd_baseline", "tm_combined"):
            step = require_step(results, name)
            assert step["scenario_seed"] == 42
            assert isinstance(step["step_seed"], int)
