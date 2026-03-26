"""End-to-end execution test: generate scenario, write YAML, run ngraph.

Step E-9: Generates a DC-BB scenario with reduced parameters for CI speed,
writes it to YAML, runs ``ngraph inspect`` and ``ngraph run``, and verifies
the results contain the expected workflow steps and metrics.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from netlab.autoresearch.scenario_generator import DcBbScenarioConfig, generate_scenario

NGRAPH_BIN = Path(__file__).resolve().parent.parent.parent / "venv" / "bin" / "ngraph"

pytestmark = pytest.mark.skipif(
    not NGRAPH_BIN.is_file(),
    reason=f"ngraph binary not found at {NGRAPH_BIN}",
)


@pytest.mark.slow
class TestE2EExecution:
    """Generate a DC-BB scenario, run ngraph, verify results."""

    @pytest.fixture(scope="class")
    def scenario_output(self, tmp_path_factory):
        """Generate scenario, write YAML, run ngraph inspect + run.

        Returns (results_dict, results_path, run_stdout, run_stderr).
        """
        tmp_path = tmp_path_factory.mktemp("e2e")

        # Reduced config: fewer failure iterations and coarser MSD resolution
        config = DcBbScenarioConfig(
            failure_iterations=10,
            msd_resolution=0.1,
        )
        scenario = generate_scenario(config)

        scenario_path = tmp_path / "scenario.yml"
        with open(scenario_path, "w") as f:
            yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)

        # Verify ngraph inspect passes first (fast sanity check)
        inspect_result = subprocess.run(
            [str(NGRAPH_BIN), "inspect", str(scenario_path)],
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
            [str(NGRAPH_BIN), "run", str(scenario_path), "-o", str(results_dir)],
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
        steps = results["steps"]
        assert "msd_baseline" in steps, (
            f"No msd_baseline in steps. Found: {list(steps.keys())}"
        )

    @pytest.mark.timeout(600)
    def test_alpha_star_positive(self, scenario_output):
        results, *_ = scenario_output
        msd_data = results["steps"]["msd_baseline"]["data"]
        alpha_star = msd_data.get("alpha_star")
        assert alpha_star is not None, "alpha_star missing from msd_baseline data"
        assert alpha_star > 0, f"alpha_star={alpha_star}, expected > 0"

    @pytest.mark.timeout(600)
    def test_tm_combined_present(self, scenario_output):
        results, *_ = scenario_output
        steps = results["steps"]
        assert "tm_combined" in steps, (
            f"No tm_combined in steps. Found: {list(steps.keys())}"
        )

    @pytest.mark.timeout(600)
    def test_tm_combined_has_flow_results(self, scenario_output):
        results, *_ = scenario_output
        tm_data = results["steps"]["tm_combined"]["data"]
        flow_results = tm_data.get("flow_results")
        assert flow_results is not None, "flow_results missing from tm_combined data"
        assert len(flow_results) > 0, "flow_results is empty"

    @pytest.mark.timeout(600)
    def test_workflow_metadata(self, scenario_output):
        results, *_ = scenario_output
        assert "workflow" in results, (
            f"No 'workflow' key in results. Keys: {list(results.keys())}"
        )
        workflow = results["workflow"]
        assert "msd_baseline" in workflow, (
            f"msd_baseline not in workflow metadata. Found: {list(workflow.keys())}"
        )
        assert "tm_combined" in workflow, (
            f"tm_combined not in workflow metadata. Found: {list(workflow.keys())}"
        )
