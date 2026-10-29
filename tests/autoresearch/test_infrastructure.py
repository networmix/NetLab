"""Check the square-mesh fixture and research template."""

from __future__ import annotations

from pathlib import Path

from netlab.autoresearch.hypothesis import HypothesisTemplate, ParamDef
from netlab.results import (
    baseline_case,
    feasible_alpha,
    require_step,
    rows,
    sampled_cases,
)
from netlab.run_record import load_run_record


class TestSquareMeshResults:
    """Verify the pre-generated square_mesh results fixture."""

    def test_loads_successfully(self, square_mesh_results: dict) -> None:
        assert isinstance(square_mesh_results, dict)

    def test_has_top_level_keys(self, square_mesh_results):
        assert {"metadata", "units_report", "steps"} <= square_mesh_results.keys()
        assert isinstance(square_mesh_results["steps"], list)

    def test_expected_steps_and_types(self, square_mesh_results):
        for name, kind in (
            ("msd_baseline", "MaximumSupportedDemand"),
            ("tm_placement", "TrafficMatrixPlacement"),
            ("node_to_node_capacity_matrix", "MaxFlow"),
        ):
            assert require_step(square_mesh_results, name)["step_type"] == kind

    def test_msd_alpha(self, square_mesh_results):
        assert feasible_alpha(require_step(square_mesh_results, "msd_baseline")) == 1

    def test_placement_cases_and_rows(self, square_mesh_results):
        step = require_step(square_mesh_results, "tm_placement")
        assert sampled_cases(step)
        baseline = baseline_case(step)
        assert rows(baseline)
        first = rows(baseline)[0]
        assert {"sources", "destinations"} <= first["endpoints"].keys()
        assert {"reference_request_units", "reference_shortfall_units"} <= first.keys()
        assert "delivered_units" in first["result"]

    def test_maxflow_cases(self, square_mesh_results):
        assert sampled_cases(
            require_step(square_mesh_results, "node_to_node_capacity_matrix")
        )


class TestSeedAnalysis:
    """Verify metrics computed from the square-mesh simulation."""

    def test_base_demand_fields(self, square_mesh_results: dict) -> None:
        base_demands = require_step(square_mesh_results, "msd_baseline")["question"][
            "demands"
        ]
        assert len(base_demands) > 0
        first = base_demands[0]
        assert "source" in first
        assert "target" in first
        assert "physical_volume" in first

    def test_compute_alpha_star_works(self, square_mesh_results: dict) -> None:
        """compute_alpha_star extracts alpha_star and base_total_demand."""
        from netlab.metrics.msd import compute_alpha_star

        alpha = compute_alpha_star(square_mesh_results)
        assert alpha.alpha_star == 1.0
        assert alpha.base_total_demand == 12.0

    def test_analyze_one_seed(self, square_mesh_results, tmp_path, monkeypatch):
        from netlab.metrics.analysis import analyze_one_seed

        monkeypatch.delenv("NGRAPH_ENABLE_MAXFLOW", raising=False)
        alpha, bac, _, latency, _, _, _ = analyze_one_seed(
            square_mesh_results,
            tmp_path,
            False,
            run_record=load_run_record(
                Path(__file__).parent / "data/square_mesh_results.json",
                square_mesh_results,
            ),
        )
        assert alpha.alpha_star == 1.0
        assert bac.offered == 12.0
        assert len(bac.series) == 1001
        assert latency.baseline["p50"] == 1.0


class TestSampleTemplate:
    """Verify the sample hypothesis_template.yml fixture."""

    def test_template_path_exists(self, sample_template_path: Path) -> None:
        assert sample_template_path.exists()

    def test_template_has_four_params(
        self, sample_template: HypothesisTemplate
    ) -> None:
        params = sample_template.params
        assert len(params) == 4

    def test_template_param_names(self, sample_template: HypothesisTemplate) -> None:
        params = sample_template.params
        assert set(params.keys()) == {
            "link_capacity",
            "preset",
            "demand_volume",
            "seed",
        }

    def test_link_capacity_param(self, sample_template: HypothesisTemplate) -> None:
        p = sample_template.params["link_capacity"]
        assert isinstance(p, ParamDef)
        assert p.type == "int"
        assert p.range == (100.0, 1000.0)
        assert p.step == 100.0
        assert p.default == 400

    def test_flow_policy_param(self, sample_template: HypothesisTemplate) -> None:
        p = sample_template.params["preset"]
        assert isinstance(p, ParamDef)
        assert p.type == "enum"
        assert p.values == ["SHORTEST_PATHS_ECMP"]
        assert p.default == "SHORTEST_PATHS_ECMP"

    def test_demand_volume_param(self, sample_template: HypothesisTemplate) -> None:
        p = sample_template.params["demand_volume"]
        assert isinstance(p, ParamDef)
        assert p.type == "float"
        assert p.range == (1000.0, 100000.0)
        assert p.step == 1000.0
        assert p.default == 10000.0

    def test_seed_param(self, sample_template: HypothesisTemplate) -> None:
        p = sample_template.params["seed"]
        assert isinstance(p, ParamDef)
        assert p.type == "int"
        assert p.range == (1.0, 1000.0)
        assert p.step == 1.0
        assert p.default == 42

    def test_all_params_have_descriptions(
        self, sample_template: HypothesisTemplate
    ) -> None:
        for name, p in sample_template.params.items():
            assert p.description, f"Parameter {name} has no description"
