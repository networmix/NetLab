"""Tests verifying the autoresearch test infrastructure works correctly."""

from __future__ import annotations

from pathlib import Path

from netlab.autoresearch.hypothesis import HypothesisTemplate, ParamDef


class TestSquareMeshResults:
    """Verify the pre-generated square_mesh results fixture."""

    def test_loads_successfully(self, square_mesh_results: dict) -> None:
        assert isinstance(square_mesh_results, dict)

    def test_has_top_level_keys(self, square_mesh_results: dict) -> None:
        assert "workflow" in square_mesh_results
        assert "steps" in square_mesh_results
        assert "scenario" in square_mesh_results

    def test_has_expected_workflow_steps(self, square_mesh_results: dict) -> None:
        steps = square_mesh_results["steps"]
        assert "msd_baseline" in steps
        assert "tm_placement" in steps
        assert "node_to_node_capacity_matrix" in steps

    def test_msd_baseline_has_alpha_star(self, square_mesh_results: dict) -> None:
        msd_data = square_mesh_results["steps"]["msd_baseline"]["data"]
        assert "alpha_star" in msd_data
        alpha = msd_data["alpha_star"]
        assert isinstance(alpha, (int, float))
        assert alpha > 0

    def test_tm_placement_has_flow_results(self, square_mesh_results: dict) -> None:
        tm_data = square_mesh_results["steps"]["tm_placement"]["data"]
        assert "flow_results" in tm_data
        flow_results = tm_data["flow_results"]
        assert isinstance(flow_results, list)
        assert len(flow_results) > 0

    def test_tm_placement_has_baseline(self, square_mesh_results: dict) -> None:
        tm_data = square_mesh_results["steps"]["tm_placement"]["data"]
        assert "baseline" in tm_data
        baseline = tm_data["baseline"]
        assert isinstance(baseline, dict)
        assert "flows" in baseline
        assert len(baseline["flows"]) > 0

    def test_tm_placement_flows_have_expected_fields(
        self, square_mesh_results: dict
    ) -> None:
        tm_data = square_mesh_results["steps"]["tm_placement"]["data"]
        baseline_flows = tm_data["baseline"]["flows"]
        first_flow = baseline_flows[0]
        assert "source" in first_flow
        assert "destination" in first_flow
        assert "demand" in first_flow
        assert "placed" in first_flow
        assert "dropped" in first_flow

    def test_maxflow_has_flow_results(self, square_mesh_results: dict) -> None:
        mf_data = square_mesh_results["steps"]["node_to_node_capacity_matrix"]["data"]
        assert "flow_results" in mf_data
        flow_results = mf_data["flow_results"]
        assert isinstance(flow_results, list)
        assert len(flow_results) > 0

    def test_workflow_metadata_has_step_types(self, square_mesh_results: dict) -> None:
        workflow = square_mesh_results["workflow"]
        assert workflow["msd_baseline"]["step_type"] == "MaximumSupportedDemand"
        assert workflow["tm_placement"]["step_type"] == "TrafficMatrixPlacement"
        assert workflow["node_to_node_capacity_matrix"]["step_type"] == "MaxFlow"


class TestAnalyzeOneSeedCompatibility:
    """Document the compatibility status of analyze_one_seed with square_mesh results.

    The current ngraph output format differs from what analyze_one_seed expects:
    - base_demands: uses source/target/volume instead of source_path/sink_path/demand
    - metadata: lacks 'baseline: true' flag
    - flow_results: baseline is stored separately, not as flow_results[0] with
      failure_id=="baseline"

    These tests document the incompatibility rather than testing analyze_one_seed
    directly, since the plan notes this is only required "if feasible".
    """

    def test_base_demands_format_differs(self, square_mesh_results: dict) -> None:
        """Current ngraph uses source/target/volume, metrics_cmd expects source_path/sink_path/demand."""
        msd_data = square_mesh_results["steps"]["msd_baseline"]["data"]
        base_demands = msd_data.get("base_demands", [])
        assert len(base_demands) > 0
        first = base_demands[0]
        # Current format uses 'source'/'target'/'volume'
        assert "source" in first
        assert "target" in first
        assert "volume" in first
        # analyze_one_seed expects 'source_path'/'sink_path'/'demand'
        assert "source_path" not in first
        assert "sink_path" not in first

    def test_metadata_lacks_baseline_flag(self, square_mesh_results: dict) -> None:
        """Current ngraph does not set metadata.baseline = true."""
        tm_meta = square_mesh_results["steps"]["tm_placement"].get("metadata", {})
        assert "baseline" not in tm_meta or tm_meta.get("baseline") is not True

    def test_compute_alpha_star_works(self, square_mesh_results: dict) -> None:
        """compute_alpha_star extracts alpha_star and base_total_demand."""
        from metrics.msd import compute_alpha_star

        alpha = compute_alpha_star(square_mesh_results)
        assert alpha.alpha_star == 1.0
        # base_total_demand reads 'volume' field (ngraph's key) with 'demand' as fallback
        assert alpha.base_total_demand == 12.0


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
            "flow_policy",
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
        p = sample_template.params["flow_policy"]
        assert isinstance(p, ParamDef)
        assert p.type == "enum"
        assert p.values == ["SHORTEST_PATHS_ECMP", "SHORTEST_PATHS_UCMP"]
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
