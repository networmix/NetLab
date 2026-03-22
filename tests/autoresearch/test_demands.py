"""Tests for _build_demands, _build_workflow, and _build_failure_policy."""

from __future__ import annotations

import re

from netlab.autoresearch.scenario_generator import (
    DcBbScenarioConfig,
    _build_demands,
    _build_failure_policy,
    _build_workflow,
)

# ---------------------------------------------------------------------------
# _build_demands tests
# ---------------------------------------------------------------------------


class TestBuildDemands:
    """Tests for _build_demands."""

    def test_returns_baseline_traffic_matrix_key(self):
        config = DcBbScenarioConfig()
        demands = _build_demands(config)
        assert "baseline_traffic_matrix" in demands

    def test_two_demand_entries(self):
        config = DcBbScenarioConfig()
        entries = _build_demands(config)["baseline_traffic_matrix"]
        assert len(entries) == 2

    def test_volume_100t_per_direction(self):
        config = DcBbScenarioConfig()
        entries = _build_demands(config)["baseline_traffic_matrix"]
        for entry in entries:
            assert entry["volume"] == 100000.0, "Volume should be 100T (100,000 Gbps)"

    def test_combine_mode(self):
        config = DcBbScenarioConfig()
        entries = _build_demands(config)["baseline_traffic_matrix"]
        for entry in entries:
            assert entry["mode"] == "combine"

    def test_flow_policy_ecmp(self):
        config = DcBbScenarioConfig()
        entries = _build_demands(config)["baseline_traffic_matrix"]
        for entry in entries:
            assert entry["flow_policy"] == "SHORTEST_PATHS_ECMP"

    def test_abc1_to_xyz1_direction(self):
        """First entry: ABC1 RSW -> XYZ1 RSW."""
        config = DcBbScenarioConfig()
        entry = _build_demands(config)["baseline_traffic_matrix"][0]
        assert entry["source"] == "^abc1/pod.*/rsw$"
        assert entry["target"] == "^xyz1/mp1/rsw$"

    def test_xyz1_to_abc1_direction(self):
        """Second entry: XYZ1 RSW -> ABC1 RSW."""
        config = DcBbScenarioConfig()
        entry = _build_demands(config)["baseline_traffic_matrix"][1]
        assert entry["source"] == "^xyz1/mp1/rsw$"
        assert entry["target"] == "^abc1/pod.*/rsw$"

    def test_source_regex_matches_abc1_rsw(self):
        """Source regex for ABC1->XYZ1 direction should match all ABC1 RSW nodes."""
        config = DcBbScenarioConfig()
        pattern = _build_demands(config)["baseline_traffic_matrix"][0]["source"]
        compiled = re.compile(pattern)
        # Should match any pod RSW in ABC1
        assert compiled.match("abc1/pod1/rsw")
        assert compiled.match("abc1/pod96/rsw")
        assert compiled.match("abc1/pod42/rsw")
        # Should NOT match non-RSW nodes
        assert not compiled.match("abc1/pod1/fsw/plane1")
        assert not compiled.match("abc1/ssw/plane1/idx1")

    def test_target_regex_matches_xyz1_rsw(self):
        """Target regex for ABC1->XYZ1 direction should match XYZ1 RSW."""
        config = DcBbScenarioConfig()
        pattern = _build_demands(config)["baseline_traffic_matrix"][0]["target"]
        compiled = re.compile(pattern)
        assert compiled.match("xyz1/mp1/rsw")
        # Should NOT match XSW or other XYZ1 nodes
        assert not compiled.match("xyz1/xsw/plane1/dev1")

    def test_bidirectional_symmetry(self):
        """Source/target should be swapped between entries."""
        config = DcBbScenarioConfig()
        entries = _build_demands(config)["baseline_traffic_matrix"]
        assert entries[0]["source"] == entries[1]["target"]
        assert entries[0]["target"] == entries[1]["source"]

    def test_demand_entry_keys(self):
        """Each entry should have exactly the expected keys."""
        config = DcBbScenarioConfig()
        expected_keys = {"source", "target", "volume", "mode", "flow_policy"}
        for entry in _build_demands(config)["baseline_traffic_matrix"]:
            assert set(entry.keys()) == expected_keys


# ---------------------------------------------------------------------------
# _build_workflow tests
# ---------------------------------------------------------------------------


class TestBuildWorkflow:
    """Tests for _build_workflow."""

    def test_two_workflow_steps(self):
        config = DcBbScenarioConfig()
        workflow = _build_workflow(config)
        assert len(workflow) == 2

    def test_msd_step_type_and_name(self):
        config = DcBbScenarioConfig()
        msd = _build_workflow(config)[0]
        assert msd["type"] == "MaximumSupportedDemand"
        assert msd["name"] == "msd_baseline"

    def test_msd_references_baseline_traffic_matrix(self):
        config = DcBbScenarioConfig()
        msd = _build_workflow(config)[0]
        assert msd["demands"] == "baseline_traffic_matrix"

    def test_msd_uses_config_seed(self):
        config = DcBbScenarioConfig(seed=123)
        msd = _build_workflow(config)[0]
        assert msd["seed"] == 123

    def test_msd_uses_config_resolution(self):
        config = DcBbScenarioConfig(msd_resolution=0.05)
        msd = _build_workflow(config)[0]
        assert msd["resolution"] == 0.05

    def test_msd_flow_policy(self):
        config = DcBbScenarioConfig()
        msd = _build_workflow(config)[0]
        assert msd["flow_policy"] == "SHORTEST_PATHS_ECMP"

    def test_tmp_step_type_and_name(self):
        config = DcBbScenarioConfig()
        tmp = _build_workflow(config)[1]
        assert tmp["type"] == "TrafficMatrixPlacement"
        assert tmp["name"] == "tm_placement"

    def test_tmp_references_baseline_traffic_matrix(self):
        config = DcBbScenarioConfig()
        tmp = _build_workflow(config)[1]
        assert tmp["demands"] == "baseline_traffic_matrix"

    def test_tmp_references_failure_policy(self):
        config = DcBbScenarioConfig()
        tmp = _build_workflow(config)[1]
        assert tmp["failure_policy"] == "dc_bb_failures"

    def test_tmp_uses_config_iterations(self):
        config = DcBbScenarioConfig(failure_iterations=500)
        tmp = _build_workflow(config)[1]
        assert tmp["iterations"] == 500

    def test_tmp_parallelism(self):
        config = DcBbScenarioConfig()
        tmp = _build_workflow(config)[1]
        assert tmp["parallelism"] == 8

    def test_tmp_uses_config_seed(self):
        config = DcBbScenarioConfig(seed=99)
        tmp = _build_workflow(config)[1]
        assert tmp["seed"] == 99

    def test_tmp_metadata_baseline_true(self):
        config = DcBbScenarioConfig()
        tmp = _build_workflow(config)[1]
        assert tmp["metadata"] == {"baseline": True}

    def test_workflow_demand_names_match(self):
        """Both workflow steps should reference the same demand name
        that _build_demands produces."""
        config = DcBbScenarioConfig()
        demands = _build_demands(config)
        workflow = _build_workflow(config)
        demand_names = set(demands.keys())
        for step in workflow:
            assert step["demands"] in demand_names

    def test_default_config_values(self):
        """MSD and TMP steps use default config values correctly."""
        config = DcBbScenarioConfig()
        msd = _build_workflow(config)[0]
        tmp = _build_workflow(config)[1]
        assert msd["seed"] == 42
        assert msd["resolution"] == 0.01
        assert tmp["iterations"] == 200
        assert tmp["seed"] == 42


# ---------------------------------------------------------------------------
# _build_failure_policy tests
# ---------------------------------------------------------------------------


class TestBuildFailurePolicy:
    """Tests for _build_failure_policy."""

    def test_returns_dc_bb_failures_key(self):
        config = DcBbScenarioConfig()
        policy = _build_failure_policy(config)
        assert "dc_bb_failures" in policy

    def test_policy_has_description(self):
        config = DcBbScenarioConfig()
        fp = _build_failure_policy(config)["dc_bb_failures"]
        assert "attrs" in fp
        assert "description" in fp["attrs"]
        assert len(fp["attrs"]["description"]) > 0

    def test_seven_modes(self):
        config = DcBbScenarioConfig()
        modes = _build_failure_policy(config)["dc_bb_failures"]["modes"]
        assert len(modes) == 7

    def test_weights_sum_to_one(self):
        config = DcBbScenarioConfig()
        modes = _build_failure_policy(config)["dc_bb_failures"]["modes"]
        total = sum(m["weight"] for m in modes)
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_each_mode_has_rules(self):
        config = DcBbScenarioConfig()
        modes = _build_failure_policy(config)["dc_bb_failures"]["modes"]
        for i, mode in enumerate(modes):
            assert "rules" in mode, f"Mode {i} missing 'rules'"
            assert len(mode["rules"]) >= 1, f"Mode {i} has no rules"

    def test_mode_1_long_haul_path(self):
        config = DcBbScenarioConfig()
        mode = _build_failure_policy(config)["dc_bb_failures"]["modes"][0]
        assert mode["weight"] == 0.10
        rule = mode["rules"][0]
        assert rule["scope"] == "risk_group"
        assert rule["mode"] == "choice"
        assert rule["count"] == 1
        assert rule["conditions"][0]["value"] == "long_haul_path"

    def test_mode_2_plane_group(self):
        config = DcBbScenarioConfig()
        mode = _build_failure_policy(config)["dc_bb_failures"]["modes"][1]
        assert mode["weight"] == 0.15
        rule = mode["rules"][0]
        assert rule["scope"] == "risk_group"
        assert rule["conditions"][0]["value"] == "plane_group"

    def test_mode_3_plane_site(self):
        config = DcBbScenarioConfig()
        mode = _build_failure_policy(config)["dc_bb_failures"]["modes"][2]
        assert mode["weight"] == 0.15
        rule = mode["rules"][0]
        assert rule["scope"] == "risk_group"
        assert rule["conditions"][0]["value"] == "plane_site"

    def test_mode_4_device_index_across_planes(self):
        config = DcBbScenarioConfig()
        mode = _build_failure_policy(config)["dc_bb_failures"]["modes"][3]
        assert mode["weight"] == 0.10
        rule = mode["rules"][0]
        assert rule["scope"] == "risk_group"
        assert rule["conditions"][0]["value"] == "device_index_across_planes"

    def test_mode_5_single_bb_device(self):
        config = DcBbScenarioConfig()
        mode = _build_failure_policy(config)["dc_bb_failures"]["modes"][4]
        assert mode["weight"] == 0.15
        rule = mode["rules"][0]
        assert rule["scope"] == "node"
        assert rule["mode"] == "choice"
        assert rule["count"] == 1
        assert rule["conditions"][0]["value"] == "bb"

    def test_mode_6_two_bb_devices(self):
        config = DcBbScenarioConfig()
        mode = _build_failure_policy(config)["dc_bb_failures"]["modes"][5]
        assert mode["weight"] == 0.10
        rule = mode["rules"][0]
        assert rule["scope"] == "node"
        assert rule["mode"] == "choice"
        assert rule["count"] == 2

    def test_mode_7_random_link_failures(self):
        config = DcBbScenarioConfig()
        mode = _build_failure_policy(config)["dc_bb_failures"]["modes"][6]
        assert mode["weight"] == 0.25
        rule = mode["rules"][0]
        assert rule["scope"] == "link"
        assert rule["mode"] == "random"
        assert rule["probability"] == 0.01
        assert rule["conditions"][0]["value"] == "bb_cross_site"

    def test_failure_policy_name_matches_workflow_reference(self):
        """The policy name must match what _build_workflow references."""
        config = DcBbScenarioConfig()
        policy = _build_failure_policy(config)
        workflow = _build_workflow(config)
        tmp_step = workflow[1]
        assert tmp_step["failure_policy"] in policy
