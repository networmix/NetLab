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
    """Tests for _build_workflow (MSD + 7 per-mode TMP + 1 combined TMP)."""

    def test_workflow_step_count(self):
        from netlab.autoresearch.scenario_generator import FAILURE_MODE_NAMES

        config = DcBbScenarioConfig()
        workflow = _build_workflow(config)
        assert (
            len(workflow) == 1 + len(FAILURE_MODE_NAMES) + 1
        )  # MSD + N per-mode + combined

    def test_msd_step_type_and_name(self):
        config = DcBbScenarioConfig()
        msd = _build_workflow(config)[0]
        assert msd["type"] == "MaximumSupportedDemand"
        assert msd["name"] == "msd_baseline"

    def test_msd_references_baseline_traffic_matrix(self):
        config = DcBbScenarioConfig()
        msd = _build_workflow(config)[0]
        assert msd["demand_set"] == "baseline_traffic_matrix"

    def test_per_mode_tmp_steps(self):
        from netlab.autoresearch.scenario_generator import FAILURE_MODE_NAMES

        config = DcBbScenarioConfig()
        workflow = _build_workflow(config)
        tmp_steps = [s for s in workflow if s["type"] == "TrafficMatrixPlacement"]
        assert len(tmp_steps) == len(FAILURE_MODE_NAMES) + 1  # N per-mode + 1 combined
        for mode_name in FAILURE_MODE_NAMES:
            matching = [s for s in tmp_steps if s["name"] == f"tm_{mode_name}"]
            assert len(matching) == 1, f"Missing TMP step for {mode_name}"
            assert matching[0]["failure_policy"] == f"fm_{mode_name}"

    def test_combined_tmp_step(self):
        config = DcBbScenarioConfig()
        workflow = _build_workflow(config)
        combined = [s for s in workflow if s["name"] == "tm_combined"]
        assert len(combined) == 1
        assert combined[0]["failure_policy"] == "fm_combined"

    def test_all_tmp_reference_msd(self):
        config = DcBbScenarioConfig()
        workflow = _build_workflow(config)
        for step in workflow:
            if step["type"] == "TrafficMatrixPlacement":
                assert step["alpha_from_step"] == "msd_baseline"

    def test_workflow_demand_names_match(self):
        config = DcBbScenarioConfig()
        demands = _build_demands(config)
        workflow = _build_workflow(config)
        demand_names = set(demands.keys())
        for step in workflow:
            assert step["demand_set"] in demand_names


# ---------------------------------------------------------------------------
# _build_failure_policy tests
# ---------------------------------------------------------------------------


class TestBuildFailurePolicy:
    """Tests for _build_failure_policy (N single-mode + 1 combined)."""

    def test_policy_count(self):
        from netlab.autoresearch.scenario_generator import FAILURE_MODE_NAMES

        config = DcBbScenarioConfig()
        policy = _build_failure_policy(config)
        assert len(policy) == len(FAILURE_MODE_NAMES) + 1  # N modes + 1 combined

    def test_single_mode_policies_have_one_mode(self):
        from netlab.autoresearch.scenario_generator import FAILURE_MODE_NAMES

        config = DcBbScenarioConfig()
        policy = _build_failure_policy(config)
        for name in FAILURE_MODE_NAMES:
            p = policy[f"fm_{name}"]
            assert len(p["modes"]) == 1
            assert p["modes"][0]["weight"] == 1.0

    def test_combined_policy_has_all_modes(self):
        from netlab.autoresearch.scenario_generator import FAILURE_MODE_NAMES

        config = DcBbScenarioConfig()
        combined = _build_failure_policy(config)["fm_combined"]
        assert len(combined["modes"]) == len(FAILURE_MODE_NAMES)

    def test_combined_weights_sum_to_one(self):
        config = DcBbScenarioConfig()
        modes = _build_failure_policy(config)["fm_combined"]["modes"]
        total = sum(m["weight"] for m in modes)
        assert abs(total - 1.0) < 1e-9

    def test_failure_policies_match_workflow(self):
        """Every TMP step references a defined policy."""
        config = DcBbScenarioConfig()
        policy = _build_failure_policy(config)
        workflow = _build_workflow(config)
        for step in workflow:
            if step["type"] == "TrafficMatrixPlacement":
                assert step["failure_policy"] in policy, (
                    f"{step['name']} refs missing policy"
                )
