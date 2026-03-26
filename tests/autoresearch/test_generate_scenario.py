"""Tests for generate_scenario and validate_config in scenario_generator.py."""

from __future__ import annotations

import pytest
import yaml

from netlab.autoresearch.scenario_generator import (
    DcBbScenarioConfig,
    generate_scenario,
    generate_scenario_with_validation,
    validate_config,
)

# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """validate_config returns errors for invalid configs, empty for valid."""

    def test_default_config_is_valid(self):
        errors = validate_config(DcBbScenarioConfig())
        assert errors == []

    def test_invalid_g_abc1(self):
        errors = validate_config(DcBbScenarioConfig(g_abc1=7))
        assert any("g_abc1=7" in e for e in errors)

    def test_invalid_g_xyz1(self):
        errors = validate_config(DcBbScenarioConfig(g_xyz1=5))
        assert any("g_xyz1=5" in e for e in errors)

    def test_invalid_layout_abc1(self):
        errors = validate_config(DcBbScenarioConfig(layout_abc1=(8, 4, 16, 4)))
        assert any("layout_abc1" in e for e in errors)

    def test_invalid_layout_xyz1(self):
        errors = validate_config(DcBbScenarioConfig(layout_xyz1=(3, 3, 3, 3)))
        assert any("layout_xyz1" in e for e in errors)

    def test_port_constraint_fadu(self):
        errors = validate_config(
            DcBbScenarioConfig(g_abc1=16, layout_abc1=(4, 4, 4, 4))
        )
        assert errors == []  # G=16 with valid layout

    def test_port_constraint_xsw(self):
        errors = validate_config(DcBbScenarioConfig(g_xyz1=64))
        assert errors == []  # G=64 is valid

    def test_valid_non_default_g_abc1(self):
        errors = validate_config(
            DcBbScenarioConfig(g_abc1=32, layout_abc1=(8, 4, 8, 4))
        )
        assert errors == []

    def test_valid_non_default_g_xyz1(self):
        errors = validate_config(
            DcBbScenarioConfig(g_xyz1=128, layout_xyz1=(16, 8, 32, 4))
        )
        assert errors == []

    def test_multiple_errors_reported(self):
        errors = validate_config(DcBbScenarioConfig(g_abc1=7, g_xyz1=5))
        assert len(errors) >= 2


# ---------------------------------------------------------------------------
# generate_scenario structure
# ---------------------------------------------------------------------------


class TestGenerateScenarioStructure:
    """Verify generate_scenario produces a well-formed scenario dict."""

    @pytest.fixture(scope="class")
    def scenario(self):
        return generate_scenario(DcBbScenarioConfig())

    def test_top_level_keys(self, scenario):
        expected_keys = {
            "seed",
            "network",
            "risk_groups",
            "demands",
            "failures",
            "workflow",
        }
        assert set(scenario.keys()) == expected_keys

    def test_seed(self, scenario):
        assert scenario["seed"] == 42

    def test_network_has_link_rules(self, scenario):
        assert "link_rules" in scenario["network"]
        assert isinstance(scenario["network"]["link_rules"], list)
        assert len(scenario["network"]["link_rules"]) > 0

    def test_risk_groups_is_list(self, scenario):
        assert isinstance(scenario["risk_groups"], list)
        assert len(scenario["risk_groups"]) == 274

    def test_demands_is_dict(self, scenario):
        assert isinstance(scenario["demands"], dict)
        assert "baseline_traffic_matrix" in scenario["demands"]

    def test_demands_use_anchors(self, scenario):
        for d in scenario["demands"]["baseline_traffic_matrix"]:
            assert d["source"].startswith("^")
            assert d["source"].endswith("$") or d["target"].endswith("$")

    def test_failures_is_dict(self, scenario):
        from netlab.autoresearch.scenario_generator import FAILURE_MODE_NAMES

        assert isinstance(scenario["failures"], dict)
        assert "fm_combined" in scenario["failures"]
        assert len(scenario["failures"]) == len(FAILURE_MODE_NAMES) + 1

    def test_workflow_is_list(self, scenario):
        from netlab.autoresearch.scenario_generator import FAILURE_MODE_NAMES

        assert isinstance(scenario["workflow"], list)
        assert len(scenario["workflow"]) == 1 + len(FAILURE_MODE_NAMES) + 1

    def test_workflow_steps_have_demand_set(self, scenario):
        for step in scenario["workflow"]:
            assert "demand_set" in step
            assert "demands" not in step

    def test_failure_rules_use_match_block(self, scenario):
        for policy_name, policy in scenario["failures"].items():
            for mode in policy["modes"]:
                for rule in mode["rules"]:
                    if rule.get("match", {}).get("conditions"):
                        assert "conditions" not in rule, (
                            f"{policy_name}: conditions at rule level"
                        )

    def test_workflow_steps_no_invalid_keys(self, scenario):
        for step in scenario["workflow"]:
            if step["type"] == "MaximumSupportedDemand":
                assert "flow_policy" not in step
                assert "metadata" not in step
            elif step["type"] == "TrafficMatrixPlacement":
                assert "metadata" not in step


# ---------------------------------------------------------------------------
# Post-expansion validation
# ---------------------------------------------------------------------------


@pytest.mark.timeout(180)
class TestPostExpansionValidation:
    """Validate the expanded network matches expected counts."""

    @pytest.fixture(scope="class")
    def expanded(self):
        from ngraph.scenario import Scenario

        config = DcBbScenarioConfig(failure_iterations=0)
        scenario, expected = generate_scenario_with_validation(config)
        yaml_str = yaml.dump(scenario, default_flow_style=False, sort_keys=False)
        sc = Scenario.from_yaml(yaml_str)
        return sc.network, expected

    def test_node_count(self, expanded):
        network, expected = expanded
        assert len(network.nodes) == expected.nodes

    def test_link_count(self, expanded):
        network, expected = expanded
        assert len(network.links) == expected.links

    def test_level2_validation(self, expanded):
        from netlab.autoresearch.scenario_validation import validate_expanded_network

        network, expected = expanded
        errors = validate_expanded_network(network, expected)
        assert errors == [], f"Validation errors: {errors}"

    def test_no_cross_group_links(self, expanded):
        from netlab.autoresearch.scenario_validation import (
            validate_no_cross_group_links,
        )

        network, _ = expanded
        errors = validate_no_cross_group_links(network)
        assert errors == [], f"Cross-group errors: {errors[:3]}"


# ---------------------------------------------------------------------------
# generate_scenario with invalid config
# ---------------------------------------------------------------------------


class TestGenerateScenarioValidation:
    """generate_scenario raises ValueError for invalid configs."""

    def test_raises_on_invalid_g(self):
        with pytest.raises(ValueError, match="Invalid config"):
            generate_scenario(DcBbScenarioConfig(g_abc1=7))

    def test_raises_on_invalid_layout(self):
        with pytest.raises(ValueError, match="Invalid config"):
            generate_scenario(DcBbScenarioConfig(layout_abc1=(8, 4, 16, 4)))
