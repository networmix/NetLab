"""Tests for generate_scenario and validate_config in scenario_generator.py.

Step E-8: End-to-end scenario generation tests.
"""

import os
import tempfile

import pytest
import yaml

from netlab.autoresearch.scenario_generator import (
    DcBbScenarioConfig,
    generate_scenario,
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
        cfg = DcBbScenarioConfig(g_abc1=7)
        errors = validate_config(cfg)
        assert len(errors) > 0
        assert any("g_abc1" in e for e in errors)

    def test_invalid_g_xyz1(self):
        cfg = DcBbScenarioConfig(g_xyz1=7)
        errors = validate_config(cfg)
        assert len(errors) > 0
        assert any("g_xyz1" in e for e in errors)

    def test_invalid_layout_abc1(self):
        # (8, 4) product is 32, not 64
        cfg = DcBbScenarioConfig(layout_abc1=(8, 4, 16, 4))
        errors = validate_config(cfg)
        assert any("layout_abc1" in e for e in errors)

    def test_invalid_layout_xyz1(self):
        cfg = DcBbScenarioConfig(layout_xyz1=(8, 4, 16, 4))
        errors = validate_config(cfg)
        assert any("layout_xyz1" in e for e in errors)

    def test_port_constraint_fadu(self):
        # G_abc1=1 means k_fadu = bb_total = 256, which exceeds 16
        # But G=1 is not viable either, so check port error is present
        cfg = DcBbScenarioConfig(g_abc1=1)
        errors = validate_config(cfg)
        assert any("k_fadu" in e or "g_abc1" in e for e in errors)

    def test_port_constraint_xsw(self):
        cfg = DcBbScenarioConfig(g_xyz1=1)
        errors = validate_config(cfg)
        assert any("k_xsw" in e or "g_xyz1" in e for e in errors)

    def test_valid_non_default_g_abc1(self):
        """g_abc1=32 with matching layout should be valid."""
        cfg = DcBbScenarioConfig(g_abc1=32, layout_abc1=(8, 4, 8, 4))
        errors = validate_config(cfg)
        assert errors == []

    def test_valid_non_default_g_xyz1(self):
        """g_xyz1=128 with matching layout should be valid."""
        cfg = DcBbScenarioConfig(g_xyz1=128, layout_xyz1=(32, 4, 32, 4))
        errors = validate_config(cfg)
        assert errors == []

    def test_multiple_errors_reported(self):
        """Both g values invalid should report multiple errors."""
        cfg = DcBbScenarioConfig(g_abc1=7, g_xyz1=7)
        errors = validate_config(cfg)
        abc1_errors = [e for e in errors if "abc1" in e]
        xyz1_errors = [e for e in errors if "xyz1" in e]
        assert len(abc1_errors) > 0
        assert len(xyz1_errors) > 0


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

    def test_node_count(self, scenario):
        assert len(scenario["network"]["nodes"]) == 3833

    def test_link_count_in_range(self, scenario):
        link_count = len(scenario["network"]["links"])
        assert 35000 <= link_count <= 48000, (
            f"Link count {link_count} outside expected range"
        )

    def test_risk_groups_is_list(self, scenario):
        assert isinstance(scenario["risk_groups"], list)
        assert len(scenario["risk_groups"]) > 0

    def test_demands_is_dict(self, scenario):
        assert isinstance(scenario["demands"], dict)
        assert "baseline_traffic_matrix" in scenario["demands"]

    def test_failures_is_dict(self, scenario):
        assert isinstance(scenario["failures"], dict)
        assert "dc_bb_failures" in scenario["failures"]

    def test_workflow_is_list(self, scenario):
        assert isinstance(scenario["workflow"], list)
        assert len(scenario["workflow"]) == 2

    def test_workflow_steps_have_demand_set(self, scenario):
        """Workflow steps must use demand_set, not demands."""
        for step in scenario["workflow"]:
            assert "demand_set" in step, f"Step {step.get('name')} missing demand_set"
            assert "demands" not in step, (
                f"Step {step.get('name')} has stale 'demands' key"
            )

    def test_failure_rules_use_match_block(self, scenario):
        """Failure rules must nest conditions inside match, not at top level."""
        policy = scenario["failures"]["dc_bb_failures"]
        for mode in policy["modes"]:
            for rule in mode["rules"]:
                if rule.get("match", {}).get("conditions"):
                    assert "conditions" not in rule, (
                        "conditions should be inside match, not at rule level"
                    )

    def test_workflow_steps_no_invalid_keys(self, scenario):
        """MSD step should not have flow_policy; TMP should not have metadata."""
        for step in scenario["workflow"]:
            if step["type"] == "MaximumSupportedDemand":
                assert "flow_policy" not in step
                assert "metadata" not in step
            elif step["type"] == "TrafficMatrixPlacement":
                assert "metadata" not in step


# ---------------------------------------------------------------------------
# Risk group reference integrity
# ---------------------------------------------------------------------------


class TestRiskGroupReferences:
    """All risk groups referenced by links must be defined."""

    @pytest.fixture(scope="class")
    def scenario(self):
        return generate_scenario(DcBbScenarioConfig())

    def test_all_link_risk_groups_defined(self, scenario):
        defined = {rg["name"] for rg in scenario["risk_groups"]}
        for link in scenario["network"]["links"]:
            for rg_name in link.get("risk_groups", []):
                assert rg_name in defined, f"Undefined risk group: {rg_name}"


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


# ---------------------------------------------------------------------------
# YAML serialization + ngraph inspect
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.isfile(
        os.path.join(os.path.dirname(__file__), "..", "..", "venv", "bin", "ngraph")
    ),
    reason="ngraph binary not found",
)
class TestNgraphInspect:
    """Write scenario YAML and verify ngraph inspect succeeds."""

    @pytest.fixture(scope="class")
    def scenario_yaml_path(self):
        cfg = DcBbScenarioConfig()
        scenario = generate_scenario(cfg)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)
            path = f.name
        yield path
        os.unlink(path)

    @pytest.mark.timeout(120)
    def test_ngraph_inspect_succeeds(self, scenario_yaml_path):
        import subprocess

        ngraph_bin = os.path.join(
            os.path.dirname(__file__), "..", "..", "venv", "bin", "ngraph"
        )
        result = subprocess.run(
            [ngraph_bin, "inspect", scenario_yaml_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"ngraph inspect failed with code {result.returncode}:\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )
        assert "INSPECTION COMPLETE" in result.stdout
