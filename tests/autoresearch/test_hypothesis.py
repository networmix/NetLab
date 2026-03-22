"""Tests for hypothesis management module.

Covers every row in the Step 1 acceptance criteria table.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from netlab.autoresearch.hypothesis import (
    Hypothesis,
    HypothesisMerger,
    HypothesisTemplate,
    ParamDef,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEMPLATE_YAML = textwrap.dedent("""\
    params:
      link_capacity:
        type: int
        range: [100, 1000]
        step: 100
        default: 500
        description: "Link capacity in Mbps"
      flow_policy:
        type: enum
        values: ["SHORTEST_PATHS_ECMP", "SHORTEST_PATHS", "MIN_COST"]
        default: "SHORTEST_PATHS_ECMP"
        description: "Flow policy for routing"
      demand_scale:
        type: float
        range: [0.1, 10.0]
        step: 0.1
        default: 1.0
        description: "Demand scaling factor"
      num_paths:
        type: int
        range: [1, 16]
        step: 1
        default: 4
        description: "Number of ECMP paths"
""")


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    p = tmp_path / "hypothesis_template.yml"
    p.write_text(TEMPLATE_YAML)
    return p


@pytest.fixture
def template(template_path: Path) -> HypothesisTemplate:
    return HypothesisTemplate(template_path)


VALID_PARAMS = {
    "link_capacity": 400,
    "flow_policy": "SHORTEST_PATHS_ECMP",
    "demand_scale": 2.5,
    "num_paths": 8,
}


# ---------------------------------------------------------------------------
# Parse valid template
# ---------------------------------------------------------------------------


class TestParseTemplate:
    def test_parse_valid_template(self, template: HypothesisTemplate) -> None:
        """template.params has 4 entries, each with correct type/range/default."""
        params = template.params
        assert len(params) == 4

        lc = params["link_capacity"]
        assert lc.type == "int"
        assert lc.range == (100.0, 1000.0)
        assert lc.step == 100.0
        assert lc.default == 500
        assert lc.description == "Link capacity in Mbps"

        fp = params["flow_policy"]
        assert fp.type == "enum"
        assert fp.values == ["SHORTEST_PATHS_ECMP", "SHORTEST_PATHS", "MIN_COST"]
        assert fp.default == "SHORTEST_PATHS_ECMP"

        ds = params["demand_scale"]
        assert ds.type == "float"
        assert ds.range == (0.1, 10.0)
        assert ds.step == 0.1
        assert ds.default == 1.0

        np_ = params["num_paths"]
        assert np_.type == "int"
        assert np_.range == (1.0, 16.0)
        assert np_.default == 4


# ---------------------------------------------------------------------------
# Validate in-range
# ---------------------------------------------------------------------------


class TestValidateInRange:
    def test_validate_in_range(self, template: HypothesisTemplate) -> None:
        """Valid params produce no errors."""
        h = Hypothesis(VALID_PARAMS, template)
        errors = h.validate()
        assert errors == []


# ---------------------------------------------------------------------------
# Reject out-of-range
# ---------------------------------------------------------------------------


class TestRejectOutOfRange:
    def test_reject_out_of_range(self, template: HypothesisTemplate) -> None:
        """Out-of-range link_capacity produces error naming link_capacity and range."""
        params = {**VALID_PARAMS, "link_capacity": 9999}
        h = Hypothesis(params, template)
        errors = h.validate()
        assert len(errors) == 1
        assert "link_capacity" in errors[0]
        assert "9999" in errors[0]
        # Should mention the range bounds
        assert "100" in errors[0]
        assert "1000" in errors[0]


# ---------------------------------------------------------------------------
# Reject wrong type
# ---------------------------------------------------------------------------


class TestRejectWrongType:
    def test_reject_wrong_type(self, template: HypothesisTemplate) -> None:
        """String for int param produces error naming link_capacity and int."""
        params = {**VALID_PARAMS, "link_capacity": "fast"}
        h = Hypothesis(params, template)
        errors = h.validate()
        assert len(errors) == 1
        assert "link_capacity" in errors[0]
        assert "int" in errors[0]


# ---------------------------------------------------------------------------
# Reject unknown param
# ---------------------------------------------------------------------------


class TestRejectUnknownParam:
    def test_reject_unknown_param(self, template: HypothesisTemplate) -> None:
        """Unknown param 'nonexistent' produces error naming it."""
        params = {**VALID_PARAMS, "nonexistent": 5}
        h = Hypothesis(params, template)
        errors = h.validate()
        assert any("nonexistent" in e for e in errors)


# ---------------------------------------------------------------------------
# Reject missing param
# ---------------------------------------------------------------------------


class TestRejectMissingParam:
    def test_reject_missing_param(self, template: HypothesisTemplate) -> None:
        """Providing only 3 of 4 params produces error naming the missing one."""
        params = {
            "link_capacity": 400,
            "flow_policy": "SHORTEST_PATHS_ECMP",
            "demand_scale": 2.5,
            # num_paths is missing
        }
        h = Hypothesis(params, template)
        errors = h.validate()
        assert any("num_paths" in e for e in errors)


# ---------------------------------------------------------------------------
# Deterministic hash
# ---------------------------------------------------------------------------


class TestDeterministicHash:
    def test_same_params_different_order(self, template: HypothesisTemplate) -> None:
        """Same params in different dict order produce identical hash."""
        params_a = {
            "link_capacity": 400,
            "flow_policy": "SHORTEST_PATHS_ECMP",
            "demand_scale": 2.5,
            "num_paths": 8,
        }
        params_b = {
            "num_paths": 8,
            "demand_scale": 2.5,
            "link_capacity": 400,
            "flow_policy": "SHORTEST_PATHS_ECMP",
        }
        h1 = Hypothesis(params_a, template)
        h2 = Hypothesis(params_b, template)
        assert h1.params_hash == h2.params_hash

    def test_hash_distinguishes_different_values(
        self, template: HypothesisTemplate
    ) -> None:
        """Different param values produce different hashes."""
        h1 = Hypothesis(VALID_PARAMS, template)
        h2 = Hypothesis({**VALID_PARAMS, "link_capacity": 500}, template)
        assert h1.params_hash != h2.params_hash


# ---------------------------------------------------------------------------
# Substitute valid (HypothesisMerger)
# ---------------------------------------------------------------------------

BASE_SCENARIO = textwrap.dedent("""\
    network:
      links:
        - name: link_a
          capacity: ${{link_capacity}}
      routing:
        policy: ${{flow_policy}}
""")


class TestSubstituteValid:
    def test_substitute_valid(self, template: HypothesisTemplate) -> None:
        """Merged YAML parses; capacity is integer 600, not string."""
        params = {**VALID_PARAMS, "link_capacity": 600}
        h = Hypothesis(params, template)
        merger = HypothesisMerger(BASE_SCENARIO, template)
        result = merger.merge(h)

        assert result["network"]["links"][0]["capacity"] == 600
        assert isinstance(result["network"]["links"][0]["capacity"], int)
        assert result["network"]["routing"]["policy"] == "SHORTEST_PATHS_ECMP"


# ---------------------------------------------------------------------------
# Reject unreplaced
# ---------------------------------------------------------------------------


class TestRejectUnreplaced:
    def test_reject_unreplaced(self, template_path: Path) -> None:
        """${{missing_param}} not in hypothesis raises error citing missing_param."""
        # Create a template with only one param
        minimal_yaml = textwrap.dedent("""\
            params:
              link_capacity:
                type: int
                range: [100, 1000]
                default: 500
        """)
        p = template_path.parent / "minimal_template.yml"
        p.write_text(minimal_yaml)
        tmpl = HypothesisTemplate(p)

        scenario_text = textwrap.dedent("""\
            network:
              capacity: ${{link_capacity}}
              extra: ${{missing_param}}
        """)
        merger = HypothesisMerger(scenario_text, tmpl)
        h = Hypothesis({"link_capacity": 600}, tmpl)

        with pytest.raises(ValueError, match="missing_param"):
            merger.merge(h)


# ---------------------------------------------------------------------------
# Cross-check: extra placeholder
# ---------------------------------------------------------------------------


class TestCrossCheckExtraPlaceholder:
    def test_extra_placeholder(self, template_path: Path) -> None:
        """Placeholder ${{foo}} not in template produces validation error citing foo."""
        tmpl = HypothesisTemplate(template_path)
        scenario_text = "value: ${{foo}}\ncap: ${{link_capacity}}"
        merger = HypothesisMerger(scenario_text, tmpl)
        errors = merger.validate_placeholders()
        assert any("foo" in e for e in errors)


# ---------------------------------------------------------------------------
# Cross-check: unused param
# ---------------------------------------------------------------------------


class TestCrossCheckUnusedParam:
    def test_unused_param(self, template_path: Path) -> None:
        """Template has 'num_paths' etc. but scenario has no placeholder for them."""
        tmpl = HypothesisTemplate(template_path)
        # Only use link_capacity placeholder; demand_scale, flow_policy, num_paths unused
        scenario_text = "cap: ${{link_capacity}}"
        merger = HypothesisMerger(scenario_text, tmpl)
        errors = merger.validate_placeholders()
        # Should report demand_scale, flow_policy, num_paths as unused
        unused_names = {"demand_scale", "flow_policy", "num_paths"}
        for name in unused_names:
            assert any(name in e for e in errors), f"Expected error about {name}"


# ---------------------------------------------------------------------------
# Collision safety: single-brace is ignored
# ---------------------------------------------------------------------------


class TestCollisionSafety:
    def test_single_brace_ignored(self, template: HypothesisTemplate) -> None:
        """${single_brace} is NOT matched; only ${{double}} is substituted."""
        scenario_text = textwrap.dedent("""\
            env: ${single_brace}
            capacity: ${{link_capacity}}
        """)
        merger = HypothesisMerger(scenario_text, template)
        h = Hypothesis(VALID_PARAMS, template)
        result = merger.merge(h)

        # The single-brace token should be preserved as-is in the parsed output.
        # YAML treats ${single_brace} as a plain string.
        assert result["env"] == "${single_brace}"
        assert result["capacity"] == 400


# ---------------------------------------------------------------------------
# Edge cases and additional coverage
# ---------------------------------------------------------------------------


class TestParamDefDataclass:
    def test_param_def_defaults(self) -> None:
        """ParamDef can be constructed with minimal args."""
        pd = ParamDef(name="x", type="int")
        assert pd.name == "x"
        assert pd.type == "int"
        assert pd.range is None
        assert pd.step is None
        assert pd.values is None
        assert pd.default is None
        assert pd.description == ""


class TestEnumValidation:
    def test_reject_invalid_enum_value(self, template: HypothesisTemplate) -> None:
        """Invalid enum value produces error."""
        params = {**VALID_PARAMS, "flow_policy": "INVALID_POLICY"}
        h = Hypothesis(params, template)
        errors = h.validate()
        assert any("flow_policy" in e for e in errors)
        assert any("INVALID_POLICY" in e for e in errors)


class TestFloatRangeValidation:
    def test_reject_float_out_of_range(self, template: HypothesisTemplate) -> None:
        """Float value out of range produces error."""
        params = {**VALID_PARAMS, "demand_scale": 99.9}
        h = Hypothesis(params, template)
        errors = h.validate()
        assert len(errors) == 1
        assert "demand_scale" in errors[0]

    def test_int_accepted_for_float_param(self, template: HypothesisTemplate) -> None:
        """An int value for a float param should be accepted."""
        params = {**VALID_PARAMS, "demand_scale": 5}
        h = Hypothesis(params, template)
        errors = h.validate()
        assert errors == []
