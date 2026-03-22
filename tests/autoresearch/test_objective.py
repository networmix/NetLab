"""Tests for objective function module.

Covers every row in the Step 3 acceptance criteria table.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from netlab.autoresearch.objective import ObjectiveFunction

# ---------------------------------------------------------------------------
# Fixtures — inline mock results and objective YAML
# ---------------------------------------------------------------------------

OBJECTIVE_YAML = textwrap.dedent("""\
    direction: maximize
    primary_metric: bac_auc
    metrics:
      alpha_star:
        path: "workflow.msd_baseline.alpha_star"
      bac_auc:
        path: "workflow.tm_placement.bac.auc_normalized"
      total_cost:
        path: "workflow.msd_baseline.total_cost"
    constraints:
      - metric: alpha_star
        operator: ">="
        value: 1.0
        name: "alpha_feasibility"
""")

OBJECTIVE_MINIMIZE_YAML = textwrap.dedent("""\
    direction: minimize
    primary_metric: total_cost
    metrics:
      alpha_star:
        path: "workflow.msd_baseline.alpha_star"
      bac_auc:
        path: "workflow.tm_placement.bac.auc_normalized"
      total_cost:
        path: "workflow.msd_baseline.total_cost"
    constraints: []
""")

OBJECTIVE_MULTI_CONSTRAINT_YAML = textwrap.dedent("""\
    direction: maximize
    primary_metric: bac_auc
    metrics:
      alpha_star:
        path: "workflow.msd_baseline.alpha_star"
      bac_auc:
        path: "workflow.tm_placement.bac.auc_normalized"
      total_cost:
        path: "workflow.msd_baseline.total_cost"
    constraints:
      - metric: alpha_star
        operator: ">="
        value: 1.0
        name: "alpha_feasibility"
      - metric: total_cost
        operator: "<="
        value: 50.0
        name: "cost_budget"
""")


def _make_results(alpha_star: float, bac_auc: float, total_cost: float) -> dict:
    return {
        "workflow": {
            "msd_baseline": {"alpha_star": alpha_star, "total_cost": total_cost},
            "tm_placement": {"bac": {"auc_normalized": bac_auc}},
        }
    }


@pytest.fixture()
def obj_maximize(tmp_path: Path) -> ObjectiveFunction:
    path = tmp_path / "objective.yml"
    path.write_text(OBJECTIVE_YAML)
    return ObjectiveFunction(path)


@pytest.fixture()
def obj_minimize(tmp_path: Path) -> ObjectiveFunction:
    path = tmp_path / "objective.yml"
    path.write_text(OBJECTIVE_MINIMIZE_YAML)
    return ObjectiveFunction(path)


@pytest.fixture()
def obj_multi_constraint(tmp_path: Path) -> ObjectiveFunction:
    path = tmp_path / "objective.yml"
    path.write_text(OBJECTIVE_MULTI_CONSTRAINT_YAML)
    return ObjectiveFunction(path)


# ---------------------------------------------------------------------------
# Test: Extract metric (alpha_star)
# ---------------------------------------------------------------------------


class TestExtractMetric:
    def test_extract_alpha_star(self, obj_maximize: ObjectiveFunction) -> None:
        """Extract metric: results dict, metric_key='alpha_star' -> exact float match."""
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        value = obj_maximize.extract_metric(results, "alpha_star")
        assert value == 1.5

    def test_extract_bac_auc(self, obj_maximize: ObjectiveFunction) -> None:
        """Extract BAC: results dict, metric_key='bac_auc' -> exact float match."""
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        value = obj_maximize.extract_metric(results, "bac_auc")
        assert value == 0.85

    def test_extract_total_cost(self, obj_maximize: ObjectiveFunction) -> None:
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        value = obj_maximize.extract_metric(results, "total_cost")
        assert value == 100.0


# ---------------------------------------------------------------------------
# Test: Primary metric score (maximize)
# ---------------------------------------------------------------------------


class TestPrimaryMetricScore:
    def test_maximize_higher_bac_scores_higher(
        self, obj_maximize: ObjectiveFunction
    ) -> None:
        """direction=maximize, bac_auc 0.9 > 0.7 -> score(0.9) > score(0.7)."""
        results_07 = _make_results(alpha_star=1.5, bac_auc=0.7, total_cost=100.0)
        results_09 = _make_results(alpha_star=1.5, bac_auc=0.9, total_cost=100.0)

        r07 = obj_maximize.evaluate(results_07)
        r09 = obj_maximize.evaluate(results_09)

        assert r09.score > r07.score
        assert r09.primary_value == 0.9
        assert r07.primary_value == 0.7

    def test_minimize_lower_cost_scores_higher(
        self, obj_minimize: ObjectiveFunction
    ) -> None:
        """direction=minimize, total_cost 100 and 200 -> score(100) > score(200)."""
        results_100 = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        results_200 = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=200.0)

        r100 = obj_minimize.evaluate(results_100)
        r200 = obj_minimize.evaluate(results_200)

        assert r100.score > r200.score
        assert r100.primary_value == 100.0
        assert r200.primary_value == 200.0


# ---------------------------------------------------------------------------
# Test: Constraint pass
# ---------------------------------------------------------------------------


class TestConstraintPass:
    def test_constraint_passes(self, obj_maximize: ObjectiveFunction) -> None:
        """alpha_star >= 1.0 with alpha_star=1.5 -> feasible, no penalty."""
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        result = obj_maximize.evaluate(results)

        assert result.status == "feasible"
        assert result.violated_constraints == []
        # Score equals primary_value for maximize direction, no penalty
        assert result.score == 0.85

    def test_constraint_exact_boundary(self, obj_maximize: ObjectiveFunction) -> None:
        """alpha_star >= 1.0 with alpha_star=1.0 -> feasible (boundary)."""
        results = _make_results(alpha_star=1.0, bac_auc=0.85, total_cost=100.0)
        result = obj_maximize.evaluate(results)

        assert result.status == "feasible"
        assert result.violated_constraints == []


# ---------------------------------------------------------------------------
# Test: Constraint fail
# ---------------------------------------------------------------------------


class TestConstraintFail:
    def test_constraint_fails(self, obj_maximize: ObjectiveFunction) -> None:
        """alpha_star >= 1.0 with alpha_star=0.8 -> infeasible, penalized score."""
        results_feasible = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        results_infeasible = _make_results(
            alpha_star=0.8, bac_auc=0.85, total_cost=100.0
        )

        r_feasible = obj_maximize.evaluate(results_feasible)
        r_infeasible = obj_maximize.evaluate(results_infeasible)

        assert r_infeasible.status == "infeasible"
        assert "alpha_feasibility" in r_infeasible.violated_constraints
        # Infeasible score must be less than feasible with same primary
        assert r_infeasible.score < r_feasible.score


# ---------------------------------------------------------------------------
# Test: Multiple constraints (one passes, one fails)
# ---------------------------------------------------------------------------


class TestMultipleConstraints:
    def test_one_pass_one_fail(self, obj_multi_constraint: ObjectiveFunction) -> None:
        """alpha_star >= 1.0 passes, total_cost <= 50.0 fails -> infeasible,
        violated_constraints lists the failing one by name."""
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        result = obj_multi_constraint.evaluate(results)

        assert result.status == "infeasible"
        assert "cost_budget" in result.violated_constraints
        assert "alpha_feasibility" not in result.violated_constraints
        assert len(result.violated_constraints) == 1

    def test_both_pass(self, obj_multi_constraint: ObjectiveFunction) -> None:
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=30.0)
        result = obj_multi_constraint.evaluate(results)

        assert result.status == "feasible"
        assert result.violated_constraints == []

    def test_both_fail(self, obj_multi_constraint: ObjectiveFunction) -> None:
        results = _make_results(alpha_star=0.5, bac_auc=0.85, total_cost=100.0)
        result = obj_multi_constraint.evaluate(results)

        assert result.status == "infeasible"
        assert "alpha_feasibility" in result.violated_constraints
        assert "cost_budget" in result.violated_constraints
        assert len(result.violated_constraints) == 2
        # Penalty is 1e6 * 2
        assert result.score == 0.85 - 2e6


# ---------------------------------------------------------------------------
# Test: Missing metric
# ---------------------------------------------------------------------------


class TestMissingMetric:
    def test_undefined_metric_key(self, obj_maximize: ObjectiveFunction) -> None:
        """metric_key='nonexistent' -> raises KeyError with metric name."""
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        with pytest.raises(KeyError, match="nonexistent"):
            obj_maximize.extract_metric(results, "nonexistent")

    def test_missing_path_in_results(self, obj_maximize: ObjectiveFunction) -> None:
        """Defined metric but path doesn't exist in results -> KeyError."""
        results = {"workflow": {"msd_baseline": {"alpha_star": 1.5}}}
        # bac_auc path is workflow.tm_placement.bac.auc_normalized — missing
        with pytest.raises(KeyError, match="bac_auc"):
            obj_maximize.extract_metric(results, "bac_auc")


# ---------------------------------------------------------------------------
# Test: Objective from YAML
# ---------------------------------------------------------------------------


class TestObjectiveFromYAML:
    def test_load_direction(self, obj_maximize: ObjectiveFunction) -> None:
        assert obj_maximize.direction == "maximize"

    def test_load_primary_metric(self, obj_maximize: ObjectiveFunction) -> None:
        assert obj_maximize.primary_metric == "bac_auc"

    def test_load_minimize(self, obj_minimize: ObjectiveFunction) -> None:
        assert obj_minimize.direction == "minimize"
        assert obj_minimize.primary_metric == "total_cost"

    def test_full_roundtrip(self, tmp_path: Path) -> None:
        """Load objective.yml, verify all fields match file contents."""
        yaml_text = textwrap.dedent("""\
            direction: minimize
            primary_metric: total_cost
            metrics:
              alpha_star:
                path: "workflow.msd_baseline.alpha_star"
              total_cost:
                path: "workflow.msd_baseline.total_cost"
            constraints:
              - metric: alpha_star
                operator: ">="
                value: 1.0
                name: "alpha_check"
              - metric: total_cost
                operator: "<="
                value: 500.0
                name: "cost_cap"
        """)
        path = tmp_path / "objective.yml"
        path.write_text(yaml_text)
        obj = ObjectiveFunction(path)

        assert obj.direction == "minimize"
        assert obj.primary_metric == "total_cost"
        # Verify constraints load by evaluating
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        result = obj.evaluate(results)
        assert result.status == "feasible"
        assert "alpha_star" in result.all_metrics
        assert "total_cost" in result.all_metrics


# ---------------------------------------------------------------------------
# Test: Score computation details
# ---------------------------------------------------------------------------


class TestScoreComputation:
    def test_maximize_score_equals_primary(
        self, obj_maximize: ObjectiveFunction
    ) -> None:
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        result = obj_maximize.evaluate(results)
        assert result.score == 0.85

    def test_minimize_score_negates_primary(
        self, obj_minimize: ObjectiveFunction
    ) -> None:
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        result = obj_minimize.evaluate(results)
        assert result.score == -100.0

    def test_penalty_magnitude(self, obj_maximize: ObjectiveFunction) -> None:
        """Infeasible result: score = primary - 1e6 * num_violations."""
        results = _make_results(alpha_star=0.5, bac_auc=0.85, total_cost=100.0)
        result = obj_maximize.evaluate(results)
        assert result.score == 0.85 - 1e6

    def test_infeasible_always_below_feasible(
        self, obj_maximize: ObjectiveFunction
    ) -> None:
        """Even with a higher primary value, infeasible scores below any feasible."""
        results_infeasible = _make_results(
            alpha_star=0.1, bac_auc=0.99, total_cost=100.0
        )
        results_feasible = _make_results(alpha_star=1.5, bac_auc=0.01, total_cost=100.0)
        r_inf = obj_maximize.evaluate(results_infeasible)
        r_feas = obj_maximize.evaluate(results_feasible)
        assert r_inf.score < r_feas.score


# ---------------------------------------------------------------------------
# Test: all_metrics populated
# ---------------------------------------------------------------------------


class TestAllMetrics:
    def test_all_metrics_present(self, obj_maximize: ObjectiveFunction) -> None:
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        result = obj_maximize.evaluate(results)
        assert result.all_metrics == {
            "alpha_star": 1.5,
            "bac_auc": 0.85,
            "total_cost": 100.0,
        }


# ---------------------------------------------------------------------------
# Test: Invalid objective YAML
# ---------------------------------------------------------------------------


class TestInvalidObjective:
    def test_invalid_direction(self, tmp_path: Path) -> None:
        yaml_text = textwrap.dedent("""\
            direction: unknown
            primary_metric: bac_auc
            metrics:
              bac_auc:
                path: "workflow.bac"
        """)
        path = tmp_path / "objective.yml"
        path.write_text(yaml_text)
        with pytest.raises(ValueError, match="direction"):
            ObjectiveFunction(path)

    def test_primary_metric_not_in_metrics(self, tmp_path: Path) -> None:
        yaml_text = textwrap.dedent("""\
            direction: maximize
            primary_metric: nonexistent
            metrics:
              bac_auc:
                path: "workflow.bac"
        """)
        path = tmp_path / "objective.yml"
        path.write_text(yaml_text)
        with pytest.raises(ValueError, match="nonexistent"):
            ObjectiveFunction(path)

    def test_unsupported_operator(self, tmp_path: Path) -> None:
        yaml_text = textwrap.dedent("""\
            direction: maximize
            primary_metric: bac_auc
            metrics:
              bac_auc:
                path: "workflow.bac"
            constraints:
              - metric: bac_auc
                operator: "!="
                value: 0.5
        """)
        path = tmp_path / "objective.yml"
        path.write_text(yaml_text)
        with pytest.raises(ValueError, match="operator"):
            ObjectiveFunction(path)


# ---------------------------------------------------------------------------
# Test: Equality constraint operator
# ---------------------------------------------------------------------------


class TestEqualityConstraint:
    def test_eq_passes(self, tmp_path: Path) -> None:
        yaml_text = textwrap.dedent("""\
            direction: maximize
            primary_metric: bac_auc
            metrics:
              bac_auc:
                path: "workflow.tm_placement.bac.auc_normalized"
              alpha_star:
                path: "workflow.msd_baseline.alpha_star"
            constraints:
              - metric: alpha_star
                operator: "=="
                value: 1.5
                name: "exact_alpha"
        """)
        path = tmp_path / "objective.yml"
        path.write_text(yaml_text)
        obj = ObjectiveFunction(path)
        results = _make_results(alpha_star=1.5, bac_auc=0.85, total_cost=100.0)
        result = obj.evaluate(results)
        assert result.status == "feasible"

    def test_eq_fails(self, tmp_path: Path) -> None:
        yaml_text = textwrap.dedent("""\
            direction: maximize
            primary_metric: bac_auc
            metrics:
              bac_auc:
                path: "workflow.tm_placement.bac.auc_normalized"
              alpha_star:
                path: "workflow.msd_baseline.alpha_star"
            constraints:
              - metric: alpha_star
                operator: "=="
                value: 1.5
                name: "exact_alpha"
        """)
        path = tmp_path / "objective.yml"
        path.write_text(yaml_text)
        obj = ObjectiveFunction(path)
        results = _make_results(alpha_star=2.0, bac_auc=0.85, total_cost=100.0)
        result = obj.evaluate(results)
        assert result.status == "infeasible"
        assert "exact_alpha" in result.violated_constraints
