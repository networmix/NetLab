"""Tests for AutoResearchRunner core loop."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from netlab.autoresearch.backend import MockBackend
from netlab.autoresearch.experiment_log import ExperimentLog, LogEntry
from netlab.autoresearch.runner import AutoResearchRunner, RunConfig

DATA_DIR = Path(__file__).parent / "data"


# Read the original square_mesh.yaml once at module level
_SQUARE_MESH_TEXT = (DATA_DIR / "square_mesh.yaml").read_text()


def _base_scenario_with_placeholder() -> str:
    """Return square_mesh.yaml with first link capacity replaced by ${{link_capacity}}."""
    return _SQUARE_MESH_TEXT.replace("capacity: 2.0", "capacity: ${{link_capacity}}", 1)


HYPOTHESIS_TEMPLATE_YAML = textwrap.dedent("""\
    params:
      link_capacity:
        type: float
        range: [0.5, 10.0]
        default: 2.0
        description: "Link capacity for the first link"
""")

# This objective.yml extracts alpha_star from the real ngraph output path
OBJECTIVE_YAML = textwrap.dedent("""\
    direction: maximize
    primary_metric: alpha_star
    metrics:
      alpha_star:
        path: "steps.msd_baseline.data.alpha_star"
""")

PROGRAM_MD = "Optimize link capacity for maximum throughput."


def make_project(tmp_path: Path) -> Path:
    """Create a minimal test project directory. Returns the project path."""
    proj = tmp_path / "project"
    proj.mkdir()
    (proj / "program.md").write_text(PROGRAM_MD)
    (proj / "hypothesis_template.yml").write_text(HYPOTHESIS_TEMPLATE_YAML)
    (proj / "objective.yml").write_text(OBJECTIVE_YAML)
    (proj / "base_scenario.yml").write_text(_base_scenario_with_placeholder())
    (proj / "results").mkdir()
    return proj


def _mock_response(link_capacity: float) -> str:
    """Build a mock LLM response with the given link_capacity."""
    return textwrap.dedent(f"""\
        I think we should try a capacity of {link_capacity}.

        ```yaml
        params:
          link_capacity: {link_capacity}
        ```
    """)


class TestInitValidatesWorkflow:
    def test_missing_msd_step_raises(self, tmp_path: Path) -> None:
        """Base scenario without MaximumSupportedDemand -> raises ValueError."""
        proj = make_project(tmp_path)
        # Overwrite base_scenario with one that has no MSD step
        no_msd_scenario = textwrap.dedent("""\
            seed: 42
            network:
              nodes:
                N1: {}
                N2: {}
              links:
                - source: N1
                  target: N2
                  capacity: ${{link_capacity}}
                  cost: 1.0
            workflow:
              - type: TrafficMatrixPlacement
                name: tm_placement
                demand_set: baseline
        """)
        (proj / "base_scenario.yml").write_text(no_msd_scenario)

        backend = MockBackend([])
        config = RunConfig(project_dir=proj, backend=backend, max_experiments=1)

        with pytest.raises(ValueError, match="MaximumSupportedDemand"):
            AutoResearchRunner(config)


class TestInitValidatesPlaceholders:
    def test_template_param_not_in_base_raises(self, tmp_path: Path) -> None:
        """Template has param 'foo' but base has no ${{foo}} -> raises ValueError."""
        proj = make_project(tmp_path)
        # Add a param 'foo' to the template that has no placeholder in base
        template_with_foo = textwrap.dedent("""\
            params:
              link_capacity:
                type: float
                range: [0.5, 10.0]
                default: 2.0
                description: "Link capacity"
              foo:
                type: float
                range: [0.0, 1.0]
                default: 0.5
                description: "A parameter with no placeholder"
        """)
        (proj / "hypothesis_template.yml").write_text(template_with_foo)

        backend = MockBackend([])
        config = RunConfig(project_dir=proj, backend=backend, max_experiments=1)

        with pytest.raises(ValueError, match="foo"):
            AutoResearchRunner(config)


class TestHappyPath:
    @pytest.mark.timeout(120)
    def test_three_experiments_succeed(self, tmp_path: Path) -> None:
        """MockBackend + real ngraph: 3 experiments, all succeed."""
        proj = make_project(tmp_path)

        responses = [
            _mock_response(2.0),
            _mock_response(3.0),
            _mock_response(5.0),
        ]
        backend = MockBackend(responses)
        config = RunConfig(
            project_dir=proj, backend=backend, max_experiments=3, seed=42
        )
        runner = AutoResearchRunner(config)
        runner.run()

        assert runner.status == "completed"

        log = ExperimentLog(proj)
        entries = log.load()
        assert len(entries) == 3

        for entry in entries:
            assert entry.status == "success"
            assert entry.metrics is not None
            assert "alpha_star" in entry.metrics
            assert entry.objective_score is not None

        for i in range(1, 4):
            exp_dir = proj / "results" / f"exp_{i:03d}"
            assert exp_dir.exists()
            assert (exp_dir / "scenario.yml").exists()
            assert (exp_dir / "scenario.results.json").exists()

        best_path = proj / "best_hypothesis.yml"
        assert best_path.exists()
        best_data = yaml.safe_load(best_path.read_text())
        assert "objective_score" in best_data
        assert "params" in best_data


class TestDeduplication:
    @pytest.mark.timeout(120)
    def test_same_params_twice_cached(self, tmp_path: Path) -> None:
        """Same params returned twice -> second is cached, only 2 ngraph calls."""
        proj = make_project(tmp_path)

        responses = [
            _mock_response(2.0),
            _mock_response(2.0),  # duplicate
            _mock_response(4.0),  # new
        ]
        backend = MockBackend(responses)
        config = RunConfig(
            project_dir=proj, backend=backend, max_experiments=3, seed=42
        )
        runner = AutoResearchRunner(config)
        runner.run()

        log = ExperimentLog(proj)
        entries = log.load()
        assert len(entries) == 3

        assert entries[0].params_hash == entries[1].params_hash

        assert entries[0].status == "success"
        assert entries[1].status == "cached"
        assert entries[2].status == "success"

        assert entries[1].metrics == entries[0].metrics
        assert entries[1].objective_score == entries[0].objective_score

        assert runner.ngraph_call_count == 2


class TestCircuitBreakerTrips:
    def test_five_bad_responses_halt(self, tmp_path: Path) -> None:
        """5 unparseable responses -> circuit breaker trips."""
        proj = make_project(tmp_path)

        bad_responses = ["not yaml at all {{{"] * 5 + [_mock_response(2.0)] * 3
        backend = MockBackend(bad_responses)
        config = RunConfig(
            project_dir=proj,
            backend=backend,
            max_experiments=10,
            circuit_breaker_threshold=5,
            seed=42,
        )
        runner = AutoResearchRunner(config)
        runner.run()

        assert runner.status == "circuit_breaker"

        log = ExperimentLog(proj)
        entries = log.load()
        assert len(entries) == 5
        for entry in entries:
            assert entry.status == "parse_error"


class TestCircuitBreakerResets:
    @pytest.mark.timeout(120)
    def test_resets_on_success(self, tmp_path: Path) -> None:
        """3 bad, 1 good, 3 bad -> breaker should NOT trip (threshold=5)."""
        proj = make_project(tmp_path)

        responses = (
            ["not yaml {{{"] * 3
            + [_mock_response(2.0)]  # good — resets counter
            + ["not yaml {{{"] * 3
            + [_mock_response(3.0)]  # would run if we get here
        )
        backend = MockBackend(responses)
        config = RunConfig(
            project_dir=proj,
            backend=backend,
            max_experiments=8,
            circuit_breaker_threshold=5,
            seed=42,
        )
        runner = AutoResearchRunner(config)
        runner.run()

        log = ExperimentLog(proj)
        entries = log.load()

        assert len(entries) == 8

        assert entries[0].status == "parse_error"
        assert entries[1].status == "parse_error"
        assert entries[2].status == "parse_error"
        assert entries[3].status == "success"
        assert entries[4].status == "parse_error"
        assert entries[5].status == "parse_error"
        assert entries[6].status == "parse_error"
        assert entries[7].status == "success"

        assert runner.status == "completed"


class TestResumeCounter:
    @pytest.mark.timeout(120)
    def test_starts_at_exp_004(self, tmp_path: Path) -> None:
        """Pre-create exp_001..exp_003 dirs + 3-entry log -> starts at exp_004."""
        proj = make_project(tmp_path)
        AutoResearchRunner(RunConfig(project_dir=proj, backend=MockBackend([])))
        results_dir = proj / "results"

        for i in range(1, 4):
            (results_dir / f"exp_{i:03d}").mkdir()

        log = ExperimentLog(proj)
        for i in range(1, 4):
            entry = LogEntry(
                exp_id=f"exp_{i:03d}",
                params={"link_capacity": float(i)},
                params_hash=f"hash_{i}",
                status="success",
                metrics={"alpha_star": 1.0},
                objective_score=1.0,
                error_detail=None,
                execution_time_s=1.0,
                seed=42,
                timestamp="2025-01-01T00:00:00+00:00",
            )
            log.append(entry)

        responses = [_mock_response(5.0)]
        backend = MockBackend(responses)
        config = RunConfig(
            project_dir=proj, backend=backend, max_experiments=1, seed=42
        )
        runner = AutoResearchRunner(config)
        runner.run()

        log2 = ExperimentLog(proj)
        entries = log2.load()
        assert len(entries) == 4
        assert entries[3].exp_id == "exp_004"


class TestResumeBest:
    @pytest.mark.timeout(120)
    def test_re_derives_best_from_log(self, tmp_path: Path) -> None:
        """Log has scores [0.5, 0.9, 0.3], stale best_hypothesis.yml says 0.5.
        Runner re-derives best as 0.9 and uses it as the comparison threshold.
        """
        proj = make_project(tmp_path)
        AutoResearchRunner(RunConfig(project_dir=proj, backend=MockBackend([])))
        results_dir = proj / "results"

        for i in range(1, 4):
            (results_dir / f"exp_{i:03d}").mkdir()

        log = ExperimentLog(proj)
        scores = [0.5, 0.9, 0.3]
        for i, score in enumerate(scores, 1):
            entry = LogEntry(
                exp_id=f"exp_{i:03d}",
                params={"link_capacity": float(i)},
                params_hash=f"hash_{i}",
                status="success",
                metrics={"alpha_star": score},
                objective_score=score,
                error_detail=None,
                execution_time_s=1.0,
                seed=42,
                timestamp="2025-01-01T00:00:00+00:00",
            )
            log.append(entry)

        # A stale best_hypothesis.yml must not override the log.
        stale_best = {
            "exp_id": "exp_001",
            "params": {"link_capacity": 1.0},
            "objective_score": 0.5,
            "metrics": {"alpha_star": 0.5},
        }
        with open(proj / "best_hypothesis.yml", "w") as f:
            yaml.dump(stale_best, f)

        # Run one experiment; its measured alpha is compared with the best logged score.
        responses = [_mock_response(6.0)]
        backend = MockBackend(responses)
        config = RunConfig(
            project_dir=proj, backend=backend, max_experiments=1, seed=42
        )
        runner = AutoResearchRunner(config)
        runner.run()

        best_path = proj / "best_hypothesis.yml"
        best_data = yaml.safe_load(best_path.read_text())
        assert best_data["exp_id"] == "exp_004"
        assert best_data["objective_score"] >= 0.9


class TestTimeoutNoResult:
    def test_queue_timeout(self, tmp_path: Path, monkeypatch) -> None:
        from netlab.simulation import SimulationResult

        proj = make_project(tmp_path)
        monkeypatch.setattr(
            "netlab.autoresearch.runner.run_simulation",
            lambda *a, **kw: SimulationResult("timeout", error="deadline exceeded"),
        )
        runner = AutoResearchRunner(
            RunConfig(
                project_dir=proj,
                backend=MockBackend([_mock_response(2.0)]),
                max_experiments=1,
            )
        )
        runner.run()
        entries = ExperimentLog(proj).load()
        assert entries[0].status == "timeout_no_result"
        assert entries[0].error_detail == "deadline exceeded"


class TestCrashStderrCaptured:
    @pytest.mark.timeout(120)
    def test_bad_scenario_captures_stderr(self, tmp_path: Path) -> None:
        """Simulation errors are recorded in the experiment log."""
        proj = make_project(tmp_path)

        # Create a base_scenario that will pass placeholder validation
        # but produces an invalid scenario after merge (bad node reference in link)
        bad_base = textwrap.dedent("""\
            seed: 42
            network:
              nodes:
                N1: {}
              links:
                - source: N1
                  target: NONEXISTENT_NODE
                  capacity: ${{link_capacity}}
                  cost: 1.0
            workflow:
              - type: MaximumSupportedDemand
                name: msd_baseline
                demand_set: baseline_traffic_matrix
                alpha_start: 1.0
                growth_factor: 2.0
                alpha_min: 0.001
                alpha_max: 1000000.0
                resolution: 0.05
                max_bracket_iters: 16
                max_bisect_iters: 32
            demands:
              baseline_traffic_matrix:
                - source: ^N1$
                  target: ^N1$
                  volume: 10.0
                  mode: pairwise
        """)
        (proj / "base_scenario.yml").write_text(bad_base)

        responses = [_mock_response(2.0)]
        backend = MockBackend(responses)
        config = RunConfig(
            project_dir=proj, backend=backend, max_experiments=1, seed=42
        )
        runner = AutoResearchRunner(config)
        runner.run()

        log = ExperimentLog(proj)
        entries = log.load()
        assert len(entries) == 1
        assert entries[0].status == "crash"
        assert entries[0].error_detail is not None
        assert len(entries[0].error_detail) > 0


@pytest.mark.parametrize(
    "changed", ["seed", "objective.yml", "base_scenario.yml", "dependencies"]
)
def test_resume_rejects_changed_research_inputs(tmp_path, monkeypatch, changed):
    proj = make_project(tmp_path)
    config = RunConfig(project_dir=proj, backend=MockBackend([]))
    AutoResearchRunner(config)
    if changed == "seed":
        config.seed += 1
    elif changed == "dependencies":
        monkeypatch.setattr(
            "netlab.autoresearch.runner.package_versions",
            lambda: {"ngraph": "different"},
        )
    else:
        path = proj / changed
        path.write_text(path.read_text() + "\n# changed input\n")
    with pytest.raises(ValueError, match="inputs or dependencies changed"):
        AutoResearchRunner(config)


def test_minimize_selects_zero_and_keeps_distinct_attempt_ids(tmp_path, monkeypatch):
    from netlab.simulation import SimulationResult

    proj = make_project(tmp_path)
    objective = proj / "objective.yml"
    objective.write_text(objective.read_text().replace("maximize", "minimize"))
    values = iter([1.0, 0.0, 2.0])

    def execute(*args, **kwargs):
        return SimulationResult(
            "success",
            results={"steps": {"msd_baseline": {"data": {"alpha_star": next(values)}}}},
        )

    monkeypatch.setattr("netlab.autoresearch.runner.run_simulation", execute)
    backend = MockBackend(
        [_mock_response(2.0), _mock_response(3.0), _mock_response(4.0)]
    )
    runner = AutoResearchRunner(
        RunConfig(
            project_dir=proj,
            backend=backend,
            max_experiments=3,
            reflection_interval=100,
        )
    )
    # A new best triggers reflection; avoid consuming the experiment responses there.
    monkeypatch.setattr(runner, "_run_reflection", lambda entries: None)
    runner.run()
    best = ExperimentLog(proj).best_entry()
    assert best.exp_id == "exp_002"
    assert best.objective_score == 0
    assert (
        yaml.safe_load((proj / "best_hypothesis.yml").read_text())["exp_id"]
        == "exp_002"
    )
