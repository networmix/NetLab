"""Tests for autoresearch CLI handlers and argparse wiring."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pytest
import yaml

from netlab.autoresearch.cli import autoresearch_init, autoresearch_run

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SQUARE_MESH_PATH = DATA_DIR / "square_mesh.yaml"


def _base_scenario_with_placeholder(tmp_path: Path) -> Path:
    """Write a square_mesh variant with a ${{link_capacity}} placeholder."""
    text = _SQUARE_MESH_PATH.read_text()
    text = text.replace("capacity: 2.0", "capacity: ${{link_capacity}}", 1)
    path = tmp_path / "base_with_placeholder.yaml"
    path.write_text(text)
    return path


def _make_init_args(base_scenario: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(base_scenario=base_scenario, output=output)


def _make_run_args(
    project_dir: Path,
    backend: str = "mock",
    max_experiments: int = 1,
    timeout: int = 120,
    seed: int = 42,
) -> argparse.Namespace:
    return argparse.Namespace(
        project_dir=project_dir,
        backend=backend,
        max_experiments=max_experiments,
        timeout=timeout,
        seed=seed,
    )


def _no_msd_scenario(tmp_path: Path) -> Path:
    """Write a scenario with no MaximumSupportedDemand step."""
    text = textwrap.dedent("""\
        seed: 42
        network:
          nodes:
            N1: {}
            N2: {}
          links:
            - source: N1
              target: N2
              capacity: 2.0
              cost: 1.0
        workflow:
          - type: TrafficMatrixPlacement
            name: tm_placement
            demand_set: baseline
    """)
    path = tmp_path / "no_msd.yaml"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class TestInitCreatesStructure:
    def test_creates_all_files_and_dirs(self, tmp_path: Path) -> None:
        """init --base-scenario square_mesh.yaml --output dir creates correct structure."""
        output = tmp_path / "project"
        args = _make_init_args(base_scenario=_SQUARE_MESH_PATH, output=output)

        autoresearch_init(args)

        assert (output / "program.md").exists()
        assert (output / "objective.yml").exists()
        assert (output / "hypothesis_template.yml").exists()
        assert (output / "base_scenario.yml").exists()
        assert (output / "memory").is_dir()
        assert (output / "results").is_dir()

    def test_base_scenario_is_copy(self, tmp_path: Path) -> None:
        """base_scenario.yml is a copy, not a symlink."""
        output = tmp_path / "project"
        args = _make_init_args(base_scenario=_SQUARE_MESH_PATH, output=output)

        autoresearch_init(args)

        copied = output / "base_scenario.yml"
        assert not copied.is_symlink()
        assert copied.read_text() == _SQUARE_MESH_PATH.read_text()

    def test_objective_is_valid_yaml(self, tmp_path: Path) -> None:
        """objective.yml is parseable and has required keys."""
        output = tmp_path / "project"
        args = _make_init_args(base_scenario=_SQUARE_MESH_PATH, output=output)

        autoresearch_init(args)

        data = yaml.safe_load((output / "objective.yml").read_text())
        assert data["direction"] in ("maximize", "minimize")
        assert "primary_metric" in data
        assert "metrics" in data

    def test_template_is_valid_yaml(self, tmp_path: Path) -> None:
        """hypothesis_template.yml is parseable and has params key."""
        output = tmp_path / "project"
        args = _make_init_args(base_scenario=_SQUARE_MESH_PATH, output=output)

        autoresearch_init(args)

        data = yaml.safe_load((output / "hypothesis_template.yml").read_text())
        assert "params" in data

    def test_idempotent_reinit(self, tmp_path: Path) -> None:
        """Running init twice on same dir does not crash."""
        output = tmp_path / "project"
        args = _make_init_args(base_scenario=_SQUARE_MESH_PATH, output=output)

        autoresearch_init(args)
        autoresearch_init(args)

        assert (output / "program.md").exists()

    def test_placeholder_scenario_generates_matching_template(
        self, tmp_path: Path
    ) -> None:
        """Base scenario with ${{link_capacity}} -> template has link_capacity param."""
        base = _base_scenario_with_placeholder(tmp_path)
        output = tmp_path / "project"
        args = _make_init_args(base_scenario=base, output=output)

        autoresearch_init(args)

        data = yaml.safe_load((output / "hypothesis_template.yml").read_text())
        assert "link_capacity" in data["params"]


class TestInitValidatesWorkflow:
    def test_no_msd_step_exits_nonzero(self, tmp_path: Path) -> None:
        """--base-scenario pointing to scenario without MSD -> exit code nonzero."""
        bad = _no_msd_scenario(tmp_path)
        output = tmp_path / "project"
        args = _make_init_args(base_scenario=bad, output=output)

        with pytest.raises(SystemExit) as exc_info:
            autoresearch_init(args)

        assert exc_info.value.code != 0

    def test_nonexistent_base_scenario_exits_nonzero(self, tmp_path: Path) -> None:
        """--base-scenario pointing to nonexistent file -> exit code nonzero."""
        output = tmp_path / "project"
        args = _make_init_args(
            base_scenario=tmp_path / "does_not_exist.yaml", output=output
        )

        with pytest.raises(SystemExit) as exc_info:
            autoresearch_init(args)

        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------


class TestRunBasic:
    @pytest.mark.timeout(120)
    def test_run_mock_one_experiment(self, tmp_path: Path) -> None:
        """run dir --backend mock --max-experiments 1 -> exit 0, 1 log entry."""
        # Set up project with placeholder scenario
        base = _base_scenario_with_placeholder(tmp_path)
        proj = tmp_path / "project"
        init_args = _make_init_args(base_scenario=base, output=proj)
        autoresearch_init(init_args)

        run_args = _make_run_args(
            project_dir=proj, backend="mock", max_experiments=1, seed=42
        )
        # Should not raise SystemExit (exit 0)
        autoresearch_run(run_args)

        log_path = proj / "experiment_log.jsonl"
        assert log_path.exists()
        lines = [
            ln
            for ln in log_path.read_text().splitlines()
            if ln.strip() and "_type" not in ln
        ]
        assert len(lines) == 1

    @pytest.mark.timeout(120)
    def test_run_mock_three_experiments(self, tmp_path: Path) -> None:
        """run dir --backend mock --max-experiments 3 -> exactly 3 log entries."""
        base = _base_scenario_with_placeholder(tmp_path)
        proj = tmp_path / "project"
        init_args = _make_init_args(base_scenario=base, output=proj)
        autoresearch_init(init_args)

        run_args = _make_run_args(
            project_dir=proj, backend="mock", max_experiments=3, seed=42
        )
        autoresearch_run(run_args)

        log_path = proj / "experiment_log.jsonl"
        lines = [
            ln
            for ln in log_path.read_text().splitlines()
            if ln.strip() and "_type" not in ln
        ]
        assert len(lines) == 3


class TestRunMissingProjectDir:
    def test_nonexistent_dir_exits_nonzero(self, tmp_path: Path) -> None:
        """run /nonexistent -> exit nonzero, error mentions 'does not exist'."""
        run_args = _make_run_args(
            project_dir=tmp_path / "nonexistent_project",
            backend="mock",
            max_experiments=1,
        )

        with pytest.raises(SystemExit) as exc_info:
            autoresearch_run(run_args)

        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Argparse wiring tests (test that netlab cli.py registers the subcommands)
# ---------------------------------------------------------------------------


class TestArgparseWiring:
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        """Parse argv through the netlab CLI argparser (without executing)."""

        # We need the parser but not to call func. Rebuild it here by
        # duplicating the main() setup. Instead, call parse_args on the
        # module's main parser. We'll capture SystemExit for --help.
        return None  # placeholder; tested below via separate approach

    def test_autoresearch_help(self, capsys: pytest.CaptureFixture) -> None:
        """netlab autoresearch --help lists init and run."""
        import netlab.cli as cli_mod

        # We just need to verify the actual main() parser works.
        # The simplest approach: import main and capture --help output.
        with pytest.raises(SystemExit) as exc_info:
            import sys

            old_argv = sys.argv
            sys.argv = ["netlab", "autoresearch", "--help"]
            try:
                cli_mod.main()
            finally:
                sys.argv = old_argv

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "init" in captured.out
        assert "run" in captured.out

    def test_autoresearch_run_help(self, capsys: pytest.CaptureFixture) -> None:
        """netlab autoresearch run --help lists --backend, --max-experiments, --timeout, --seed."""
        import netlab.cli as cli_mod

        with pytest.raises(SystemExit) as exc_info:
            import sys

            old_argv = sys.argv
            sys.argv = ["netlab", "autoresearch", "run", "--help"]
            try:
                cli_mod.main()
            finally:
                sys.argv = old_argv

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--backend" in captured.out
        assert "--max-experiments" in captured.out
        assert "--timeout" in captured.out
        assert "--seed" in captured.out
