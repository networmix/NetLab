from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _run_cli(argv: list[str]) -> int:
    import netlab.cli as cli

    old = sys.argv[:]
    try:
        sys.argv = ["netlab"] + argv
        try:
            cli.main()
            return 0
        except SystemExit as e:  # die() uses sys.exit
            return int(e.code) if e.code is not None else 0
    finally:
        sys.argv = old


def test_cli_build_no_yamls_exits_early(tmp_path: Path, capsys) -> None:
    code = _run_cli(["build", str(tmp_path)])
    captured = capsys.readouterr()
    assert code != 0
    assert "No YAMLs under" in captured.err


def test_cli_run_no_yamls_exits_early(tmp_path: Path, capsys) -> None:
    code = _run_cli(["run", str(tmp_path), "--seeds", "11", "12"])
    captured = capsys.readouterr()
    assert code != 0
    assert "No YAMLs under" in captured.err


@pytest.mark.parametrize(
    "inspect_status, run_status", [("❌", "⏭️ skipped"), ("✅", "❌")]
)
def test_cli_run_reports_ngraph_failure(
    tmp_path, monkeypatch, capsys, inspect_status, run_status
):
    import netlab.cli as cli

    master = tmp_path / "scenario.yml"
    master.write_text("{}")
    monkeypatch.setattr(cli, "_generate_masters", lambda **kwargs: [])
    monkeypatch.setattr(cli, "_build_seed_scenario", lambda *args: (master, 0))
    monkeypatch.setattr(
        cli, "_inspect_run_one", lambda *args: (master, inspect_status, run_status)
    )
    output = tmp_path / "scenarios"
    code = _run_cli(
        ["run", str(master), "--seeds", "42", "--scenarios-dir", str(output)]
    )
    assert code != 0
    assert "NetGraph failed" in capsys.readouterr().err
    assert (output / "_run_summaries/scenario.tsv").is_file()
    assert (output / "provenance.json").is_file()


def test_cli_metrics_summary_smoke(capsys) -> None:
    # Uses bundled test data under tests/data
    data_root = Path(__file__).resolve().parent / "data"
    scen = data_root / "scenarios"
    code = _run_cli(["metrics", str(scen), "--summary"])  # print-only path
    out = capsys.readouterr().out
    assert code == 0
    assert "Consolidated project metrics" in out


def test_cli_metrics_run_on_temp_copy(tmp_path: Path) -> None:
    # Copy scenarios from tests/data into a temp dir and run metrics without plots
    data_root = Path(__file__).resolve().parent / "data"
    src = data_root / "scenarios"
    dst = tmp_path / "scenarios"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*.results.json"):
        target = dst / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    code = _run_cli(["metrics", str(dst), "--no-plots"])  # compute path
    assert code == 0
    # Verify outputs exist
    out_root = tmp_path / "scenarios_metrics"
    assert (out_root / "project.csv").exists()


def test_cli_test_subcommand_smoke(capsys) -> None:
    data_root = Path(__file__).resolve().parent / "data"
    scen = data_root / "scenarios"
    code = _run_cli(["test", str(scen), "small_baseline", "small_clos"])  # no crash
    out = capsys.readouterr().out
    assert code == 0
    # Either insight table or absence message paths are acceptable
    assert (
        ("Paired t-tests" in out)
        or ("No common-seed paired results" in out)
        or ("no project insights available" in out)
    )
