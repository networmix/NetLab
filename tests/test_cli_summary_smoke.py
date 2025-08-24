from __future__ import annotations

from pathlib import Path

from netlab.metrics_cmd import print_summary_from_csv


def test_print_summary_from_csv_smoke(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    # Should print consolidated and possibly normalized tables from the reference metrics dir
    print_summary_from_csv(repo_root / "scenarios", plots=False)
    out = capsys.readouterr().out
    assert "Consolidated project metrics" in out
