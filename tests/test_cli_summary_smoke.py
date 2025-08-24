from __future__ import annotations

from pathlib import Path

from netlab.metrics_cmd import print_summary_from_csv


def test_print_summary_from_csv_smoke(capsys) -> None:
    data_root = Path(__file__).resolve().parent / "data"
    # Should print consolidated and possibly normalized tables from the bundled metrics dir
    print_summary_from_csv(data_root / "scenarios", plots=False)
    out = capsys.readouterr().out
    assert "Consolidated project metrics" in out
