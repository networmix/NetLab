"""One reader for the metric artifacts used by tables and paired comparisons."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def _optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected metric JSON object: {path}")
    return data


def collect_seed_metrics(scenario_dir: Path) -> dict[int, dict[str, float]]:
    """Read current JSON schemas; missing optional artifacts omit their metrics."""
    output = {}
    counts = {}
    counts_file = scenario_dir / "network_stats_summary.csv"
    if counts_file.exists():
        with counts_file.open() as stream:
            counts = {
                int(row["seed"]): {
                    key: float(row[key]) for key in ("node_count", "link_count")
                }
                for row in csv.DictReader(stream)
            }
    for directory in sorted(scenario_dir.glob("seed*")):
        if not directory.is_dir():
            continue
        seed = int(directory.name.removeprefix("seed"))
        alpha = _optional_json(directory / "alpha.json")
        bac = _optional_json(directory / "bac.json")
        latency = _optional_json(directory / "latency.json")
        costs = _optional_json(directory / "costpower.json")
        raw = {
            **counts.get(seed, {}),
            "alpha_star": alpha.get("alpha_star"),
            "auc_norm": bac.get("auc_normalized"),
            **{
                f"bw_p{label}_pct": bac.get("bw_at_probability_pct", {}).get(
                    probability
                )
                for label, probability in (
                    ("90", "90.0"),
                    ("95", "95.0"),
                    ("99", "99.0"),
                    ("999", "99.9"),
                )
            },
            "lat_fail_p99": latency.get("failures", {}).get("p99"),
            **{
                target: latency.get("derived", {}).get(source)
                for source, target in (
                    ("TD99", "lat_TD99"),
                    ("SLO_1_2_drop", "lat_SLO_1_2_drop"),
                    ("best_path_share_drop", "lat_best_path_drop"),
                    ("WES_delta", "lat_WES_delta"),
                )
            },
            **{
                key: costs.get(key)
                for key in (
                    "USD_per_Gbit_offered",
                    "Watt_per_Gbit_offered",
                    "USD_per_Gbit_p999",
                    "Watt_per_Gbit_p999",
                    "capex_total",
                )
            },
        }
        metrics = {
            key: float(value)
            for key, value in raw.items()
            if isinstance(value, (int, float))
        }
        if metrics:
            output[seed] = metrics
    return output


def select_baseline(names: list[str], requested: str | None = None) -> str:
    """Use the same explicit/environment/automatic baseline in all reports."""
    if not names:
        raise ValueError("No scenarios available to select a baseline")
    chosen = requested or os.environ.get("NGRAPH_BASELINE_SCENARIO")
    if chosen:
        if chosen not in names:
            raise ValueError(f"Baseline scenario not found: {chosen}")
        return chosen
    return next(
        (name for name in sorted(names) if "baseline" in name.lower()), sorted(names)[0]
    )
