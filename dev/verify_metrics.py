#!/usr/bin/env python3
"""Run NetGraph experiments and compare metrics with independent calculations.

Run: venv/bin/python dev/verify_metrics.py --output build/metrics-verification
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from netlab.metrics.analysis import analyze_one_seed
from netlab.metrics.bac import compute_bac
from netlab.metrics.latency import compute_latency_stretch
from netlab.metrics.matrixdump import compute_pair_matrices
from netlab.metrics.sps import compute_sps
from netlab.simulation import simulate

EDGES = [
    ("a/dc/node", "u", 5, 1),
    ("u", "b/dc/node", 5, 1),
    ("a/dc/node", "v", 3, 2),
    ("v", "b/dc/node", 3, 2),
]


def scenario_document(mask: int = 0, seed: int = 42) -> dict[str, Any]:
    nodes = {
        name: {"attrs": {"hardware": {"component": "router", "count": 1}}}
        for name in ["a/dc/node", "u", "v", "b/dc/node"]
    }
    links = [
        dict(
            source=s,
            target=t,
            capacity=cap,
            cost=cost,
            attrs={
                "selected": bool(mask & (1 << i)),
                "hardware": {
                    side: {"component": "optic", "count": 1}
                    for side in ["source", "target"]
                },
            },
        )
        for i, (s, t, cap, cost) in enumerate(EDGES)
    ]
    failure = {"failure_policy": "selected"}
    workflow = [
        dict(
            type="MaximumSupportedDemand",
            name="msd_baseline",
            demand_set="traffic",
            resolution=0.0001,
        ),
        dict(
            type="TrafficMatrixPlacement",
            name="tm_placement",
            demand_set="traffic",
            iterations=1,
            parallelism=1,
            seed=seed,
            include_flow_details=True,
            alpha_from_step="msd_baseline",
            alpha_from_field="data.alpha_star",
            **failure,
        ),
        dict(
            type="MaxFlow",
            name="node_to_node_capacity_matrix",
            source="^(a/dc/node)$",
            target="^(b/dc/node)$",
            mode="pairwise",
            iterations=1,
            parallelism=1,
            seed=seed,
            include_flow_details=True,
            **failure,
        ),
        dict(type="CostPower", name="cost_power"),
        dict(type="NetworkStats", name="network_statistics"),
    ]
    return dict(
        seed=seed,
        network=dict(nodes=nodes, links=links),
        components={
            "router": dict(component_type="chassis", capex=100, power_watts=10),
            "optic": dict(component_type="optic", capex=2, power_watts=1),
        },
        demands={
            "traffic": [
                dict(
                    source="^a/dc/node$",
                    target="^b/dc/node$",
                    volume=8,
                    mode="pairwise",
                    flow_policy="TE_WCMP_UNLIM",
                )
            ]
        },
        failures={
            "selected": {
                "modes": [
                    {
                        "weight": 1,
                        "rules": [
                            dict(
                                scope="link",
                                mode="all",
                                match={
                                    "conditions": [
                                        dict(attr="selected", op="==", value=True)
                                    ]
                                },
                            )
                        ],
                    }
                ]
            }
        },
        workflow=workflow,
    )


def native(doc: dict[str, Any]) -> dict[str, Any]:
    outcome = simulate(yaml.safe_dump(doc))
    if not outcome.success:
        raise AssertionError(outcome.error)
    return outcome.results


def assert_close(actual: Any, expected: Any) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def verify_exhaustive(output: Path) -> list[dict[str, Any]]:
    records, rows = [], []
    for mask in range(16):
        raw = native(scenario_document(mask))
        (output / f"diamond-mask-{mask:02d}.json").write_text(
            json.dumps(raw, indent=2) + "\n"
        )
        graph = nx.DiGraph()
        graph.add_nodes_from(["a/dc/node", "u", "v", "b/dc/node"])
        for i, (s, t, cap, _cost) in enumerate(EDGES):
            if not mask & (1 << i):
                graph.add_edge(s, t, capacity=cap)
                graph.add_edge(t, s, capacity=cap)
        capacity = nx.maximum_flow_value(graph, "a/dc/node", "b/dc/node")
        short = 5 if not mask & 3 else 0
        long = 3 if not mask & 12 else 0
        assert capacity == short + long
        alpha, bac, mf, latency, cp, iterations, sps = analyze_one_seed(
            raw, output, False, enable_maxflow=True
        )
        assert_close(alpha.alpha_star, 1)
        assert_close(alpha.base_total_demand, 8)
        assert_close(bac.series, [8, capacity])
        assert_close(bac.auc_normalized, (1 + capacity / 8) / 2)
        assert_close(bac.bw_at_probability_abs[99.9], capacity)
        assert mf is not None and sps is not None
        assert_close(mf.series, [8, capacity])
        assert sps is not None
        assert_close(sps.series, [capacity / 8])
        assert_close([cp.capex_total, cp.power_total_w], [416, 48])
        assert_close(cp.per_metro["capex_total"].sum(), 416)
        assert_close(cp.per_metro["power_total_watts"].sum(), 48)
        assert_close(cp.per_offered_demand__usd_per_gbit, 52)
        if capacity:
            assert_close(cp.per_reliable_p999__usd_per_gbit, 416 / capacity)
            assert_close(latency.failures["p99"], 2 if long else 1)
            assert_close(latency.failures["WES"], long / capacity)
            assert_close(latency.failures["SLO_1_2"], short / capacity)
        else:
            assert cp.per_reliable_p999__usd_per_gbit is None
            assert np.isnan(latency.failures["p99"])
        assert iterations.failures_count == iterations.unique_patterns == 1
        tm_abs, tm_norm, mf_abs, mf_norm = compute_pair_matrices(raw, True)
        assert_close(tm_abs.to_numpy(), capacity)
        assert_close(tm_norm.to_numpy(), capacity / 8)
        assert mf_abs is not None and mf_norm is not None
        assert_close(mf_abs.to_numpy(), capacity)
        assert_close(mf_norm.to_numpy(), capacity / 8)
        rows.append(
            dict(
                mask=mask,
                expected_capacity=capacity,
                actual_capacity=bac.series.iloc[1],
                auc=bac.auc_normalized,
            )
        )
        records.append(raw)
    # Exact weighted population built from native outcomes; compare compressed vs expanded.
    combined = deepcopy(records[0])
    expanded = deepcopy(combined)
    for name in ["tm_placement", "node_to_node_capacity_matrix"]:
        patterns = []
        for mask, raw in enumerate(records):
            pattern = deepcopy(raw["steps"][name]["data"]["flow_results"][0])
            pattern["occurrence_count"] = mask + 1
            patterns.append(pattern)
        combined["steps"][name]["data"]["flow_results"] = patterns
        expanded["steps"][name]["data"]["flow_results"] = [
            dict(p, occurrence_count=1)
            for p in patterns
            for _ in range(p["occurrence_count"])
        ]
    assert (
        compute_bac(combined, "tm_placement").to_jsonable()
        == compute_bac(expanded, "tm_placement").to_jsonable()
    )
    assert compute_sps(combined).to_jsonable() == compute_sps(expanded).to_jsonable()
    latency_combined = compute_latency_stretch(combined).per_iteration
    latency_expanded = compute_latency_stretch(expanded).per_iteration
    assert latency_combined is not None and latency_expanded is not None
    np.testing.assert_allclose(
        latency_combined["p99"],
        latency_expanded["p99"],
        equal_nan=True,
    )
    (output / "native-summary.json").write_text(
        json.dumps(
            {"exhaustive_masks": rows, "weighted_failure_samples": 136}, indent=2
        )
        + "\n"
    )
    return records


def verify_variants(output: Path) -> dict[str, dict[str, Any]]:
    outcomes = {}
    # Priority classes contend for the same three units after the short path fails.
    doc = scenario_document(1)
    template = doc["demands"]["traffic"][0]
    doc["demands"]["traffic"] = [
        dict(template, volume=5, priority=0),
        dict(template, volume=3, priority=1),
    ]
    raw = native(doc)
    alpha, bac, _, latency, _, _, sps = analyze_one_seed(
        raw, output, False, enable_maxflow=True
    )
    assert_close(alpha.alpha_star, 1)
    assert_close(bac.series, [8, 3])
    assert sps is not None
    assert_close(sps.series, [3 / 8])
    assert_close(latency.baseline["p99"], 2)
    classes = raw["steps"]["tm_placement"]["data"]["flow_results"][0]["flows"]
    assert {r["priority"]: r["placed"] for r in classes} == {0: 3, 1: 0}
    outcomes["priorities"] = raw

    # Regex expansion splits the given volume across pairs; it does not multiply it.
    doc = scenario_document(0)
    doc["demands"]["traffic"][0]["source"] = "^(u|v)$"
    doc["workflow"][2]["source"] = "^(u|v)$"
    raw = native(doc)
    alpha, bac, _, _, _, _, sps = analyze_one_seed(
        raw, output, False, enable_maxflow=True
    )
    flows = raw["steps"]["tm_placement"]["data"]["baseline"]["flows"]
    assert len(flows) == 2
    assert_close(sum(r["demand"] for r in flows), alpha.alpha_star * 8)
    assert_close(alpha.alpha_star, 1)
    assert_close(bac.offered, 8)
    assert sps is not None
    assert_close(sps.series, [1])
    outcomes["pairwise_expansion"] = raw

    # Combine-mode pseudo endpoints must remain distinct and fully supported.
    doc = scenario_document(1)
    doc["demands"]["traffic"][0]["mode"] = "combine"
    raw = native(doc)
    _, bac, _, latency, _, _, _ = analyze_one_seed(raw, output, False)
    assert_close(bac.series, [8, 3])
    assert_close(latency.failures["p99"], 2)
    outcomes["combine"] = raw

    # Opposite directions use independent link capacities.
    doc = scenario_document(1)
    doc["demands"]["traffic"].append(
        dict(doc["demands"]["traffic"][0], source="^b/dc/node$", target="^a/dc/node$")
    )
    doc["workflow"][2].update(source="^([ab]/dc/node)$", target="^([ab]/dc/node)$")
    raw = native(doc)
    _, bac, _, _, _, _, sps = analyze_one_seed(raw, output, False, enable_maxflow=True)
    assert_close(bac.series, [16, 6])
    assert len(bac.per_flow) == 2
    assert sps is not None
    assert_close(sps.series, [3 / 8])
    outcomes["bidirectional"] = raw

    # Search-limited alpha is a reported bound, not a claim of true network maximum.
    doc = scenario_document(0)
    doc["workflow"][0]["alpha_max"] = 0.5
    raw = native(doc)
    alpha, bac, _, _, cp, _, _ = analyze_one_seed(raw, output, False)
    assert_close(alpha.alpha_star, 0.5)
    assert_close(bac.offered, 4)
    assert_close(cp.per_offered_demand__usd_per_gbit, 104)
    outcomes["bounded_msd"] = raw

    # No failure policy means a baseline-only native result.
    doc = scenario_document(0)
    for step in doc["workflow"]:
        step.pop("failure_policy", None)
    raw = native(doc)
    _, bac, _, _, _, operations, sps = analyze_one_seed(
        raw, output, False, enable_maxflow=True
    )
    assert bac.series.tolist() == [8]
    assert operations.failures_count == 0 and operations.total_iterations_count == 1
    assert sps is not None and sps.series.empty
    outcomes["baseline_only"] = raw

    # The engine refuses MSD when no positive feasible alpha exists.
    doc = scenario_document(0)
    for link in doc["network"]["links"]:
        link["disabled"] = True
    disconnected = simulate(yaml.safe_dump(doc))
    assert not disconnected.success and "No feasible alpha" in disconnected.error
    (output / "disconnected-msd.json").write_text(
        json.dumps(
            {"status": disconnected.status, "error": disconnected.error}, indent=2
        )
        + "\n"
    )
    for name, raw in outcomes.items():
        (output / f"variant-{name}.json").write_text(json.dumps(raw, indent=2) + "\n")
    return outcomes


def verify_multiseed(output: Path, plots: bool = False) -> Path:
    import pandas as pd
    from scipy import stats

    from netlab.metrics.batch import run_metrics
    from netlab.metrics.comparisons import compare_scenarios
    from netlab.metrics.reporting import print_summary_from_csv

    root = output / "multiseed"
    root.mkdir(exist_ok=True)
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for scenario, long_capacity in [("baseline", 3), ("upgraded", 6), ("lean", 2)]:
        for index, seed in enumerate(range(101, 106)):
            doc = scenario_document(0, seed)
            for link in doc["network"]["links"][2:]:
                link["capacity"] = long_capacity
            doc["failures"]["selected"]["modes"][0]["rules"][0].update(
                mode="choice", count=1, match={"conditions": []}
            )
            iterations = 9 + index * 10
            for step in doc["workflow"][1:3]:
                step["iterations"] = iterations
            raw = native(doc)
            (root / f"{scenario}__seed{seed}_scenario.results.json").write_text(
                json.dumps(raw, indent=2) + "\n"
            )
            total = 5 + long_capacity
            delivered, stretch = [total], []
            patterns = raw["steps"]["tm_placement"]["data"]["flow_results"]
            for pattern in patterns:
                failed = pattern["failure_state"]["excluded_links"]
                assert len(failed) == 1
                lost_short = "u" in failed[0].split("|")
                value = long_capacity if lost_short else 5
                count = pattern["occurrence_count"]
                delivered.extend([value] * count)
                stretch.extend([2 if lost_short else 1] * count)
                assert_close(pattern["summary"]["total_placed"], value)
            assert len(stretch) == iterations

            def bw(percent, delivered=delivered):
                return max(
                    v
                    for v in set(delivered)
                    if Fraction(sum(x >= v for x in delivered), len(delivered))
                    >= Fraction(str(percent)) / 100
                )

            mid = statistics.median(stretch)
            bw999 = bw(99.9)
            expected[scenario, seed] = dict(
                alpha_star=total / 8,
                bac_auc=statistics.mean(delivered) / total,
                bw_p90=bw(90) / total,
                bw_p95=bw(95) / total,
                bw_p99=bw(99) / total,
                bw_p999=bw999 / total,
                lat_base_p50=1 if 5 / total >= 0.5 else 2,
                lat_fail_p99=mid,
                lat_TD99=mid / 2,
                lat_SLO_1_2_drop=5 / total - (2 - mid),
                lat_best_path_drop=5 / total - (2 - mid),
                lat_WES_delta=(mid - 1) - long_capacity / total,
                iters_fail=iterations,
                iters_total=iterations + 1,
                unique_patterns=len(patterns),
                node_count=4,
                link_count=4,
                capex_total=416,
                USD_per_Gbit_offered=416 / total,
                Watt_per_Gbit_offered=48 / total,
                USD_per_Gbit_p999=416 / bw999,
                Watt_per_Gbit_p999=48 / bw999,
                samples=[x / total for x in delivered],
            )
    log = io.StringIO()
    with redirect_stdout(log):
        run_metrics(root, no_plots=not plots, enable_maxflow=True)
        if plots:
            print_summary_from_csv(root, plots=True, quiet=True)
    (output / "multiseed.log").write_text(log.getvalue())
    report = output / "multiseed_metrics"
    project = pd.read_csv(report / "project.csv").set_index("scenario")
    normalized = pd.read_csv(report / "project_baseline_normalized.csv").set_index(
        "scenario"
    )
    rename = {
        "bac_auc": "auc_norm",
        "bw_p90": "bw_p90_pct",
        "bw_p95": "bw_p95_pct",
        "bw_p99": "bw_p99_pct",
        "bw_p999": "bw_p999_pct",
    }
    ratio_metrics = [
        "bac_auc",
        "bw_p90",
        "bw_p95",
        "bw_p99",
        "bw_p999",
        "lat_fail_p99",
        "USD_per_Gbit_offered",
        "Watt_per_Gbit_offered",
        "USD_per_Gbit_p999",
        "Watt_per_Gbit_p999",
    ]
    for scenario in ["baseline", "upgraded", "lean"]:
        seed_rows = [expected[scenario, seed] for seed in range(101, 106)]
        for key in seed_rows[0]:
            if key == "samples":
                continue
            assert_close(
                project.loc[scenario, key],
                statistics.median(row[key] for row in seed_rows),
            )
        for key in ratio_metrics:
            ratios = [
                expected[scenario, seed][key] / expected["baseline", seed][key]
                for seed in range(101, 106)
            ]
            assert_close(
                normalized.loc[scenario, rename.get(key, key) + "_r"],
                statistics.median(ratios),
            )
        for key in ["lat_SLO_1_2_drop", "lat_best_path_drop", "lat_WES_delta"]:
            differences = [
                expected[scenario, seed][key] - expected["baseline", seed][key]
                for seed in range(101, 106)
            ]
            assert_close(
                normalized.loc[scenario, key + "_d"], statistics.median(differences)
            )
        summary = json.loads((report / scenario / "bac_summary.json").read_text())
        pooled = [sample for row in seed_rows for sample in row["samples"]]
        assert_close(summary["pooled_tail"]["auc_norm"], statistics.mean(pooled))
        for x, probability in zip(
            summary["pooled_grid"]["x_pct"],
            summary["pooled_grid"]["availability"],
            strict=True,
        ):
            assert_close(
                probability, sum(sample * 100 >= x for sample in pooled) / len(pooled)
            )
        for x, q25, q75 in zip(
            summary["pooled_iqr"]["x_pct"],
            summary["pooled_iqr"]["a_q25"],
            summary["pooled_iqr"]["a_q75"],
            strict=True,
        ):
            values = sorted(
                sum(sample * 100 >= x for sample in row["samples"])
                / len(row["samples"])
                for row in seed_rows
            )
            # Five seed observations: linear 25th/75th percentiles are ranks 1 and 3.
            assert_close([q25, q75], [values[1], values[3]])
    comparisons = compare_scenarios(report, scenarios=("upgraded", "baseline"))
    for record in comparisons:
        metric = str(record["metric"])
        source = {"bw_p999_pct": "bw_p999"}.get(metric, metric)
        a = np.array([expected["upgraded", seed][source] for seed in range(101, 106)])
        b = np.array([expected["baseline", seed][source] for seed in range(101, 106)])
        assert record["n"] == 5
        assert_close(record["mean_diff"], statistics.mean((a - b).tolist()))
        if not np.all(a - b == (a - b)[0]):
            assert_close(record["p"], stats.ttest_rel(a, b).pvalue)
    (output / "multiseed-expected.json").write_text(
        json.dumps(
            {f"{scenario}/{seed}": row for (scenario, seed), row in expected.items()},
            indent=2,
        )
        + "\n"
    )
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("build/metrics-verification")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    verify_exhaustive(args.output)
    verify_variants(args.output)
    verify_multiseed(args.output, args.plots)
    print(
        "Verified all 16 edge failure subsets and 136 weighted native samples, plus seven native model variants."
    )


if __name__ == "__main__":
    main()
