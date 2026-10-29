"""Physical quantities and random sampling on the mini DC-BB network."""

from pathlib import Path

import pytest
import yaml

from netlab.metrics.bac import compute_bac
from netlab.metrics.iterops import compute_iter_ops
from netlab.metrics_failure import extract_failure_ratios, extract_network_stats
from netlab.results import baseline_case, require_step, rows, total_bandwidth
from netlab.simulation import simulate
from tests.result_samples import timing_record


def mini_document():
    return yaml.safe_load((Path(__file__).parent / "data/mini_dcbb.yaml").read_text())


@pytest.mark.timeout(60)
def test_capacity_units_routing_and_physical_graph():
    document = mini_document()
    capacity = {
        "type": "MaxFlow",
        "name": "capacity",
        "source": "^abc1/rsw$",
        "target": "^xyz1/rsw$",
        "workers": 1,
        "iterations": 1,
        "outputs": {"edge_flows": True, "min_cut": True},
    }
    document["workflow"] = [
        {"type": "NetworkStats", "name": "network_statistics"},
        {"type": "BuildGraph", "name": "build_graph"},
        capacity,
        dict(capacity, name="shortest_capacity", preset="SHORTEST_PATHS_ECMP"),
    ]
    outcome = simulate(yaml.safe_dump(document))
    assert outcome.success, outcome.error
    result = outcome.results
    # L=10+2*12=34; Gamma=L/4+2=10.5. With c_min=50 and p=.001,
    # the split bound is1/210, so the largest permitted power of two is1/256.
    assert result["units_report"]["q"] == 1 / 256
    unrestricted = baseline_case(require_step(result, "capacity"))
    shortest = baseline_case(require_step(result, "shortest_capacity"))
    assert rows(unrestricted)[0]["result"]["delivered_units"] == 76800
    # Two planes, each with parallel capacities 100+50; shortest-path ECMP uses 100 only.
    assert total_bandwidth(result, unrestricted) == 300
    assert total_bandwidth(result, shortest) == 200
    assert extract_network_stats(result) == {
        "node_count": 10,
        "link_count": 12,
        "total_capacity": 4300,
    }
    assert sum(link["capacity"] for link in document["network"]["links"]) == 4300
    graph = require_step(result, "build_graph")["result"]["graph"]
    assert len(graph["edges"]) == 24
    assert sum(edge["capacity"] for edge in graph["edges"]) == 8600
    cross_site = [
        edge
        for edge in rows(unrestricted)[0]["result"]["edge_flows"]
        if edge["edge"]["link"].startswith("bb/abc1/pl")
        and "|bb/xyz1/" in edge["edge"]["link"]
        and edge["edge"]["direction"] == "forward"
    ]
    assert len(cross_site) == 4
    assert (
        sum(edge["units"] for edge in cross_site) * result["units_report"]["q"] == 300
    )


@pytest.mark.timeout(60)
def test_sampled_empty_draws_and_unsampled_reference():
    document = mini_document()
    document["failures"] = {
        "bb_random": {
            "modes": [
                {
                    "weight": 1,
                    "rules": [
                        {
                            "scope": "node",
                            "mode": "random",
                            "probability": 0.1,
                            "match": {
                                "conditions": [
                                    {"attr": "role", "op": "==", "value": "bb"}
                                ]
                            },
                        }
                    ],
                }
            ]
        }
    }
    document["workflow"] = document["workflow"][:2]
    document["workflow"][1].update(
        name="tm_placement",
        failure_policy="bb_random",
        iterations=100,
        workers=1,
        seed=42,
    )
    outcome = simulate(yaml.safe_dump(document))
    assert outcome.success, outcome.error
    result = outcome.results
    step = require_step(result, "tm_placement")
    baseline = baseline_case(step)
    assert baseline["case"]["occurrences"] == 71
    assert baseline["states"]["converged"]["totals"]["delivered_units"] == 153600
    assert total_bandwidth(result, baseline) == 600
    bac = compute_bac(result, "tm_placement")
    assert len(bac.series) == 101 and bac.auc_normalized == pytest.approx(169 / 202)
    samples = extract_failure_ratios(result)["tm_placement"]
    assert len(samples) == 100 and sum(samples) == 83.5
    counts = compute_iter_ops(result, outcome.run_record())
    assert (
        counts.failures_count,
        counts.total_iterations_count,
        counts.unique_patterns,
    ) == (100, 101, 9)


def test_rounded_requests_keep_the_declared_offer_denominator(tmp_path):
    from netlab.metrics.analysis import analyze_one_seed
    from tests.result_samples import exported, msd, placement, row

    result = exported(
        msd(1, (0.3,)),
        placement([row(delivered=1, requested=1)], baseline_count=1, policy=None),
        {
            "name": "cost_power",
            "step_type": "CostPower",
            "result": {
                "levels": [
                    {"level": 0, "rows": [{"capex_total": 3, "power_total_watts": 6}]}
                ]
            },
        },
        q=0.25,
    )
    alpha, bac, _, _, cost, _, _ = analyze_one_seed(
        result, tmp_path, False, run_record=timing_record(result)
    )
    assert alpha.base_total_demand == 0.3
    assert bac.offered == 0.25
    assert cost.per_offered_demand__usd_per_gbit == 10
    assert cost.per_offered_demand__watt_per_gbit == 20
    assert cost.per_reliable_p999__usd_per_gbit == 12
    assert cost.per_reliable_p999__watt_per_gbit == 24
