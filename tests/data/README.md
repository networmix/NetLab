# Simulation fixtures

| Fixture | Purpose |
| --- | --- |
| `mini_dcbb.yaml` | Live verification against hand-calculated expectations |
| `mini_dcbb_output/mini_dcbb.results.json` | Analysis-loop tests |
| `../autoresearch/data/square_mesh_results.json` | Autoresearch and metrics tests |
| `scenarios/` | Four topology configurations, each built for seeds 11 and 12 |
| `scenarios_metrics/` | CLI, integration, summary, and plotting references |

Regenerate from an environment installed with the dependencies declared in
`pyproject.toml`:

```bash
venv/bin/python dev/regenerate_fixtures.py
```

The script builds `topogen_configs_small` from the four cached corridor MultiGraphs,
executes NetGraph, and computes the metrics. It also runs the mini DC-BB and
square-mesh scenarios. All API tasks finish successfully before the fixtures are
replaced. Logs, the previous fixtures, and the recorded provenance are saved
under `build/fixture-regeneration/`. Installed sources must be released versions
or clean Git revisions; the script refuses editable checkouts with uncommitted
changes.

Each small scenario uses 1,000 failure draws plus one baseline. Its workflow
inherits the scenario seed. Demand order is seed-dependent and affects greedy
placement within a priority: alpha is 1.40625 for seed 11 and 1.6875 for seed 12.
The metrics weight each failure pattern by its `occurrence_count`. Timing values
record execution duration and are not performance assertions.

Current fixtures were generated with Python 3.13, `ngraph==0.24.0` and
`netgraph-core==0.11.0` from PyPI, and TopoGen commit
`3bf0fa4aa08604a886421461f7ce36dbd7275121`.

The cached geography uses corridor MultiGraphs. BAC curves store unique thresholds
and inclusive `P(delivered >= threshold)` values; sample quantiles use all
occurrence-weighted draws.
