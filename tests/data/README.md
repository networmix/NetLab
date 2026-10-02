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

The script builds `topogen_configs_small` from the four cached integrated graphs,
executes NetGraph, and computes the metrics. It also runs the mini DC-BB and
square-mesh scenarios. All commands finish successfully before the fixtures are
replaced. Logs, the previous fixtures, and the recorded provenance are saved
under `build/fixture-regeneration/`. Installed sources must be released versions
or clean Git revisions; the script refuses editable checkouts with uncommitted
changes.

Each small scenario uses 1,000 failure draws plus one baseline. Its workflow
inherits the scenario seed. Demand order is seed-dependent and affects greedy
placement within a priority: alpha is 1.40625 for seed 11 and 1.6875 for seed 12.
The metrics weight each failure pattern by its `occurrence_count`. Timing values
record execution duration and are not performance assertions.

Current fixtures were generated with Python 3.13, `ngraph==0.23.1` and
`netgraph-core==0.10.0` from PyPI, and TopoGen commit
`5f7cbbea12d7ec27d108ed5a13c58352c92dcc9d` from `main`.
