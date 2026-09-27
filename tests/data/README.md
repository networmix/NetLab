# Simulation fixtures

| Fixture | Purpose |
| --- | --- |
| `mini_dcbb.yaml` | Live verification against hand-calculated expectations |
| `mini_dcbb_output/mini_dcbb.results.json` | Analysis-loop tests |
| `../autoresearch/data/square_mesh_results.json` | Autoresearch and metrics tests |
| `scenarios/` | Four topology configurations, each built for seeds 11 and 12 |
| `scenarios_metrics/` | CLI, integration, summary, and plotting references |

Run with the selected NetGraph and TopoGen sources installed in the environment:

```bash
venv/bin/python dev/regenerate_fixtures.py --topogen-commit <verified-SHA>
```

The script requires a clean TopoGen checkout at the specified commit. It builds
`topogen_configs_small` using the four cached integrated graphs, executes
NetGraph, and computes the metrics. It also runs the mini DC-BB and square-mesh
scenarios. All commands finish successfully before the fixtures are replaced.
Logs, input snapshots, and source provenance are saved under
`build/fixture-regeneration/`.

Each small scenario uses 1,000 failure draws plus one baseline. Its workflow
inherits the scenario seed. Demand order is seed-dependent and affects greedy
placement within a priority: alpha is 1.40625 for seed 11 and 1.6875 for seed 12.
The metrics weight each failure pattern by its `occurrence_count`. Timing values
record execution duration and are not performance assertions.

The fixture sources are TopoGen commit
`a3d54a1ea50c9600cde1f1994a3bfd33cc3cbb81` and the NetGraph working tree at base
commit `775530768eec3951b7aec738f7e99d1abb6b7efc`, with tracked diff SHA256
`52501226e5d7167ea2b4ed3311e962b90ef527ec10acaa69be2555298351707a`.
