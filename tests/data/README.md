# Saved networks and metric examples

The tests use these small scenarios and saved results to check simulation,
metric calculation, tables and figures without downloading geographic datasets.
A fixture is one of these saved inputs or expected outputs.

| Fixture | Used for |
| --- | --- |
| `mini_dcbb.yaml` | A small data-center/backbone network with hand-calculated answers |
| `mini_dcbb_output/mini_dcbb.results.json` | Research analysis-loop tests |
| `../autoresearch/data/square_mesh_results.json` | Research and metric calculations on a mesh network |
| `scenarios/` | Baseline, Clos and Dragonfly configurations built with seeds 11 and 12 |
| `scenarios_metrics/` | Expected CLI summaries, tables and figures |

## Regenerate the files

From the repository root, install the dependencies declared in `pyproject.toml`
and run:

```bash
venv/bin/python dev/regenerate_fixtures.py
```

The script uses the saved corridor graphs to build `topogen_configs_small`,
runs NetGraph and computes metrics. It also runs the mini data-center/backbone
and square-mesh scenarios. All simulation tasks must succeed before it replaces
the fixtures.

The script saves logs, a copy of the existing fixtures and input/software records
under `build/fixture-regeneration/`. Editable source checkouts must have no
uncommitted changes; released installations are also accepted. Run regeneration
in a clean checkout rather than discarding work to satisfy that check.

## What the examples exercise

Each small scenario places its traffic under 1,000 sampled failures plus a
no-failure baseline. Its workflow inherits the scenario seed. The seed can change
demand order; when placement processes demands in order, that can also change
the accepted traffic multiplier. In the saved results, all four configurations
have `alpha_star` 1.40625 with seed 11 and 1.6875 with seed 12, for the same
90,000 Gbps base traffic.

Metrics weight each pattern by `occurrence_count`. BAC curves store distinct
bandwidth thresholds and the fraction of samples at or above each threshold.
Repeated observations remain part of the calculation. Saved durations describe
one execution environment and are not performance expectations.

The accompanying `provenance.json` files record package versions, input hashes
and analysis settings. Read them when comparing regenerated outputs. A fixture's
recorded dependency version documents its origin; it does not set the package's
supported versions, which are declared in [pyproject.toml](../../pyproject.toml).
