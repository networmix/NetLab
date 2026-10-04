# NetLab

NetLab builds and runs [NetGraph](https://github.com/networmix/NetGraph) scenarios
and compares bandwidth availability, latency stretch, capacity, and cost.
It also supports parameter sweeps and LLM-assisted topology experiments.

## Install

Requires Python 3.11+ and Git. NetGraph, TopoGen, and the other dependencies are
installed automatically; exact requirements are in [pyproject.toml](pyproject.toml).

```bash
pip install git+https://github.com/networmix/NetLab
```

For the examples below, clone the repository and install the development environment:

```bash
git clone https://github.com/networmix/NetLab
cd NetLab
make dev
source venv/bin/activate
```

## Run scenarios

This example uses a small, checked-in graph and needs no external geographic data:

```bash
netlab run topogen_configs_small/small_baseline.yml --seeds 11 12 \
  --graphs-dir tests/data/scenarios/small_baseline/small_baseline \
  --scenarios-dir scenarios
netlab metrics scenarios/
```

Scenario YAML and results are saved under
`scenarios/<master>/<master>__seed<N>/`. Metrics, tables, and figures go into
`scenarios_metrics/`; start with `project.csv` and `BAC.png`.

To generate geography from source data, omit `--graphs-dir`:

```bash
netlab run topogen_configs_small --seeds 11 12
netlab build topogen_configs_small --seeds 11 12   # Build without simulating
```

Source datasets and `lib/` are resolved from the working directory. Full-size
configurations are in `topogen_configs/`; reduced versions are in
`topogen_configs_small/`. Supplied graphs must be corridor MultiGraphs named
`<master>_integrated_graph.json`.

- `--build-jobs` and `--run-jobs` set process concurrency.
- `--build-timeout` sets a deadline per graph/build task; `--run-timeout` defaults
  to 600 seconds per simulation. Deadlines start when a task begins executing.
- `--force-run` reruns simulations. `--force` also regenerates geography, unless
  supplied through `--graphs-dir`.

Scenario assembly runs every time. Cached results are reused only when scenario,
package, and result hashes match. Failed tasks return a nonzero CLI status;
independent tasks continue. Each run writes provenance, batch summaries, and
`.run.json` outcomes alongside its artifacts.

## Analyze results

```bash
netlab metrics scenarios/ --no-plots
netlab metrics scenarios/ --only small_clos,small_dragonfly
netlab metrics scenarios/ --summary    # Read saved tables and render figures
netlab test scenarios/ small_baseline small_clos
```

An analysis replaces the complete `scenarios_metrics/` report, including when
`--only` selects a subset. A failed analysis keeps the previous report.

The metrics CLI requires workflow steps named `msd_baseline`
(`MaximumSupportedDemand`), `tm_placement` (`TrafficMatrixPlacement` with flow
details), and `network_statistics` (`NetworkStats`). `--enable-maxflow` adds
pairwise capacity analysis and requires `node_to_node_capacity_matrix`.

See [metrics.md](metrics.md) for metric definitions, required fields, aggregation
rules, and output files. BAC describes the sampled failure experiment; latency
stretch describes delivered traffic relative to its baseline path cost.

## Python API

```python
from pathlib import Path
from netlab.simulation import run_simulation

if __name__ == "__main__":
    outcome = run_simulation(Path("scenario.yml"), timeout=60)
    print(outcome.status, outcome.error)
```

`run_simulation` uses a local process queue and saves successful results.
Use `simulate(yaml_text)` for synchronous execution or `inspect_scenario(yaml_text)`
to validate and count model objects. Scripts using the process queue need the
`__main__` guard. See [the architecture notes](dev/ARCHITECTURE.md) for batch APIs.

Metrics can also be computed directly from a result dictionary:

```python
import json
from netlab.metrics.bac import compute_bac
from netlab.metrics.latency import compute_latency_stretch
from netlab.metrics.msd import compute_alpha_star

with open("scenario.results.json") as f:
    results = json.load(f)

print(compute_alpha_star(results).alpha_star)
print(compute_bac(results, step_name="tm_placement").auc_normalized)
print(compute_latency_stretch(results).failures.get("p99"))
```

## Autoresearch

Create a parameterized project, review its templates, then run experiments:

```bash
netlab autoresearch init --base-scenario scenario.yml --output project/
netlab autoresearch run project/ --backend claude-cli --model sonnet
```

Backends are `mock` (default), `claude-cli`, `codex-cli`, and `openai`.
`HypothesisManager` also accepts a free-text hypothesis: it generates and simulates
a scenario, computes metrics, and requests an interpretation and next experiment.
Cycle artifacts are saved under `<project>/cycles/` for review.

DC-BB layout studies have dedicated commands:

```bash
netlab autoresearch structural-analysis --output layouts.json
netlab autoresearch sweep abc1 --output-dir results/
netlab autoresearch cross-sweep --output-dir results/
```

Research logs are tied to the inputs, seed, and installed dependencies. Use a new
output directory when these change, and one writer per project. Successful sweep
entries are skipped on resume; failed entries are retried. Scores rank higher as
better, and infeasible candidates cannot become best.

Study inputs: [DC-BB research](research_projects/dc-bb-autoresearch/program.md).
Standalone experiments: [DC-BB interconnect](experiments/dc-bb-interconnect/README.md).

## Development

```bash
make check-ci   # Formatting, lint, type checks, and tests
make check      # Apply pre-commit fixes, then run tests and lint
make test       # Tests with coverage
make lint       # Formatting, lint, and type checks
```

The tests include TopoGen → NetGraph → metrics runs using cached geography.
See [fixture regeneration](tests/data/README.md) and
[metric verification](dev/METRICS_VERIFICATION.md).

Superset setup creates a `venv`, installs `.[dev]`, and copies missing untracked
`.env` files from the root checkout. Its Run button executes `make check-ci`.
To run setup manually:

```bash
SUPERSET_ROOT_PATH=/path/to/NetLab bash .superset/workspace.sh setup
```

To test local upstream checkouts:

```bash
bash dev/check_ngraph_integration.sh ~/ws/NetGraph ~/ws/NetGraph-Core ~/ws/TopoGen
```

This builds Core and runs all checks in a temporary environment. Source revisions,
dependency versions, and results are saved under `build/ngraph-integration/`.

## License

[MIT](LICENSE)
