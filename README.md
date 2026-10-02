# NetLab

NetLab builds and runs [NetGraph](https://github.com/networmix/NetGraph) scenarios
and analyzes their results. It computes bandwidth availability, latency stretch,
capacity, and cost metrics. Its autoresearch tools use an LLM to propose scenarios
and interpret simulation results.

## Installation

Requires Python 3.11+ and Git. Dependencies: `ngraph >= 0.23.1` (which brings
`netgraph-core >= 0.10.0`) and [TopoGen](https://github.com/networmix/TopoGen),
installed from its `main` branch. NetLab is not published on PyPI.

```bash
pip install git+https://github.com/networmix/NetLab
```

For development:

```bash
git clone --recurse-submodules https://github.com/networmix/NetLab
cd NetLab
make dev
source venv/bin/activate
```

`make dev` installs the package, development dependencies, and pre-commit hooks.
The `.claude/skills` submodule provides the NetGraph DSL reference that
autoresearch uses as its generation prompt; without it a short built-in summary
is used.

## Scenarios

`netlab run` builds one scenario per TopoGen master configuration and seed, then
runs `ngraph inspect` and `ngraph run` on each:

```bash
# Build scenarios from topogen_configs/ for two seeds and simulate them
netlab run --seeds 11 12

# Use another configuration directory and output root
netlab run topogen_configs_small --seeds 11 12 --scenarios-dir scenarios

# Build the TopoGen masters only
netlab build topogen_configs_small
```

Outputs go under `scenarios/` by default: the master scenario per configuration,
one `<name>__seed<N>/` directory per seed with the scenario YAML, results JSON,
and inspect and run logs, plus `_run_summaries/*.tsv` and `provenance.json`.
`--force` rebuilds and reruns everything; `--force-run` repeats only the NetGraph
runs. `--topogen-bin` and `--ngraph-bin` select executables; by default NetLab
uses `$NETLAB_TOPOGEN_BIN` and `$NETLAB_NGRAPH_BIN`, then `PATH`, then the current
Python environment. The command exits with a nonzero status when any inspect or
run step fails.

`topogen_configs/` holds the full-size masters and `topogen_configs_small/` the
reduced variants used for the test fixtures. Both select workflow and
failure-policy templates from `lib/` by name.

## Metrics

```bash
# Analyze all scenarios in a directory
netlab metrics scenarios/

# Compute metrics without plots
netlab metrics scenarios/ --no-plots

# Select scenarios
netlab metrics scenarios/ --only small_clos,small_dragonfly

# Print summary tables and render cross-seed figures from existing CSVs
netlab metrics scenarios/ --summary

# Paired t-tests between two scenarios
netlab test scenarios/ small_baseline small_clos
```

Results are written next to the input root as `scenarios_metrics/`. The metrics
CLI expects workflow steps named `msd_baseline` (`MaximumSupportedDemand`),
`tm_placement` (`TrafficMatrixPlacement` with flow details), and
`network_statistics` (`NetworkStats`). `--enable-maxflow` also requires
`node_to_node_capacity_matrix` (`MaxFlow` with flow details).

MSD demand records use `source`, `target`, and `volume`; flow records use `source`,
`destination`, and `demand`. Each failure pattern requires a positive integer
`occurrence_count`, which weights that pattern in the metrics.

| Metric | Measures |
|--------|----------|
| BAC | Distribution of delivered bandwidth, normalized by baseline delivery; aggregate and per direction. |
| Latency stretch | Path cost relative to the least-cost path carrying baseline traffic for that pair, weighted by delivered volume. |
| Alpha (MSD) | Demand multiplier found by the configured capacity search. |
| SPS | Demand-weighted pairwise max-flow capacity under failures; each pair is evaluated independently. |
| Cost/power | Capital cost and power per unit of offered or reliable bandwidth. |
| Iteration statistics | Failure iterations, distinct failure patterns, and execution time. |

See [metrics.md](metrics.md) for formulas, aggregation rules, and output files.

### Python API

```python
import json

from metrics.bac import compute_bac
from metrics.latency import compute_latency_stretch
from metrics.msd import compute_alpha_star

with open("scenario.results.json") as f:
    results = json.load(f)

alpha = compute_alpha_star(results)
bac = compute_bac(results, step_name="tm_placement")
latency = compute_latency_stretch(results)

print(f"alpha_star: {alpha.alpha_star}")
print(f"BAC AUC: {bac.auc_normalized:.4f}")
for direction, flow in bac.per_flow.items():
    print(f"{direction}: AUC={flow.auc_normalized:.4f}")
print(f"failure p99 stretch: {latency.failures.get('p99')}")
```

## Autoresearch

The hypothesis API asks an LLM to write a scenario, checks it with `ngraph inspect`,
and runs the simulation. Failed candidates are retried with error feedback.
NetLab then computes metrics and asks the LLM to interpret them and suggest the
next experiment.

```python
import sys
from pathlib import Path

from netlab.autoresearch.backend import ClaudeCLIBackend
from netlab.autoresearch.hypothesis_manager import HypothesisManager

manager = HypothesisManager(
    project_dir=Path("research"),
    backend=ClaudeCLIBackend(model="sonnet"),
    ngraph_bin=str(Path(sys.executable).parent / "ngraph"),
)
cycle = manager.run_cycle("""
Two sites, three backbone planes, 100 Gbps cross-site per plane.
Internal links: 500 Gbps. Backbone nodes have role: bb.
Demands: 100 Gbps each direction using ECMP.
Failures: one random backbone node per iteration, 20 iterations.
""")
print(cycle.status)
if cycle.analysis is not None:
    print(cycle.analysis.metrics_report)
    print(cycle.analysis.interpretation)
    print(cycle.analysis.next_hypothesis)
```

Cycle artifacts are saved under `research/cycles/`. Successful cycles include the
scenario, simulation results, metrics report, and interpretation. Each cycle
records its hypothesis and status: `analyzed`, `analysis_incomplete`,
`generation_failed`, or `skipped`. `cycle_log.jsonl` records the cycle summaries.
LLM interpretations and suggested experiments should be assessed against the
saved metrics and scenario.

The CLI also supports parameterized projects and DC-BB sweeps. Backends are
`mock` (default), `claude-cli`, `codex-cli`, and `openai`:

```bash
netlab autoresearch init --base-scenario scenario.yml --output project/
netlab autoresearch run project/ --backend claude-cli --model sonnet

netlab autoresearch structural-analysis --output layouts.json
netlab autoresearch sweep abc1 --output-dir results/
netlab autoresearch cross-sweep --output-dir results/
```

The DC-BB study inputs live in `research_projects/dc-bb-autoresearch/`; standalone
topology experiments live in `experiments/dc-bb-interconnect/`.

## Development

```bash
make check-ci   # Formatting, lint, type checks, and tests
make check      # Apply pre-commit fixes, then run tests and lint
make test       # Tests with coverage
make lint       # Formatting, lint, and type checks
make qt         # Tests excluding benchmarks, without coverage
```

The test suite includes TopoGen → NetGraph → metrics pipeline checks on a tiny
cached geography; they need no Census data. [tests/data/README.md](tests/data/README.md)
describes the checked-in fixtures and how to regenerate them.

### Superset workspaces

`.superset/config.json` configures [Superset](https://docs.superset.sh/setup-teardown-scripts)
workspaces. Setup checks out the skills submodule, creates a `venv`, installs
`.[dev]`, verifies imports, and copies missing untracked `.env` and `.env.*` files
from the root checkout. The Run button executes `make check-ci`. There is no dev
server, so teardown is a no-op and no ports are allocated. The same steps run
manually:

```bash
SUPERSET_ROOT_PATH=/path/to/NetLab bash .superset/workspace.sh setup
bash .superset/workspace.sh check
```

### Test local NetGraph sources

```bash
bash dev/check_ngraph_integration.sh ~/ws/NetGraph ~/ws/NetGraph-Core ~/ws/TopoGen
```

The script builds a Core wheel, installs the selected NetGraph and TopoGen
checkouts in a temporary environment, and runs lint, type checks, and the full
test suite against them. Commits, source fingerprints, dependencies, wheel hashes,
and results are saved under `build/ngraph-integration/`. The gate fails if a
checkout changes during the run. Normal installs use the dependencies declared in
`pyproject.toml`.

## License

[MIT](LICENSE)
