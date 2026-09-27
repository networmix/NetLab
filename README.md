# NetLab

NetLab analyzes [NetGraph](https://github.com/networmix/NetGraph) network simulations
and runs topology experiments. It computes bandwidth availability, latency stretch,
capacity, and cost metrics. Its autoresearch tools use an LLM to propose scenarios
and interpret simulation results.

## Installation

Requires Python 3.11+, `ngraph >= 0.23.1`, and `netgraph-core >= 0.10.0` (via ngraph).

```bash
pip install netlab
```

For development:

```bash
git clone https://github.com/networmix/NetLab
cd NetLab
make dev
source venv/bin/activate
```

`make dev` installs the package, development dependencies, and pre-commit hooks.

## Metrics

```bash
# Analyze all scenarios in a directory
netlab metrics path/to/scenarios/

# Compute metrics without plots
netlab metrics path/to/scenarios/ --no-plots

# Select scenarios
netlab metrics path/to/scenarios/ --only small_clos,small_dragonfly
```

The metrics CLI expects workflow steps named `msd_baseline`
(`MaximumSupportedDemand`), `tm_placement` (`TrafficMatrixPlacement` with flow
details), and `network_statistics` (`NetworkStats`). `--enable-maxflow` also
requires `node_to_node_capacity_matrix` (`MaxFlow` with flow details).

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
records its hypothesis and status. Status is
`analyzed`, `analysis_incomplete`, `generation_failed`, or `skipped`.
`cycle_log.jsonl` records the cycle summaries. LLM interpretations and suggested
experiments should be assessed against the saved metrics and scenario.

The CLI also supports parameterized projects and DC-BB sweeps:

```bash
netlab autoresearch init --base-scenario scenario.yml --output project/
netlab autoresearch run project/ --backend claude-cli --model sonnet

netlab autoresearch structural-analysis
netlab autoresearch sweep abc1 --output-dir results/
netlab autoresearch cross-sweep --output-dir results/
```

## Development

```bash
make check-ci   # Formatting, lint, type checks, and tests
make check      # Apply pre-commit fixes, then run tests and lint
make test       # Tests with coverage
make lint       # Formatting, lint, and type checks
make qt         # Tests excluding benchmarks, without coverage
```

### Superset workspaces

`.superset/config.json` creates a separate `venv` for each workspace, installs
`.[dev]`, and checks dependencies and imports. Setup copies missing, untracked
root-level `.env` and `.env.*` files from `$SUPERSET_ROOT_PATH`, preserving existing
workspace files and tracked templates. Environment files and
`.superset/config.local.json` are gitignored.

The **Run** button executes `make check-ci`. NetLab has no dev server; setup starts
no services, so teardown has nothing to stop and no ports need allocation.
Setup can be rerun without installing shared Git hooks:

```bash
SUPERSET_ROOT_PATH=/path/to/NetLab bash .superset/workspace.sh setup
bash .superset/workspace.sh check
bash .superset/workspace.sh teardown
```

Merge the configuration into the root checkout's branch to use it for new
workspaces and the project's Run button. Superset reads Run commands from the root
project config. See the [lifecycle documentation](https://docs.superset.sh/setup-teardown-scripts).

### Test local NetGraph sources

```bash
bash dev/check_ngraph_integration.sh ~/ws/NetGraph ~/ws/NetGraph-Core ~/ws/TopoGen
```

The script builds a Core wheel and installs the selected NetGraph and TopoGen
sources in a temporary environment. It runs lint, type checks, and all tests,
including five CLI → TopoGen → NetGraph → metrics pipelines. The pipelines use
cached synthetic geography; they do not download Census data.

Commits, source fingerprints, dependencies, wheel hashes, and test results are
saved under `build/ngraph-integration/`. The gate fails if a source checkout changes
during testing, and removes its temporary environment on exit. Normal workspace
setup uses the dependencies declared in `pyproject.toml`.

See [tests/data/README.md](tests/data/README.md) for fixture regeneration.

## License

[MIT](LICENSE)
