# NetLab

NetLab builds network scenarios with
[NetLab-TopoGen](https://github.com/networmix/NetLab-TopoGen), runs them with
[NetGraph](https://github.com/networmix/NetGraph), and compares the results. It
reports how much traffic survives failures, how path costs change, and how much
the modeled equipment costs and consumes.

You can run several configurations and random seeds in one batch. NetLab also
supports parameter searches and experiments proposed by a language model.

## Install

Use Python 3.11 or later and Git. To install the command:

```bash
python -m pip install git+https://github.com/networmix/NetLab
```

NetLab is installed from GitHub because the name `netlab` on PyPI belongs to
another project. The command also installs NetGraph (`ngraph`), NetLab-TopoGen
(`netlab-topogen`) and the other dependencies from PyPI;
[pyproject.toml](pyproject.toml) lists their supported versions.

To use the included examples, clone the repository and create its environment:

```bash
git clone https://github.com/networmix/NetLab
cd NetLab
make dev
source venv/bin/activate
```

Run the commands below from the repository root.

## Run an example

This example uses saved geographic routes, so no Census download is needed. It
still needs network access: the build draws a site map for each scenario from
downloaded map tiles. To build offline, set `visualization.site_graph.export:
false` in a copy of the configuration.

```bash
netlab run topogen_configs_small/small_baseline.yml --seeds 11 12 \
  --graphs-dir tests/data/scenarios/small_baseline/small_baseline \
  --scenarios-dir scenarios
netlab metrics scenarios/
```

The first command builds a network for each seed and runs its NetGraph workflow.
A **seed** is the scenario's random seed. It drives NetLab-TopoGen's random traffic
choices and NetGraph's failure sampling, so the same seed repeats the same
experiment. The second command calculates metrics and renders comparison figures.

| Location | Contents |
| --- | --- |
| `scenarios/<config>/<config>__seed<N>/` | Scenario YAML, NetGraph results and NetLab's run record |
| `scenarios/<config>/<config>_integrated_graph.json` | Geography generated when `--graphs-dir` is omitted |
| `scenarios/provenance.json` | Input hashes, settings, package versions and outcomes of the last batch |
| `scenarios_metrics/project.csv` | One summary row per configuration |
| `scenarios_metrics/BAC.png` | How often each delivered bandwidth was met or exceeded |
| `scenarios_metrics/<config>/seed<N>/` | Metrics for one seed |

To build without simulating, use `netlab build` with the same input arguments.
To generate geography yourself, omit `--graphs-dir` and first download the
[data required by NetLab-TopoGen](https://github.com/networmix/NetLab-TopoGen#readme).
The `data/` and `lib/` paths are relative to your working directory. From the
repository root, `lib/workflows.yml` replaces NetLab-TopoGen's `design_analysis_brief`
workflow with NetLab's version, and `lib/failure_policies.yml` adds `mc_hard`.

Saved geography must be a corridor `MultiGraph`: a graph that can keep several
routes between the same metro pair. Each file is named
`<config>_integrated_graph.json`. The `topogen_configs/` directory contains larger
designs; `topogen_configs_small/` contains smaller versions for experimentation.

## Control a batch

| Option | Effect |
| --- | --- |
| `--build-jobs`, `--run-jobs` | Number of build and simulation tasks allowed to run at once |
| `--build-timeout` | Time limit for each geography or scenario build task |
| `--run-timeout` | Time limit for each simulation; defaults to 600 seconds |
| `--force-run` | Run simulations again even when cached results match |
| `--force` | Rerun simulations and regenerate geography, unless `--graphs-dir` supplies it |

Time limits start when a task starts running. Scenario assembly runs every time.
NetLab reuses saved work only when a hash of its inputs still matches; a hash
identifies file contents, so any change to the inputs invalidates the cache.
Generated geography is reused while the configuration, data files and package
versions are unchanged. A simulation result is reused while the scenario,
package versions, the saved result file and its run record are unchanged.
Failed tasks make the command return a failure status; independent tasks continue.
Each run's outcome and the wall time of each workflow step are saved in a run
record, `<name>.results.run.json`, beside the results; a reused result keeps the
times of the run that produced it.

## Read and compare metrics

```bash
netlab metrics scenarios/ --no-plots
netlab metrics scenarios/ --only small_baseline
netlab metrics scenarios/ --summary
```

`--no-plots` calculates tables without figures. `--only` takes a comma-separated
list of configuration names, such as `small_baseline,small_clos`.
`--summary` reads the saved tables and renders figures.
A calculation replaces the complete report directory, even with `--only`.
If calculation fails, the existing report is kept.

The workflow must include these step names:

| Step name | NetGraph analysis |
| --- | --- |
| `msd_baseline` | `MaximumSupportedDemand`, which searches for a traffic multiplier |
| `tm_placement` | `TrafficMatrixPlacement` with `alpha_from_step: msd_baseline`, and `outputs: {cost_distribution: true}` for latency |
| `network_statistics` | `NetworkStats`, which counts network objects |

`--enable-maxflow` also needs a `node_to_node_capacity_matrix` step for pairwise
capacity analysis. Each result needs the run record NetLab writes beside it.
See [metrics.md](metrics.md) for definitions, required fields, worked examples
and output files. Failure statistics describe the configured
experiment; they are not measurements of network uptime.

To compare two configurations with matching seeds:

```bash
netlab test scenarios/ small_baseline small_clos
```

Both configurations must have been analyzed. The statistical test needs at least
three matching finite seed results; the two-seed quick start is for running the
pipeline, not statistical inference.

## Python API

Save this as a script and point it to a NetGraph scenario:

```python
from pathlib import Path
from netlab.simulation import run_simulation

if __name__ == "__main__":
    outcome = run_simulation(Path("scenario.yml"), timeout=60)
    print(outcome.status, outcome.error)
```

`run_simulation` runs in a local worker process. A successful run saves
`scenario.results.json` next to the scenario; every run records its status and
the wall time of each step it ran in `scenario.results.run.json`. The `__main__`
guard keeps a spawned worker from running the script again. Use `simulate(yaml_text)` to run in the calling
process, or `inspect_scenario(yaml_text)` to validate and count network objects.
See [architecture and batch APIs](dev/ARCHITECTURE.md).

Metric functions read a saved result dictionary:

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

They return the per-seed values described in [metrics.md](metrics.md).
`netlab metrics` also checks the result arithmetic before it computes them.

## Automated experiments

An autoresearch project gives a language model a task description
(`program.md`), the parameters it may change (`hypothesis_template.yml`) and the
metric to optimize (`objective.yml`). `init` copies a base scenario that has a
`MaximumSupportedDemand` step and at least one `${{name}}` placeholder. Each
placeholder becomes a parameter, and the objective maximizes
`steps.msd_baseline.result.alpha_feasible.value`, the multiplier of the step named
`msd_baseline`. A candidate whose search finds no feasible multiplier fails; it
never scores zero. Review the generated files before a run:

```bash
netlab autoresearch init --base-scenario scenario.yml --output project/
netlab autoresearch run project/ --backend mock --max-experiments 1
```

The `mock` backend is for testing the workflow. To use an external model, select
`claude-cli`, `codex-cli` or `openai` and configure that backend's credentials.
CLI backends need their executable installed and signed in.
`HypothesisManager` can also turn a free-text hypothesis into a scenario, run it
and request an interpretation. Those cycle files are saved under `<project>/cycles/`.

Run one writer per project; nothing locks the files. A project directory is tied
to its input files, seed and installed package versions. A sweep output directory
is tied to its sweep type, seed, iteration count and package versions. If any of
them changes, NetLab refuses to continue there; start a new directory.
Rerunning a sweep in its directory skips successful entries and retries failed
ones. Higher scores rank better; candidates that fail the configured constraints
cannot become best.

For data-center-to-backbone designs, see the
[research project](research_projects/dc-bb-autoresearch/program.md) and
[included interconnect experiments](experiments/dc-bb-interconnect/README.md).

## Development

```bash
make check-ci
```

This checks formatting, lint, types and tests, including NetLab-TopoGen → NetGraph →
metrics runs using saved geography. See [fixture regeneration](tests/data/README.md)
and [metric verification](dev/METRICS_VERIFICATION.md).

In Superset, `bash .superset/workspace.sh setup` creates a `venv`, installs the
development dependencies and copies missing untracked `.env` files from the root
checkout. The Run button executes `make check-ci`.

To test local checkouts of all three dependencies:

```bash
bash dev/check_ngraph_integration.sh /path/to/NetGraph /path/to/NetGraph-Core /path/to/NetLab-TopoGen
```

Replace the paths with your checkouts. The script builds Core and runs checks in
a temporary environment. It saves source revisions, dependency versions and
results under `build/ngraph-integration/`.

[Changelog](CHANGELOG.md) · [MIT license](LICENSE)
