# How NetLab runs work

TopoGen creates geographic routes and network scenarios. NetGraph runs the
scenario's analysis steps. NetLab schedules those calls, saves their results
and turns the results into metrics, tables and figures.

## Worker processes and time limits

`TaskQueue` runs tasks in a local pool of spawned processes using
[Pebble](https://pebble.readthedocs.io/en/stable/). The pool limits how many tasks
run at once. No separate worker service is needed.

A task's time limit starts when it begins executing. If it times out, its worker
is stopped and replaced. If a worker crashes, the pool replaces it too.
Other tasks can continue; failed tasks are reported without automatic retries.

`SimulationBatch` submits scenario files to a queue managed by its caller.
Each task loads a scenario once and executes its NetGraph workflow. The parent
process saves results only after success.

## Pipeline and cache

A master configuration is a TopoGen input file. `run_pipeline` generates or
loads one geographic graph per master configuration.
It builds a scenario for each requested seed and can then simulate each scenario.
Relative datasets and `lib/` definitions are read from the working directory.

```python
from pathlib import Path
from netlab.pipeline import PipelineConfig, run_pipeline

if __name__ == "__main__":
    result = run_pipeline(PipelineConfig(
        masters=[Path("topogen_configs_small/small_baseline.yml")],
        seeds=[11, 12],
        output_dir=Path("scenarios"),
        graphs_dir=Path("tests/data/scenarios/small_baseline/small_baseline"),
        build_jobs=2,
        run_jobs=2,
    ))
    assert result.success, result.errors
```

Run the example from the repository root. The supplied graph avoids a Census
download. Its filename prefix must match the master configuration.

A generated graph is reused while the configuration, its data files and the
package versions are unchanged. Scenario assembly runs on each invocation; when
the rebuilt scenario differs from the saved one, its old results are removed.
A saved simulation is reused only when the scenario, package and result hashes
match. For editable packages, the package record also includes a fingerprint of
the source files, so source changes invalidate cached work. A failed build
removes the corresponding saved scenario and results rather than leaving them
available as valid outputs.

## Metric calculation

Metric functions read NetGraph result dictionaries. `metrics/analysis.py`
validates and analyzes one seed. `metrics/batch.py` groups seeds and creates a
complete report. That report replaces the destination only when analysis
succeeds. `metrics/reporting.py` reads saved metric files to produce summaries
and figures.

A per-seed metric summarizes failure samples for one random seed. A project
metric usually takes the median of those seed summaries. Pooled curves instead
use all samples, so seeds with more samples have more weight. Definitions and
examples are in [metrics.md](../metrics.md).

## Research runs

Research runners use the same simulation service. A project defines inputs,
parameters, constraints and a score. Its log records input and dependency hashes
so resumed work uses the same experiment definition. Higher scores rank better;
a candidate that violates a constraint cannot become best.

Parameter sweeps skip successful entries on resume and retry failed ones.
Generation loops can propose a revised candidate after a failure, using the
recorded error. External language-model CLI backends run as subprocesses.

## Where to change the code

Paths in the table are relative to `netlab/`.

| Responsibility | Modules |
| --- | --- |
| Queue, simulation and saved outputs | `tasks.py`, `simulation.py`, `artifacts.py` |
| TopoGen builds and batch execution | `topology.py`, `pipeline.py`, `cli.py` |
| Experiment inputs | `scenario.py`, `experiment.py` |
| Metric calculation and validation | `metrics/` |
| Experiment failure summaries, comparisons and topology rendering | `metrics_failure.py`, `comparison.py`, `visualize.py` |
| Research loops, scoring and model connections | `autoresearch/` |
| Data-center-to-backbone model, failure modes and sweeps | `autoresearch/dcbb_config.py`, `scenario_generator.py`, `dcbb_failures.py`, `structural_analysis.py`, `sweep.py` |

Run `make check-ci` after changes. Tests cover worker timeouts and crashes,
cache invalidation, failed builds, metric arithmetic and research resume/ranking.
The [metric verification guide](METRICS_VERIFICATION.md) explains how to run
small network experiments with independently calculated answers.
