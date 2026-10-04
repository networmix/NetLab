# NetLab architecture

TopoGen builds geography and scenarios; NetGraph expands and simulates them.
NetLab calls their Python APIs, schedules work, saves results, and computes metrics.
CLI handlers pass arguments to these library services.

## Execution and caching

`TaskQueue` uses [Pebble](https://pebble.readthedocs.io/en/stable/) with a bounded
pool of spawned processes. Deadlines start when execution begins. A timed-out
worker is stopped and replaced, and independent jobs continue. No broker or worker
service is needed for these local batches.

`SimulationBatch` submits independent scenario files to a caller-managed queue.
Each simulation loads its scenario once. The parent publishes results after
success, and cache reuse requires matching scenario, package, and result hashes.
Editable packages include a source fingerprint. Failed builds invalidate earlier
scenario and result files.

`run_pipeline` generates one graph per master, builds each seed, and optionally
simulates it. Relative datasets and `lib/` resolve from the working directory.
Supplied geography must use TopoGen's corridor MultiGraph format.

```python
from pathlib import Path
from netlab.pipeline import PipelineConfig, discover_configs, run_pipeline

if __name__ == "__main__":
    result = run_pipeline(PipelineConfig(
        masters=discover_configs(Path("topogen_configs_small")),
        seeds=[11, 12], output_dir=Path("scenarios"),
        graphs_dir=Path("graphs"), build_jobs=2, run_jobs=2,
    ))
    assert result.success, result.errors
```

Failed tasks are reported without automatic retries. Research generation loops
can explicitly revise failed candidates using the recorded error. Subprocesses
remain only for external LLM CLI backends.

## Where to make changes

| Area | Modules under `netlab/` |
| --- | --- |
| Task scheduling and results | `tasks.py`, `simulation.py`, `artifacts.py` |
| TopoGen batches | `topology.py`, `pipeline.py`, `cli.py` |
| Experiment inputs and runs | `scenario.py`, `experiment.py` |
| Tables and graph rendering | `comparison.py`, `visualize.py` |
| Research generation and analysis | `autoresearch/generation_loop.py`, `analysis_loop.py`, `hypothesis_manager.py` |
| Parameter searches | `autoresearch/hypothesis.py`, `runner.py`, `objective.py`, `sweep.py` |
| Research history and prompts | `autoresearch/experiment_log.py`, `memory.py`, `prompt.py`, `metrics_report.py` |
| LLM connections and research CLI | `autoresearch/backend.py`, `cli.py`; executable lookup in `runtime.py` |
| DC-BB topology model | `autoresearch/dcbb_config.py`, `dcbb_failures.py`, `scenario_generator.py`, `scenario_validation.py`, `structural_analysis.py` |
| Metric calculations | `metrics/bac.py`, `latency.py`, `msd.py`, `sps.py`, `costpower.py`, `iterops.py`, `matrixdump.py`; failure summaries in `metrics_failure.py` |
| Metric input and aggregation | `metrics/validation.py`, `common.py`, `analysis.py`, `aggregate.py`, `batch.py`, `seed_data.py` |
| Statistics and reports | `metrics/distributions.py`, `paired.py`, `comparisons.py`, `summary.py`, `reporting.py`, `plot_*.py` |

Metric functions consume result dictionaries. `analysis.py` validates and analyzes
one seed; `batch.py` groups seeds and publishes a complete report. `reporting.py`
reads saved tables and metric files. Shared seed readers and statistical functions
keep table, plot, and comparison calculations consistent.

Research runners use the same simulation service. Logs bind resumed work to input
and dependency fingerprints; feasible candidates are ranked by higher scores.
Only successful sweep entries are skipped on resume.

## Checks

Run `make check-ci` for formatting, lint, type checks, and tests. Execution tests
cover worker timeouts and crashes, cache invalidation, failed builds, and queued
TopoGen → NetGraph runs with cached geography. Research tests cover resume,
parameter validation, and ranking. Rendering tests check component bounds.

For the independent metric experiments and their results, see
[METRICS_VERIFICATION.md](METRICS_VERIFICATION.md).
