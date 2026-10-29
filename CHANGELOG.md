# Changelog

## [Unreleased]

## [0.6.0] - 2026-10-29

### Changed

- **BREAKING**: **Dependencies**: Require `ngraph>=0.26.0,<0.27` and `netlab-topogen>=0.6.1,<0.7`, which replaces TopoGen.
- **BREAKING**: **Scenarios**: Workflows use NetGraph 0.26 keys: `alpha_tolerance`, `workers`, `outputs.cost_distribution`, `preset`.
- **BREAKING**: **TopoGen configs**: Drop `build.tm_sizing.flow_placement`; traffic uses `TE_MAXFLOW_UNLIM` instead of `TE_WCMP_UNLIM`.
- **BREAKING**: **Input formats**: Read NetGraph 0.26 results: named steps, sampled cases, units of `units_report.q`, MSD `alpha_feasible`.
- **BREAKING**: **Metrics**: `netlab metrics` requires each result's run record, `<name>.results.run.json`, with every step's wall time.
- **BREAKING**: **Research**: Objectives read `steps.msd_baseline.result.alpha_feasible.value`; `init` writes that path.
- **BREAKING**: **Python API**: `compute_iter_ops` and `analyze_one_seed` take the run record; `pair_totals` takes the results, step and case.
- **Runs**: A run whose MSD search finds no feasible multiplier fails with the reason `none_feasible`.
- **Metrics**: Bandwidth follows NetGraph 0.26's whole units of `q`, and `random` failure rules draw per matched entity.
- **Metrics**: TE latency percentiles and shares read NetGraph's marginal cost curve; WES uses each row's mean path cost.
- **Metrics**: Provenance records NetLab's `commit` (clean checkout or GitHub install) and `dirty`, not its install URL; null if unknown.
- **Docs**: Explain samples and the baseline reference, bandwidth units, run records and TE latency; examples say when they need network.
- **Internal**: Regenerate fixtures with run records on the published packages; dev and workspace scripts use `netlab_topogen`.

### Added

- **Runs**: Run records store each step's wall time, measured by NetLab, as `step_durations_sec`; `tm_duration_*` read it.

### Removed

- **Python API**: Remove `expand_flow_results` and `flow_occurrence_count` from `netlab.metrics.common`.

## [0.5.0] - 2026-10-25

### Fixed

- **Experiments**: DC-BB `summary.json` and `--compare` report `alpha_star` as the median over seeds, plus `alpha_star_by_seed`; zero shows as `0.00`.
- **Research**: Generated DC-BB workflow steps inherit the scenario seed, so `netlab autoresearch run --seed` changes failure sampling.
- **Research**: The DC-BB `program.md` example returns integer `g_abc1` and `g_xyz1`, which the parameter template accepts.

### Changed

- **Dependencies**: Use TopoGen 0.5.2 with `ngraph>=0.24.1,<0.25.0`.
- **BREAKING**: **Experiments**: The DC-BB workflow names its MSD step `msd_baseline`; read `steps.msd_baseline` in its results.
- **BREAKING**: **Experiments**: Rename the two-plane topology to `dc4x9_bb2x4_bb2x4_dc4x9_two_to_one`, matching its backbone grid.
- **Docs**: Explain setup, metric calculations and research workflows with self-contained examples and reproducible checks.
- **Internal**: Remove the unused DC-BB `blueprints/blocks.yml`.

## [0.4.1] - 2026-10-25

### Changed

- **Dependencies**: Require `ngraph>=0.24.1,<0.25.0` and use TopoGen 0.5.1 to stay on the supported NetGraph API.

## [0.4.0] - 2026-10-04

### Changed

- **BREAKING**: **Execution**: Use TopoGen `3bf0fa4a` and NetGraph `>=0.24.0` through their Python APIs, with a local process queue. TopoGen and NetGraph executable overrides are removed.
- **BREAKING**: **Scenarios**: `netlab build` now produces seeded scenarios. Supply existing corridor MultiGraphs with `--graphs-dir`. Placement workflows inherit the scenario seed unless a step sets its own.
- **BREAKING**: **Python API**: Import metrics from `netlab.metrics`; the top-level `metrics` package is removed.
- **BREAKING**: **Input formats**: Require list-form workflows and current NetGraph result fields: MSD `source`/`target`/`volume` under `msd_baseline`, timing in `metadata.duration_sec`, and graph `edges`. Failure patterns require a positive integer `occurrence_count`.
- **BREAKING**: **Metric files**: Preserve full endpoint names and bandwidth above baseline; write unavailable numbers as `null`. Research logs use `bw_p99_pct` for availability. See [metrics.md](metrics.md) for output fields and definitions.
- **BREAKING**: **Research**: Templates require `params`, and programmatic generators receive typed values. Resuming requires matching inputs, seed, and dependencies; use a new output directory when these change.

### Fixed

- **Bandwidth**: Correct availability thresholds and failure-pattern weighting. Sum priority classes without merging distinct endpoints, and keep complete outages in the samples.
- **Latency and comparisons**: Use each endpoint pair's baseline path cost and compare matching seeds. Fix errors caused by measurement scale and inconsistent baseline selection in tables and plots.
- **Runs and reports**: Enforce timeouts, reject stale cached results, and return a failing CLI status when a task fails. Publish complete metric reports only after successful analysis, preserving the previous report on failure.
- **Research**: Rank minimized objectives correctly and prevent infeasible candidates from winning. Allow failed hypotheses and sweep entries to be retried.

### Added

- **Workspaces**: Superset setup installs a development environment and copies missing `.env` files; Run executes `make check-ci`.

## [0.3.0] - 2026-03-26

### Fixed

- All metric modules now expand deduplicated flow_results by `occurrence_count` before statistical computation
- Removed dead `iteration_metrics` extraction from iterops (field never existed in ngraph)

### Added

- Per-direction BAC (`per_flow` field on `BacResult`) for directional asymmetry analysis
- Shared utilities in `metrics/common.py`: `expand_flow_results`, `canonical_dc`, `baseline_demand_map`
- Mini DC-BB verification scenario with hand-calculated metric assertions
- DC-BB autoresearch framework: scenario generator, structural analysis, parametric sweep, generation loop
- CLI subcommands: `netlab autoresearch structural-analysis`, `sweep`, `cross-sweep`

### Changed

- Scenario generator produces per-mode TMP workflow steps (`tm_lh_path`, `tm_combined`, etc.) instead of single `tm_placement`

## [0.2.2] - 2026-03-15

### Changed

- Relicensed from AGPL-3.0 to MIT

## [0.2.1] - 2026-02-02

### Added

- `experiments/dc-bb-interconnect/` - DC-backbone interconnect analysis experiment with multiple topology scenarios
- `netlab.__version__` - Runtime version access via `importlib.metadata`

### Fixed

- Replaced incomplete LICENSE file with full AGPL-3.0 text

## [0.2.0] - 2025-12-06

### Changed

- **BREAKING**: Minimum Python version raised to 3.11
- **Dependencies**: Updated ngraph to >=0.12.0

## [0.1.0] - Previous Release

Initial release of NetLab metrics and analysis tools.
