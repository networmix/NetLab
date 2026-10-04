# Changelog

## [Unreleased]

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
