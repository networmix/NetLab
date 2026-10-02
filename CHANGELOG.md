# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **CLI**: Invoke `ngraph inspect` without the unsupported output option.
- **CLI**: Return a nonzero exit status when scenario inspection or simulation fails.
- **BAC**: Keep destinations distinct for flows sharing a source and sum priority classes within each direction.
- **Sweeps**: Normalize directional BAC by baseline delivery, matching the metrics pipeline.
- **Plots**: Compute iteration-count and timing medians and interquartile ranges from per-seed data.

### Changed

- **BREAKING**: **Dependencies**: Require NetGraph `>=0.23.1`.
- **BREAKING**: **Workflows**: Placement inherits the scenario seed; set step `seed` explicitly to keep using `42`.
- **BREAKING**: **Metrics**: Require a positive integer `occurrence_count` on every failure flow result.
- **BREAKING**: **Summaries**: Replace operation counters with `iters_fail`, `iters_total`, and `unique_patterns`.
- **Scenarios**: Generated scenarios, shared templates, and the autoresearch prompt use current NetGraph keys.
- **Autoresearch**: Prompts distinguish measurements from explanations and flag missing or inconsistent evidence.
- **Autoresearch**: Load the NetGraph DSL reference from the `.claude/skills` submodule; run `git submodule update --init` to enable it.
- **Fixtures**: Regenerate scenarios, results, and metrics; see `tests/data/README.md` for numerical changes.
- **Docs**: Correct command examples, metric definitions, and scenario descriptions.
- **Internal**: Consolidate metric and provenance helpers, run the TopoGen pipeline checks in the default test suite, and remove unused code, fixtures, and the PyPI publish workflow.

### Added

- **Workspaces**: Superset setup creates an isolated environment and copies missing `.env` files; Run executes `make check-ci`.
- **Integration**: Add local-source validation and fixture regeneration commands.

### Removed

- **BREAKING**: **MSD**: Remove `base_demands` aliases `source_path`, `sink_path`, and `demand`; use `source`, `target`, and `volume`.
- **BREAKING**: **MSD**: Read alpha from `msd_baseline.data.alpha_star`; remove `msd` and placement-probe inference.
- **BREAKING**: **Timing**: Read step duration from `metadata.duration_sec`; remove the `execution_time` alias.
- **BREAKING**: **Graphs**: Require `edges` in graph results; remove support for the `links` alias.
- **BREAKING**: **Autoresearch**: Remove the `parameters` alias in hypothesis templates; use `params`.

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
