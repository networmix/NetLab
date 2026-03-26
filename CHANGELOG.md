# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
