# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-27

### Added

- **Workspaces**: Superset setup, teardown, and run commands with isolated environments.
- **Integration**: Local-source test gate and fixture regeneration tooling.

### Changed

- **Dependencies**: Require NetGraph 0.23.1 or newer.
- **Fixtures**: Regenerate scenarios, results, and metrics with current NetGraph and TopoGen.
- **Docs**: Clarify commands, metric definitions, and scenario descriptions.

### Fixed

- **BAC**: Preserve distinct destinations for flows sharing a source.
- **Plots**: Compute iteration medians and interquartile ranges from per-seed data.
- **CLI**: Use current NetGraph commands and return failures from build and run steps.

### Removed

- **BREAKING**: Alternate MSD demand keys; records require `source`, `target`, and `volume`.

## [0.3.0] - 2026-03-26

### Fixed

- Weight metrics by failure-pattern `occurrence_count`.
- Remove unused iteration metric extraction.

### Added

- Per-direction BAC and shared metric helpers.
- DC-BB autoresearch and CLI commands.
- Hand-calculated DC-BB verification fixtures.

### Changed

- Generate separate placement steps for each failure mode.

## [0.2.2] - 2026-03-15

### Changed

- Relicense from AGPL-3.0 to MIT.

## [0.2.1] - 2026-02-02

### Added

- DC-BB interconnect experiments.
- Runtime version reporting through `netlab.__version__`.

### Fixed

- Complete the AGPL license text.

## [0.2.0] - 2025-12-06

### Changed

- **BREAKING**: Require Python 3.11 or newer.
- Require ngraph 0.12.0 or newer.

## [0.1.0] - Previous Release

### Added

- Initial release of NetLab metrics and analysis tools.
