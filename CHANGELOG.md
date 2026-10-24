# Changelog — `park-ez-vision`

All notable engineering changes to `park-ez-vision` (`park_ez`) are documented in this file.

## [0.3.0] — 2026-10-24
### Added
- Documented CLI diagnostic usage (`python -m park_ez --self-test --json`) and deterministic invariant verification workflows in `README.md`.
- Added release history and semantic versioning audit trail in `CHANGELOG.md`.

## [0.2.1] — 2026-10-23
### Added
- Added automated CLI and runtime invariant regression suite in `tests/test_diagnostics_cli.py` covering seed determinism, negative-seed rejection, JSON output schema, and `--self-test` exit codes.

## [0.2.0] — 2026-10-21
### Added
- Added `src/park_ez/diagnostics.py` (`run_diagnostic_suite`, `validate_runtime_invariants`) and module CLI runner `src/park_ez/__main__.py`.
- Exported diagnostic helpers in `park_ez.__all__`.

## [0.1.0] — 2026-10-19
### Added
- Initial architecture and core test suite: Park-EZ (1st Place SMU OIT Challenge @ HackSMU VI) - Computer-vision polygon IoU parking stall occupancy tracker and A* ADA campus lot router.
