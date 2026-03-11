# 2026-03-11 22:16 docs/local setup

## Included changes

- Expanded the root README with project overview, architecture, expected data columns, evaluation flow, and local execution notes.
- Added `requirements.txt` for local dependency installation.
- Updated `hm_refactored/configs/config.json` to use repository-local paths and local MLflow storage.
- Added `hm_refactored/configs/config.m1_smoke.json` as a lightweight smoke-test config for Apple Silicon and local runs.

## Excluded from this commit

- Runtime changes in `hm_refactored/train.py` were intentionally left out for a separate commit.
