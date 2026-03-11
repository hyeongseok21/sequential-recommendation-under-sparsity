# 2026-03-11 22:33 m1 runtime compat

## Included changes

- Added automatic device selection for `cuda`, `mps`, and `cpu` in `hm_refactored/train.py`.
- Switched tensor materialization to `torch.as_tensor(..., device=...)` for better MPS compatibility.
- Added an MLflow fallback so training can proceed even if `mlflow` is not installed.
- Fixed `benchmark=false` training completion logic in `hm_refactored/train.py`.
- Replaced deprecated `Series.append()` usage with `pandas.concat()` in preprocessing code for pandas 2.x compatibility.
- Updated `config.m1_smoke.json` to use the smoke dataset and disable benchmark mode.

## Excluded from this commit

- Generated artifacts such as `.venv`, `mlruns/`, sampled CSVs, and preprocessed pickle files were left out.
- Finder metadata files such as `.DS_Store` were left out.
