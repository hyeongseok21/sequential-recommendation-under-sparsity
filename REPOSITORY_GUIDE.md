# Repository Guide

This document is a lightweight navigation index for the repository.
It does not replace the main project README; it helps locate experiment policy, portfolio artifacts, and historical records.

## Start Here

- Main project entry: [`README.md`](README.md)
- Portfolio-facing reports: [`reports/README.md`](reports/README.md)
- Final config index: [`configs/README.md`](configs/README.md)

## Top-Level Research / Experiment Docs

- Experiment governance: [`docs/framework/AGENT.md`](docs/framework/AGENT.md)
- Run procedure: [`docs/framework/RUNBOOK.md`](docs/framework/RUNBOOK.md)
- Self-evolution loop: [`docs/framework/SELF_EVOLUTION_LOOP.md`](docs/framework/SELF_EVOLUTION_LOOP.md)
- Experiment protocol: [`docs/framework/protocol.md`](docs/framework/protocol.md)
- Phase definitions: [`docs/EXPERIMENT_PHASES.md`](docs/EXPERIMENT_PHASES.md)
- Closure plan: [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md)
- Closure model set: [`docs/CLOSURE_MODELS.md`](docs/CLOSURE_MODELS.md)
- Closure preflight checklist: [`docs/CLOSURE_PREFLIGHT.md`](docs/CLOSURE_PREFLIGHT.md)
- Slice definition note: [`SLICE_EVALUATION.md`](SLICE_EVALUATION.md)
- SASRec baseline sanity note: [`SASREC_SANITY_FIX.md`](SASREC_SANITY_FIX.md)

## Artifact Directories

- [`reports/`](reports)
  - curated portfolio-facing summaries
- [`plots/`](plots)
  - final figures used by README and reports
- [`results/`](results)
  - aligned comparison metrics and service-style evaluation outputs
- [`data/metrics/research_study/`](data/metrics/research_study)
  - analysis tables and JSON artifacts used to regenerate the research package
- [`hm_refactored/configs/`](hm_refactored/configs)
  - runnable experiment configs
- [`updates/`](updates)
  - chronological experiment logs and decisions

## What To Read For Different Goals

### Understand the project quickly

1. [`README.md`](README.md)
2. [`reports/research_summary.md`](reports/research_summary.md)
3. [`reports/canonical_evaluation.md`](reports/canonical_evaluation.md)

### Reproduce portfolio artifacts

1. [`README.md`](README.md)
2. [`configs/README.md`](configs/README.md)
3. [`reports/README.md`](reports/README.md)
4. [`scripts/run_phase_agent.py`](scripts/run_phase_agent.py)

### Audit the experiment process

1. [`docs/EXPERIMENT_PHASES.md`](docs/EXPERIMENT_PHASES.md)
2. [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md)
3. [`docs/framework/RUNBOOK.md`](docs/framework/RUNBOOK.md)
4. [`docs/framework/SELF_EVOLUTION_LOOP.md`](docs/framework/SELF_EVOLUTION_LOOP.md)
5. [`updates/`](updates)
