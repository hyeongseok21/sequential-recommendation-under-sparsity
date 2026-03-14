---
name: recsys-self-evolution-runbook
description: Runs recommendation experiments in this repository using the local AGENT.md, RUNBOOK.md, SELF_EVOLUTION_LOOP.md, protocol.md, and agents/ role docs. Use when the user asks to operate a recursive experiment loop, define or follow self-evolving baseline-vs-treatment workflows, promote champions, record gate results, or maintain experiment memory.
---

# Recsys Self Evolution Runbook

Use this skill when the task is to run or manage recursive recommendation experiments in this repository.
Prefer serving-oriented decision loops over benchmark-only tuning when the docs indicate `P4` or `evaluation-policy` is active.

## Always Read First

1. `/Users/conan/projects/personalized-fashion-recommendation/AGENT.md`
2. `/Users/conan/projects/personalized-fashion-recommendation/RUNBOOK.md`
3. `/Users/conan/projects/personalized-fashion-recommendation/SELF_EVOLUTION_LOOP.md`
4. `/Users/conan/projects/personalized-fashion-recommendation/protocol.md`
5. `/Users/conan/projects/personalized-fashion-recommendation/agents/orchestration.md`
6. `/Users/conan/projects/personalized-fashion-recommendation/agents/handoff.md`
7. `/Users/conan/projects/personalized-fashion-recommendation/agents/automation.md`
8. `/Users/conan/projects/personalized-fashion-recommendation/templates/agent_handoff_template.md`
9. `/Users/conan/projects/personalized-fashion-recommendation/templates/agent_prompt_template.md`
10. `/Users/conan/projects/personalized-fashion-recommendation/templates/automation_prompt_bundle_template.md`

## Workflow

1. Identify the current `phase`.
2. Identify the current `decision_mode`:
   - `research`
   - `serving`
   - `hybrid`
3. Write one-sentence `hypothesis`.
4. Fix the `baseline` as the current champion.
5. Apply one `treatment` mutation only.
6. Run baseline/treatment with the canonical command from `RUNBOOK.md`.
7. Evaluate the gate:
   - use `python3 scripts/evaluate_gate.py`
8. Update memory:
   - use `python3 scripts/update_experiment_memory.py`
9. Validate docs if files were added:
   - use `python3 scripts/lint_experiment_docs.py`
10. Write update log from `templates/update_log_template.md`.
11. If `serving` or `hybrid`, preserve both `research champion` and `serving companion` when they differ.

## Hard Rules

- One hypothesis per loop.
- Do not mix unrelated refactors into experiment loops.
- Use `B_NDCG` as the primary metric for research loops unless the runbook explicitly says otherwise.
- In serving-oriented loops, interpret `T_MAP`, `T_HR`, and `dual-best` together with `B_NDCG`.
- Fast-scout PASS is not enough for champion promotion.
- Distinguish `concept failed` from `current implementation failed`.

## Output Contract

Return these sections:

1. `Hypothesis`
2. `Mutation`
3. `Commands`
4. `Metric Deltas vs Baseline`
5. `Gate Verdict`
6. `Learning_t`
7. `Champion Decision`
8. `Next Hypothesis`
