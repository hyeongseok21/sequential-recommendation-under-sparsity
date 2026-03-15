---
name: recsys-self-evolution-runbook
description: Runs recommendation experiments in this repository using the local framework docs under docs/framework, the protocol, and agent role docs under docs/agents. Use when the user asks to operate a recursive experiment loop, define or follow self-evolving baseline-vs-treatment workflows, promote champions, record gate results, or maintain experiment memory.
---

# Recsys Self Evolution Runbook

Use this skill when the task is to run or manage recursive recommendation experiments in this repository.
Prefer serving-oriented decision loops over benchmark-only tuning when the docs indicate `P4` or `evaluation-policy` is active.
If the docs indicate `P6` or `closure`, prefer completion artifacts over new exploration.

## Always Read First

1. `docs/framework/AGENT.md`
2. `docs/framework/RUNBOOK.md`
3. `docs/framework/SELF_EVOLUTION_LOOP.md`
4. `docs/framework/protocol.md`
5. `SLICE_EVALUATION.md`
6. `docs/agents/orchestration.md`
7. `docs/agents/handoff.md`
8. `docs/agents/automation.md`
9. `docs/templates/agent_handoff_template.md`
10. `docs/templates/agent_prompt_template.md`
11. `docs/templates/automation_prompt_bundle_template.md`

## Workflow

1. Identify the current `phase`.
2. Identify the current `decision_mode`:
   - `research`
   - `serving`
   - `hybrid`
   - `closure`
3. Write one-sentence `hypothesis`.
4. Fix the `baseline` as the current champion.
5. Apply one `treatment` mutation only.
6. Run baseline/treatment with the canonical command from `docs/framework/RUNBOOK.md`.
7. Evaluate the gate:
   - use `python3 scripts/evaluate_gate.py`
8. Update memory:
   - use `python3 scripts/update_experiment_memory.py`
9. Validate docs if files were added:
   - use `python3 scripts/lint_experiment_docs.py`
10. Write update log from `docs/templates/update_log_template.md`.
11. If `serving` or `hybrid`, preserve both `research champion` and `serving companion` when they differ.
12. If the active family is `slice-analysis`, ground the hypothesis and interpretation in `sparse-history user` or `multi-interest user`.
13. If `closure`, run only the MUST queue and avoid new exploratory branches.

## Hard Rules

- One hypothesis per loop.
- Do not mix unrelated refactors into experiment loops.
- Use `B_NDCG` as the primary metric for research loops unless the runbook explicitly says otherwise.
- In serving-oriented loops, interpret `T_MAP`, `T_HR`, and `dual-best` together with `B_NDCG`.
- Fast-scout PASS is not enough for champion promotion.
- Distinguish `concept failed` from `current implementation failed`.
- In closure mode, prioritize overall table, slice table, graph, and README artifacts over further tuning.

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
