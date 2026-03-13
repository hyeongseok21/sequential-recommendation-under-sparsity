---
name: recsys-self-evolution-runbook
description: Runs recommendation experiments in this repository using the local AGENT.md, RUNBOOK.md, SELF_EVOLUTION_LOOP.md, protocol.md, and agents/ role docs. Use when the user asks to operate a recursive experiment loop, define or follow self-evolving baseline-vs-treatment workflows, promote champions, record gate results, or maintain experiment memory.
---

# Recsys Self Evolution Runbook

Use this skill when the task is to run or manage recursive recommendation experiments in this repository.

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
2. Write one-sentence `hypothesis`.
3. Fix the `baseline` as the current champion.
4. Apply one `treatment` mutation only.
5. Run baseline/treatment with the canonical command from `RUNBOOK.md`.
6. Evaluate the gate:
   - use `python3 scripts/evaluate_gate.py`
7. Update memory:
   - use `python3 scripts/update_experiment_memory.py`
8. Validate docs if files were added:
   - use `python3 scripts/lint_experiment_docs.py`
9. Write update log from `templates/update_log_template.md`.

## Hard Rules

- One hypothesis per loop.
- Do not mix unrelated refactors into experiment loops.
- Use `B_NDCG` as the primary metric unless the runbook explicitly says otherwise.
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
