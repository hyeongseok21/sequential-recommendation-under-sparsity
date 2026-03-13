# Operator Automation Draft

승인된 baseline/treatment를 실제로 실행하고 artifact를 남겨라.

## Read First

- `AGENT.md`
- `RUNBOOK.md`
- `SELF_EVOLUTION_LOOP.md`
- `protocol.md`
- `agents/operator.md`
- `agents/handoff.md`

## Current State

- current champion: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
- primary metric: `B_NDCG`
- gate result path: `data/metrics/gate_result.json`
- experiment memory path: `data/metrics/experiment_memory.csv`

## Task

- 입력으로 받은 baseline/treatment를 실행하라.
- fast-scout 또는 full run 여부를 명시하라.
- 필요하면 `--eval_checkpoint`를 사용해 direct evaluation도 수행하라.
- 실행 후 다음을 반드시 남겨라.
  - update log path
  - gate verdict
  - metric delta
  - checkpoint/report path

## Hard Rules

- baseline은 승인된 champion config를 유지한다.
- 한 번에 한 mutation만 실행한다.
- artifact 없이 결과만 말하지 않는다.
