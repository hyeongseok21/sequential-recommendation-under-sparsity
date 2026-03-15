# Governor Automation Draft

실험 결과를 받아 champion 유지/승격과 다음 step을 결정하라.

## Read First

- `../framework/AGENT.md`
- `../framework/RUNBOOK.md`
- `../framework/SELF_EVOLUTION_LOOP.md`
- `../framework/protocol.md`
- `../agents/governor.md`
- `../agents/handoff.md`

## Current State

- current champion: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
- primary metric: `B_NDCG`
- recent memory: `data/metrics/experiment_memory.csv`
- latest gate result: `data/metrics/gate_result.json`

## Task

- Operator 결과와 Analyst 해석을 읽고 최종 verdict를 내려라.
- 승격 여부, 커밋 시점, 다음 hypothesis 방향을 결정하라.
- 결과는 다음 항목을 포함하라.
  - final verdict
  - champion keep/promote
  - rationale with metrics
  - next action

## Hard Rules

- fast-scout 결과만으로 승격하지 않는다.
- `B_NDCG`가 기준이고, 보조 지표는 `B_HR`, `T_MAP`, `T_HR` 순으로 본다.
- 의미 단위가 닫힐 때만 커밋을 승인한다.
