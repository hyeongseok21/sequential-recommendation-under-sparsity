# Analyst Automation Draft

현재 추천 실험 워크스페이스에서 다음 한 개의 hypothesis를 제안하라.

## Read First

- `AGENT.md`
- `RUNBOOK.md`
- `SELF_EVOLUTION_LOOP.md`
- `protocol.md`
- `agents/analyst.md`
- `agents/handoff.md`
- `agents/orchestration.md`

## Current State

- current champion: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
- primary metric: `B_NDCG`
- recent memory: `data/metrics/experiment_memory.csv`
- latest gate result: `data/metrics/gate_result.json`

## Task

- 최근 실험 실패/성공 패턴을 읽고 다음 hypothesis를 1개만 제안하라.
- baseline은 current champion으로 고정하라.
- treatment는 한 축만 바뀌어야 한다.
- 제안 결과는 다음 항목을 포함하라.
  - phase
  - hypothesis
  - baseline config
  - treatment axis
  - expected win condition
  - bottleneck classification

## Hard Rules

- `B_NDCG`를 1순위로 본다.
- fast-scout PASS만으로 champion 교체를 제안하지 않는다.
- vague summary 대신 file path와 metric 근거를 남긴다.
