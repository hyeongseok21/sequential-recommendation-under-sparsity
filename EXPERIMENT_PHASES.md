# Experiment Phases

## 목적

- 실험 phase를 한 장에서 볼 수 있게 정리한다.
- `AGENT.md`의 운영 규칙, `SELF_EVOLUTION_LOOP.md`의 반복 루프, `RUNBOOK.md`의 실행 절차를 phase 기준으로 연결한다.

## 문서 역할 분리

- [`AGENT.md`](/Users/conan/projects/personalized-fashion-recommendation/AGENT.md)
  - 공통 헌장
  - gate 기준
  - champion 정책
- [`SELF_EVOLUTION_LOOP.md`](/Users/conan/projects/personalized-fashion-recommendation/SELF_EVOLUTION_LOOP.md)
  - recursion rule
  - exploit / explore 순서
  - 축 이동 규칙
- [`RUNBOOK.md`](/Users/conan/projects/personalized-fashion-recommendation/RUNBOOK.md)
  - 실제 실행 명령
  - checkpoint / report 절차
- [`EXPERIMENT_PLAN.md`](/Users/conan/projects/personalized-fashion-recommendation/EXPERIMENT_PLAN.md)
  - 현재 backlog
  - immediate next experiment
- 이 문서:
  - phase 정의
  - phase별 entry / exit criteria
  - 현재 active phase

## Phase Map

### `P0` Bring-up

- 목적:
  - 환경, 데이터, 로깅, 평가 루프가 끝까지 도는지 확인
- 주요 작업:
  - 로컬 실행
  - device fallback
  - smoke test
  - 경로/의존성 안정화
- entry:
  - 새 환경 또는 새 저장소 상태
- exit:
  - smoke run 성공
  - train/eval/checkpoint가 end-to-end로 동작

### `P1` Baseline

- 목적:
  - 재현 가능한 baseline과 초기 champion 확보
- 주요 작업:
  - seed 고정
  - 평가 안정화
  - baseline config 확정
  - meta baseline 확보
- entry:
  - `P0` 완료
- exit:
  - 동일 config 재실행 시 동일 metric
  - baseline/champion config path가 고정됨

### `P2` Fast-Scout

- 목적:
  - 저비용 screening으로 유망 후보만 고른다
- 주요 작업:
  - single hypothesis test
  - batch 축소
  - epoch 1 기반 screening
  - feature subset / weighting / head / lr 탐색
- entry:
  - 현재 champion 존재
- exit:
  - PASS 후보를 `P3`로 승격
  - FAIL 후보는 memory에 기록 후 종료

### `P3` Full Validation

- 목적:
  - fast-scout PASS 후보를 full budget에서 검증
- 주요 작업:
  - `batch_size=16`, `epoch=3` 같은 full profile 검증
  - early-stop 판단
  - champion promotion 여부 결정
- entry:
  - `P2` PASS
- exit:
  - `B_NDCG` 기준 champion 유지 또는 교체

### `P4` Evaluation Policy / Inference

- 목적:
  - 모델보다 선택/보고/serving 정책을 정리
- 주요 작업:
  - checkpoint selection
  - `benchmark-best` vs `test-best`
  - direct checkpoint evaluation
  - dual-best report
  - eval-gap leaderboard
- entry:
  - `P3`에서 evaluation-gap이 반복되거나, champion 후보가 비슷한 성능으로 갈릴 때
- exit:
  - checkpoint/report 정책이 고정됨

### `P5` Finalization

- 목적:
  - champion 고정
  - 결과 요약
  - 재실행 및 handoff 준비
- 주요 작업:
  - summary log
  - final report
  - agent handoff
  - automation wiring
- entry:
  - champion과 운영 정책이 모두 고정
- exit:
  - 다음 phase 또는 다음 프로젝트로 handoff 가능

## Active Phase

- 현재 active phase:
  - `P2` + `P3`
- 이유:
  - current champion 주변에서 `metadata-input`, `attention-capacity`를 fast-scout으로 스크리닝하고
  - 살아남은 후보만 full validation으로 올리고 있음
- 참고 champion:
  - [`hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json)

## Phase Transition Rules

1. `P0 -> P1`
   - smoke run과 eval loop가 안정화되면 이동
2. `P1 -> P2`
   - baseline이 고정되면 이동
3. `P2 -> P3`
   - fast-scout PASS가 나오면 이동
4. `P3 -> P2`
   - full validation FAIL이면 돌아감
5. `P3 -> P4`
   - evaluation-gap 또는 checkpoint policy 이슈가 반복되면 이동
6. `P4 -> P2/P3`
   - 평가 정책이 정리되면 다시 모델 실험으로 복귀
7. `P4 -> P5`
   - champion과 selection policy가 모두 고정되면 이동

## Current Priority

1. `metadata-input`
2. `attention-capacity`
3. `architecture`
4. `evaluation-policy`

## Naming Rule

- update log에는 현재 실험의 `axis family`를 드러낸다.
- `experiment_memory.csv`에는 `phase`와 `axis`를 함께 남긴다.
- 새 후보는 항상 `baseline -> treatment -> gate -> log` 순서로 닫는다.
