# Experiment Phases

## 목적

- 실험 phase를 한 장에서 볼 수 있게 정리한다.
- `AGENT.md`의 운영 규칙, `SELF_EVOLUTION_LOOP.md`의 반복 루프, `RUNBOOK.md`의 실행 절차를 phase 기준으로 연결한다.

## 문서 역할 분리

- [`../AGENT.md`](../AGENT.md)
  - 공통 헌장
  - gate 기준
  - champion 정책
- [`../SELF_EVOLUTION_LOOP.md`](../SELF_EVOLUTION_LOOP.md)
  - recursion rule
  - exploit / explore 순서
  - 축 이동 규칙
- [`../RUNBOOK.md`](../RUNBOOK.md)
  - 실제 실행 명령
  - checkpoint / report 절차
- [`../SLICE_EVALUATION.md`](../SLICE_EVALUATION.md)
  - slice 정의
  - slice metric 해석
  - serving phase 기준 slice report 정책
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)
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
  - research champion / serving companion 분리
  - 기본 slice report 정의
- entry:
  - `P3`에서 evaluation-gap이 반복되거나, champion 후보가 비슷한 성능으로 갈릴 때
- exit:
  - checkpoint/report 정책이 고정됨

### `P5` Slice / Serving Finalization

- 목적:
  - serving 관점에서 overall + slice 해석을 고정한다
- 주요 작업:
  - `sparse-history user`
  - `multi-interest user`
  - slice companion 해석
  - retrieval phase 진입 전 slice backbone 고정
- entry:
  - `P4`에서 dual-best, checkpoint policy가 정리됨
- exit:
  - slice 정의와 해석 규칙이 고정됨
  - retrieval 또는 다음 서비스 실험으로 handoff 가능

### `P6` Portfolio Closure

- 목적:
  - 포트폴리오 제출 가능한 결과물을 완성한다
- 주요 작업:
  - overall 결과 표
  - slice 결과 표
  - 그래프 2~4개
  - concise findings
  - limitations
  - README-ready summary
- entry:
  - champion, 운영 정책, slice 정의가 모두 고정
- exit:
  - 포트폴리오 패키지 완료

## Active Phase

- 현재 active phase:
  - `P6`
- 이유:
  - 남은 시간이 1주라서 추가 탐색보다 closure가 중요하다
  - baseline, metadata champion, dual-best 정책은 이미 확보했다
  - 이제는 overall/slice 결과 표와 README summary를 만드는 것이 핵심이다
- 참고 champion:
  - [`../hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`](../hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json)

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
   - dual-best와 serving companion 정책이 정리되면 slice 해석 phase로 이동
8. `P5 -> P2/P3`
   - slice 정의가 고정되면 다시 모델 실험으로 복귀
9. `P5 -> P6`
   - champion과 selection policy, slice 해석이 모두 고정되면 이동

## Current Priority

1. `portfolio-closure`
2. `slice-analysis`
3. `serving-proxy`
4. `evaluation-policy`
5. `architecture`

## Naming Rule

- update log에는 현재 실험의 `axis family`를 드러낸다.
- `experiment_memory.csv`에는 `phase`와 `axis`를 함께 남긴다.
- 새 후보는 항상 `baseline -> treatment -> gate -> log` 순서로 닫는다.
