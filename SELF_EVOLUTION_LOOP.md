# Self Evolution Loop

## 목적

- 추천 실험을 단발성 trial이 아니라, champion을 기준으로 스스로 다음 가설을 생성하고 검증하는 반복 루프로 관리한다.
- 현재 루프의 목적은 단순 benchmark 최고점이 아니라, 실제 추천 시스템에 가까운 `serving-safe decision loop`를 만드는 것이다.
- closure week에는 loop를 확장하지 않고, 이미 확보한 결과를 comparative package로 닫는 것을 우선한다.

## Loop State

- `champion_config`
- `champion_metrics`
- `serving_companion_checkpoint`
- `serving_metrics`
- `phase`
- `last_failed_axis`
- `last_failed_axis_family`
- `experiment_memory`

## Core Loop

1. 현재 champion을 읽는다.
2. 가장 유력한 병목 축 하나를 고른다.
3. 가설을 한 문장으로 쓴다.
4. baseline을 champion으로 고정한다.
5. treatment를 한 축만 바꿔 만든다.
6. baseline vs treatment를 같은 조건으로 비교한다.
7. gate를 평가한다.
8. 결과를 memory와 update log에 남긴다.
9. PASS면 champion을 교체한다.
10. FAIL이면 원인을 짧게 기록하고 다음 가설을 제안한다.
11. champion과 serving companion이 다르면 둘 다 유지한다.

## Recursion Policy

- 기본은 `serving exploit 2회 : research exploit 1회 : explore 1회`
- closure mode에서는 exploit/explore를 중단하고 `must-run closure queue`만 수행한다.
- serving exploit:
  - `dual-best`
  - checkpoint policy
  - slice evaluation
  - serving-safe model 선택
- exploit:
  - current research champion 근처의 하이퍼파라미터 또는 low-risk 구조 조정
- explore:
  - 다른 fusion, 다른 scoring, 다른 sequence interaction
  - retrieval / ranking 분리
  - user slice 설계

## Axis Priority

### axis family

1. `evaluation-policy`
2. `slice-analysis`
3. `serving-proxy`
4. `architecture`
5. `metadata-input`
6. `attention-capacity`
7. `retrieval`

### closure priority

1. overall table completion
2. slice table completion
3. graph packaging
4. README summary

### research exploit 우선순위

1. `lr`
2. `drop_out`
3. `weight_decay`
4. `clip_grad_ratio`

### explore 우선순위

1. user slice definition
2. retrieval architecture
3. metadata fusion
4. sequence-side interaction
5. candidate scoring

### serving exploit 우선순위

1. `dual-best` report
2. direct checkpoint evaluation
3. slice evaluation
4. serving candidate summary

## Recursive Learning Rule

- 같은 축에서 2회 연속 FAIL이면 그 축은 한 phase 동안 냉각한다.
- 같은 `axis family`에서 3회 연속 FAIL이면 다른 family로 이동한다.
- fast-scout에서 PASS한 축만 full validation으로 올린다.
- full validation FAIL이면 champion 근처 탐색으로 돌아간다.
- 같은 family에서 `fast PASS / full FAIL` 패턴이 2회 이상 반복되면, 해당 family는 구조 탐색 대신 serving/evaluation 관점으로 재해석한다.

## Promotion Rule

- `B_NDCG` 개선이 있으면 우선 승격 후보
- `B_NDCG` 동률이면 `B_HR`, `T_MAP` 순으로 본다
- 구조가 복잡해졌는데 gain이 미미하면 champion 유지
- `T_MAP`, `T_HR`가 더 좋지만 `B_NDCG`가 근소 열세면 `serving companion` 후보로 유지할 수 있다.
- 연구 champion과 serving companion은 동시에 유지될 수 있다.

## Failure Taxonomy

- `optimization`
- `regularization`
- `feature-fusion`
- `architecture`
- `runtime`
- `evaluation-gap`
- `serving-misalignment`
- `slice-instability`

## End Condition

- 최근 5개 loop 중 4개 이상 FAIL이면, tuning 대신 구조/평가 설계를 재점검한다.
- 최근 4개 loop 중 3개 이상이 `fast PASS / full FAIL`이면, benchmark-only 탐색을 멈추고 phase를 `P4` 또는 `slice-analysis`로 전환한다.
- serving phase로 pivot한 뒤에는, 신규 architecture 탐색 전에 기본 slice 정의를 먼저 고정한다.
- closure week에는 새 explore branch를 열지 않는다.
