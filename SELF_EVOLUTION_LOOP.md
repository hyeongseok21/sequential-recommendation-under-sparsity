# Self Evolution Loop

## 목적

- 추천 실험을 단발성 trial이 아니라, champion을 기준으로 스스로 다음 가설을 생성하고 검증하는 반복 루프로 관리한다.

## Loop State

- `champion_config`
- `champion_metrics`
- `phase`
- `last_failed_axis`
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

## Recursion Policy

- 기본은 `exploit 3회 : explore 1회`
- exploit:
  - current champion 근처의 하이퍼파라미터 또는 low-risk 구조 조정
- explore:
  - 다른 fusion, 다른 scoring, 다른 sequence interaction

## Axis Priority

### exploit 우선순위

1. `lr`
2. `drop_out`
3. `weight_decay`
4. `clip_grad_ratio`

### explore 우선순위

1. metadata fusion
2. sequence-side interaction
3. candidate scoring
4. loss/regularization

## Recursive Learning Rule

- 같은 축에서 2회 연속 FAIL이면 그 축은 한 phase 동안 냉각한다.
- fast-scout에서 PASS한 축만 full validation으로 올린다.
- full validation FAIL이면 champion 근처 탐색으로 돌아간다.

## Promotion Rule

- `B_NDCG` 개선이 있으면 우선 승격 후보
- `B_NDCG` 동률이면 `B_HR`, `T_MAP` 순으로 본다
- 구조가 복잡해졌는데 gain이 미미하면 champion 유지

## Failure Taxonomy

- `optimization`
- `regularization`
- `feature-fusion`
- `architecture`
- `runtime`
- `evaluation-gap`

## End Condition

- 최근 5개 loop 중 4개 이상 FAIL이면, tuning 대신 구조/평가 설계를 재점검한다.
