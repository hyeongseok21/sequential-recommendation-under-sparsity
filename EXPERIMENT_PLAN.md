# Experiment Plan

## Current Champion

- config: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`
- primary metric: `B_NDCG`
- current best full score:
  - `B_HR 0.0309`
  - `B_NDCG 0.0139`
  - `T_HR 0.0042`
  - `T_MAP 0.0015`

## Goal

1. feature weighting으로 all-features champion을 더 밀어 올릴 수 있는지 확인
2. `benchmark-best`와 `test-best` gap을 줄일 수 있는지 확인
3. `attention-capacity` 축의 `h4`를 다시 살릴 가치가 있는지 재평가

## Backlog

### `attention-capacity`

1. `all_features + h4 + batch_size 32`
2. `all_features + h4 + drop_out 0.05`
3. 앞선 후보가 PASS할 때만 `all_features + h4 + batch_size 32 + drop_out 0.05`

### `metadata-input`

1. `garment_group_scale > 1.0`
2. `product_type_scale` 추가 미세조정
3. `department_scale` 재탐색은 우선순위 낮음

### `architecture`

1. feature-specific history projection
2. feature-specific metadata weighting
3. attention input / item representation 분리 경로 정교화

### `evaluation-policy`

1. dual-best 기준 유지
2. 새 후보는 항상 direct checkpoint evaluation 같이 수행

## Immediate Next Experiment

- family: `evaluation-policy`
- hypothesis: `product_type15` champion의 dual-best report를 기준으로 benchmark-best와 test-best를 운영 후보로 분리하는 편이 더 실용적이다.
- baseline:
  - benchmark-best 단일 checkpoint 운용
- treatment:
  - dual-best report + direct checkpoint evaluation 기반 동시 운용
- expected gate:
  - research champion과 test-oriented companion을 명시적으로 분리
  - 이후 후보 비교 시 dual-best를 기본 artifact로 유지

## Latest Result

- family: `attention-capacity`
- treatment:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15_h4.json`
- verdict: `FAIL`
- observed:
  - fast-scout:
    - `B_HR 0.0208`
    - `B_NDCG 0.0088`
    - `T_HR 0.0018`
    - `T_MAP 0.0003`
  - full best epoch `1`:
    - `B_HR 0.0280`
    - `B_NDCG 0.0134`
    - `T_HR 0.0048`
    - `T_MAP 0.0017`
- decision:
  - `h4`는 `product_type15` baseline 위에서 fast-scout 신호는 있었지만 full validation에서 current champion `B_NDCG 0.0139`를 넘지 못함
  - 다음 실험 축은 `metadata-input` 조합으로 복귀

## Next Immediate Experiment

- family: `evaluation-policy`
- hypothesis: current champion도 `benchmark-best`와 `test-best`가 갈리므로, champion 승격 이후에는 dual-best report를 기본 artifact로 운영하는 것이 맞다.
- baseline:
  - single best checkpoint only
- treatment:
  - dual-best report + direct checkpoint evaluation

## Latest Metadata-Input Result

- `department_scale = 1.5`
  - fast-scout verdict: `FAIL`
  - observed:
    - `B_NDCG 0.0066`
    - `B_HR 0.0167`
- `department_scale = 0.5`
  - fast-scout verdict: `PASS`
  - fast-scout observed:
    - `B_NDCG 0.0091`
    - `B_HR 0.0190`
  - full verdict: `FAIL`
  - full observed:
    - `[0 epoch] B_NDCG 0.0082`
    - `[0 epoch] B_HR 0.0178`
    - `[0 epoch] T_HR 0.0036`
    - `[0 epoch] T_MAP 0.0011`
- interpretation:
  - `department` weighting은 benchmark보다 test 쪽 metric에 더 민감하게 작동하는 경향이 있음
  - `product_type_scale = 1.5`는 full validation까지 PASS해서 새 benchmark champion이 됨
  - `garment_group` 추가 weighting은 `1.5`, `1.2` 모두 champion을 넘지 못함
  - `product_type15 + department05`는 fast-scout `PASS`였지만 full best `B_NDCG 0.0108`로 실패
  - `product_type17`도 fast-scout `PASS`였지만 full best `B_NDCG 0.0129`로 실패
  - 현재까지 `metadata-input` 축에서 full champion으로 남은 건 `product_type_scale = 1.5` 단일 weighting 뿐
  - 이 축은 당분간 냉각하고 `evaluation-policy`와 `dual-best` 운영 정리로 이동
