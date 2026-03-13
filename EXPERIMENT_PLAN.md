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

- family: `attention-capacity`
- hypothesis: `all_features + h4`는 `batch_size 32`에서 runtime과 quality의 균형이 좋아져 full champion에 더 가까워질 수 있다.
- baseline:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
- treatment:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs32_seq30_do01_concat_lr2e4_hms15_all_features_h4.json`
- expected gate:
  - `B_NDCG`가 current champion에 근접하거나 개선
  - runtime penalty가 `bs16 + h4`보다 완화

## Latest Result

- family: `attention-capacity`
- treatment:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs32_seq30_do01_concat_lr2e4_hms15_all_features_h4.json`
- verdict: `FAIL`
- observed:
  - `[0 epoch] B_HR 0.0220`
  - `[0 epoch] B_NDCG 0.0092`
  - `[0 epoch] T_HR 0.0024`
  - `[0 epoch] T_MAP 0.0005`
- decision:
  - `bs16 + h4`의 runtime stall은 피했지만 current champion `B_NDCG 0.0137`과 격차가 커서 조기 중단
  - 다음 실험 축은 `metadata-input`의 feature-specific weighting으로 이동

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
  - 다음 우선순위는 `metadata-input` 추가 탐색보다 `evaluation-policy` 정리
