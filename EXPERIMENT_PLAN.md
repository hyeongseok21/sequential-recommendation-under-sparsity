# Experiment Plan

## Current Champion

- config: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
- primary metric: `B_NDCG`
- current best full score:
  - `B_HR 0.0291`
  - `B_NDCG 0.0137`
  - `T_HR 0.0054`
  - `T_MAP 0.0016`

## Goal

1. `h4`가 full budget에서도 유지되는지 확인
2. all-features를 줄이는 대신 더 잘 쓰는 방향을 찾기
3. `benchmark-best`와 `test-best` gap을 줄일 수 있는지 확인

## Backlog

### `attention-capacity`

1. `all_features + h4 + batch_size 32`
2. `all_features + h4 + drop_out 0.05`
3. 앞선 후보가 PASS할 때만 `all_features + h4 + batch_size 32 + drop_out 0.05`

### `metadata-input`

1. `department_scale > 1.0`
2. `product_type_scale > 1.0`
3. `garment_group_scale` 약화 또는 유지 비교

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

- family: `metadata-input`
- hypothesis: `all_features`를 유지한 채 `department` feature 비중만 높이면, subset을 줄이지 않고도 sequence-side metadata quality를 개선할 수 있다.
- baseline:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features.json`
- treatment:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features_department15.json`
- expected gate:
  - fast-scout `B_NDCG`가 baseline `0.0075`를 초과
  - `B_HR` 또는 `T_MAP`이 동반 악화되지 않을 것
