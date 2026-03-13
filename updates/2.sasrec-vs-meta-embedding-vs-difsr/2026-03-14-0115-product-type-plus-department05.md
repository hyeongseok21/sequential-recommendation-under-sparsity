# Product Type Plus Department Downweight

## 가설

- `product_type_scale = 1.5` champion 위에서 `department_scale = 0.5`를 결합하면 benchmark ranking이 더 좋아질 수 있다.

## 비교 조건

### Baseline

- fast-scout: `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features_product_type15.json`
- full: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`

### Treatment

- fast-scout: `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features_product_type15_department05.json`
- full: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15_department05.json`

### 고정 조건

- `DIFSR`
- `fusion_type = concat`
- `history_meta_scale = 1.5`
- `product_type_scale = 1.5`
- `metadata_features = [product_type, department, garment_group]`
- `lr = 2e-4`
- `seed = 42`

## 결과

### Baseline

- fast-scout:
  - `B_HR 0.0190`
  - `B_NDCG 0.0080`
  - `T_HR 0.0024`
  - `T_MAP 0.0003`
- full best epoch `2`:
  - `B_HR 0.0309`
  - `B_NDCG 0.0139`
  - `T_HR 0.0042`
  - `T_MAP 0.0015`

### Treatment

- fast-scout:
  - `B_HR 0.0178`
  - `B_NDCG 0.0089`
  - `T_HR 0.0018`
  - `T_MAP 0.0003`
- full best epoch `2`:
  - `B_HR 0.0238`
  - `B_NDCG 0.0108`
  - `T_HR 0.0042`
  - `T_MAP 0.0013`

## 해석

- `department_scale = 0.5`는 fast-scout에서는 긍정 신호가 있었다.
- 하지만 full budget에서는 benchmark ranking이 크게 꺾였다.
- 즉 `department` downweight는 short-budget에선 regularization처럼 보였지만, 충분히 학습시키면 champion보다 약하다.
- 현재 `metadata-input` 축에서는 `product_type_scale = 1.5` 단일 weighting이 가장 안정적이다.

## 현재 판단

- gate verdict: `FAIL`
- champion: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json` 유지

## 다음 후보

1. `product_type_scale` 미세조정 (`1.7` 또는 `2.0`) fast-scout
