# Product Type Champion H4 Revisit

## 가설

- `product_type_scale = 1.5` champion 위에서 `n_heads = 4`를 쓰면 benchmark ranking이 더 좋아질 수 있다.

## 비교 조건

### Baseline

- fast-scout: `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features_product_type15.json`
- full: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`

### Treatment

- fast-scout: `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features_product_type15_h4.json`
- full: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15_h4.json`

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
  - `B_HR 0.0208`
  - `B_NDCG 0.0088`
  - `T_HR 0.0018`
  - `T_MAP 0.0003`
- full best epoch `1`:
  - `B_HR 0.0280`
  - `B_NDCG 0.0134`
  - `T_HR 0.0048`
  - `T_MAP 0.0017`

## 해석

- `h4`는 fast-scout에서는 다시 유의미한 상승을 보였다.
- 하지만 full validation에서는 benchmark 기준으로 current champion을 넘지 못했다.
- test 쪽 metric은 일부 개선됐지만 current policy는 `B_NDCG` 우선이므로 champion 승격 조건을 만족하지 못한다.
- runtime도 `h2` baseline보다 더 무거워 운영 비용 측면의 이득이 없다.

## 현재 판단

- gate verdict: `FAIL`
- champion: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json` 유지

## 다음 후보

1. `product_type15 + department05` 조합 fast-scout
