# metadata input product type plus garment weighting

## Hypothesis

`product_type_scale = 1.5` 위에 `garment_group_scale = 1.5`를 추가하면, `product_type + garment_group` 조합 신호를 all-features 안에서 더 강하게 살려 current champion을 넘을 수 있다.

## Mutations

- fast-scout:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features_product_type15_garment15.json`
- full validation:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15_garment15.json`

## Baseline

- benchmark champion:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`
  - `B_HR 0.0309`
  - `B_NDCG 0.0139`
  - `T_HR 0.0042`
  - `T_MAP 0.0015`

## Results

### Fast-Scout

- `BPRLoss 51.8340`
- `B_HR 0.0208`
- `B_NDCG 0.0088`
- `T_HR 0.0030`
- `T_MAP 0.0004`
- verdict:
  - `PASS`

### Full Validation

- epoch 0:
  - `BPRLoss 10.4656`
  - `B_HR 0.0208`
  - `B_NDCG 0.0102`
  - `T_HR 0.0048`
  - `T_MAP 0.0019`
- epoch 1:
  - `BPRLoss 6.4673`
  - `B_HR 0.0256`
  - `B_NDCG 0.0119`
  - `T_HR 0.0048`
  - `T_MAP 0.0016`
- epoch 2:
  - `BPRLoss 5.6020`
  - `B_HR 0.0274`
  - `B_NDCG 0.0122`
  - `T_HR 0.0048`
  - `T_MAP 0.0013`
- verdict:
  - `FAIL`

## Learning

- `garment_group`를 `1.5`까지 같이 올리면 fast-scout은 좋아지지만 full validation에서 benchmark ranking이 유지되지 않는다.
- 현재 champion은 `product_type` 강화까지만이 적정선이고, `garment_group`는 같은 강도로 올리면 과한 weighting일 가능성이 높다.
- 다음 실험은 같은 방향을 버리기보다 `garment_group_scale`을 더 작은 값으로 낮춰 re-test하는 쪽이 타당하다.
