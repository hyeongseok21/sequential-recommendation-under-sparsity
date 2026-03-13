# metadata input product type weighting

## Hypothesis

`product_type` 비중을 `1.5`로 올리면, `product_type + garment_group` 조합에서 보였던 신호를 all-features 안에서도 더 잘 살려 current champion을 넘을 수 있다.

## Mutations

- fast-scout:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features_product_type15.json`
- full validation:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`

## Baselines

- fast-scout baseline:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features.json`
  - `B_HR 0.0178`
  - `B_NDCG 0.0075`
  - `T_HR 0.0012`
  - `T_MAP 0.0002`
- full baseline:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
  - `B_HR 0.0291`
  - `B_NDCG 0.0137`
  - `T_HR 0.0054`
  - `T_MAP 0.0016`

## Results

### Fast-Scout

- `BPRLoss 50.0405`
- `B_HR 0.0190`
- `B_NDCG 0.0080`
- `T_HR 0.0024`
- `T_MAP 0.0003`
- verdict:
  - `PASS`

### Full Validation

- epoch 0:
  - `BPRLoss 10.2146`
  - `B_HR 0.0232`
  - `B_NDCG 0.0105`
  - `T_HR 0.0048`
  - `T_MAP 0.0018`
- epoch 1:
  - `BPRLoss 6.4306`
  - `B_HR 0.0286`
  - `B_NDCG 0.0136`
  - `T_HR 0.0059`
  - `T_MAP 0.0021`
- epoch 2:
  - `BPRLoss 5.5915`
  - `B_HR 0.0309`
  - `B_NDCG 0.0139`
  - `T_HR 0.0042`
  - `T_MAP 0.0015`
- verdict:
  - `PASS`

## Champion Decision

- promote
- new benchmark champion:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`

## Learning

- `department` weighting은 full benchmark로 이어지지 않았지만, `product_type` weighting은 full validation까지 유지됐다.
- all-features를 버리지 않고도 feature importance를 조정해 champion을 넘길 수 있다는 근거가 생겼다.
- 다만 `T_MAP`는 이전 champion보다 약간 낮아져 `benchmark-best`와 `test-best`의 긴장은 여전히 남아 있다.
