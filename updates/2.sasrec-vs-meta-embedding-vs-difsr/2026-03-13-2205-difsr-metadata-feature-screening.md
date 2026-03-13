# DIF-SR metadata feature screening

## 목적
- `DIF-SR`에서 어떤 metadata feature를 임베딩하는 것이 가장 성능에 기여하는지 fast-scout으로 먼저 가려내고, 가장 유망한 subset만 full validation으로 올린다.

## baseline
- fast-scout baseline: `all_features`
  - config: `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features.json`
  - `B_HR 0.0178`
  - `B_NDCG 0.0075`
  - `T_HR 0.0012`
  - `T_MAP 0.0002`
- full baseline champion:
  - config: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
  - `B_HR 0.0291`
  - `B_NDCG 0.0137`
  - `T_HR 0.0054`
  - `T_MAP 0.0016`

## single feature 결과
- `product_type only`
  - `B_HR 0.0131`
  - `B_NDCG 0.0059`
  - `T_HR 0.0006`
  - `T_MAP 0.0002`
- `department only`
  - `B_HR 0.0155`
  - `B_NDCG 0.0070`
  - `T_HR 0.0030`
  - `T_MAP 0.0007`
- `garment_group only`
  - `B_HR 0.0131`
  - `B_NDCG 0.0055`
  - `T_HR 0.0000`
  - `T_MAP 0.0000`

## pair feature 결과
- `product_type + department`
  - `B_HR 0.0131`
  - `B_NDCG 0.0062`
  - `T_HR 0.0024`
  - `T_MAP 0.0005`
- `product_type + garment_group`
  - `B_HR 0.0173`
  - `B_NDCG 0.0083`
  - `T_HR 0.0012`
  - `T_MAP 0.0001`
- `department + garment_group`
  - `B_HR 0.0143`
  - `B_NDCG 0.0070`
  - `T_HR 0.0030`
  - `T_MAP 0.0008`

## fast-scout ranking
1. `product_type + garment_group` : `B_NDCG 0.0083`
2. `all_features` : `B_NDCG 0.0075`
3. `department only` : `B_NDCG 0.0070`
4. `department + garment_group` : `B_NDCG 0.0070`
5. `product_type + department` : `B_NDCG 0.0062`
6. `product_type only` : `B_NDCG 0.0059`
7. `garment_group only` : `B_NDCG 0.0055`

## full validation
- treatment: `product_type + garment_group`
- config: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_product_type_garment_group.json`
- epoch 0 기준:
  - `B_HR 0.0178`
  - `B_NDCG 0.0075`
  - `T_HR 0.0012`
  - `T_MAP 0.0004`
- current champion 대비 `B_NDCG` 격차가 크고 runtime도 무거워 조기 중단

## 결론
- 단일 feature 중에서는 `department`가 가장 강했다.
- pair 중에서는 `product_type + garment_group`가 fast-scout best였다.
- 하지만 full validation에서는 current champion을 넘지 못했다.
- 따라서 현재 full champion은 그대로 `all_features + concat DIF-SR + lr=2e-4 + history_meta_scale=1.5` 유지다.

## 해석
- `department`는 혼자 써도 안정적인 signal을 준다.
- `product_type`와 `garment_group`의 조합은 짧은 예산에서는 강하지만 full budget에서는 유지되지 않았다.
- 현재 champion은 metadata 개수를 줄이는 것보다, all-feature 표현을 더 잘 활용하는 방향이 여전히 낫다.

## 다음 가설
- feature subset 자체보다 feature별 scale 또는 projection 분리를 보는 것이 더 유망하다.
- 특히 `department`는 강한 단일 signal이므로, all-feature를 유지하되 `department` 전용 weighting 실험으로 이어가는 것이 자연스럽다.
