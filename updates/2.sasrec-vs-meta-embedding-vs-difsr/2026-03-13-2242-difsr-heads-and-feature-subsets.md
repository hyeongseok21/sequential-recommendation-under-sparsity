# DIF-SR head 수 / metadata subset 확장 비교

## 목적
- metadata subset만 볼 때의 결과를 `n_heads` 축과 함께 다시 본다.
- `1 / 2 / 4 heads` 중 어떤 구성이 가장 좋은지 확인하고, 가장 유망한 subset과 조합해 full validation 후보를 고른다.

## 현재 기준
- full champion:
  - `all_features`
  - `n_heads=2`
  - `history_meta_scale=1.5`
  - `B_HR 0.0291`
  - `B_NDCG 0.0137`
  - `T_HR 0.0054`
  - `T_MAP 0.0016`

## fast-scout: head 수 비교 (`all_features`)
- `h1`
  - `B_HR 0.0190`
  - `B_NDCG 0.0081`
  - `T_HR 0.0018`
  - `T_MAP 0.0003`
- `h2`
  - `B_HR 0.0178`
  - `B_NDCG 0.0075`
  - `T_HR 0.0012`
  - `T_MAP 0.0002`
- `h4`
  - `B_HR 0.0196`
  - `B_NDCG 0.0086`
  - `T_HR 0.0030`
  - `T_MAP 0.0005`

## head 수 결론
- fast-scout 기준 ranking은 `h4 > h1 > h2`
- 따라서 논문 기억과 비슷하게 이 저장소에서도 `4 heads`가 가장 유망한 head 수로 보였다.

## fast-scout: `h4` 기준 subset 재비교
- `product_type + garment_group + h4`
  - `B_HR 0.0202`
  - `B_NDCG 0.0091`
  - `T_HR 0.0012`
  - `T_MAP 0.0001`
- `department only + h4`
  - `B_HR 0.0143`
  - `B_NDCG 0.0078`
  - `T_HR 0.0036`
  - `T_MAP 0.0009`
- `all_features + h4`
  - `B_HR 0.0196`
  - `B_NDCG 0.0086`
  - `T_HR 0.0030`
  - `T_MAP 0.0005`

## fast-scout 결론
- `h4` 기준 best subset은 `product_type + garment_group`
- `department only`는 단일 feature로는 여전히 강한 편이지만 `h4` 조합에선 top이 아니었다.

## full validation

### 1. `product_type + garment_group + h4`
- config: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_h4_product_type_garment_group.json`
- epoch 0:
  - `B_HR 0.0190`
  - `B_NDCG 0.0081`
  - `T_HR 0.0012`
  - `T_MAP 0.0007`
- current champion 대비 `B_NDCG`가 크게 낮아 조기 중단

### 2. `all_features + h4`
- config: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_h4.json`
- start epoch만 확인 후 first epoch training runtime이 과도하게 증가
- current champion 대비 runtime cost가 너무 커 조기 중단

## 최종 해석
- `4 heads`는 fast-scout signal 자체는 가장 좋다.
- 하지만 full budget으로 올리면 품질 유지 또는 runtime 측면에서 아직 champion을 넘지 못했다.
- 따라서 현재 champion은 그대로 `all_features + h2 + history_meta_scale=1.5` 유지다.

## 다음 가설
- `h4` 자체는 유망하므로, 전체 3 epoch full validation보다 더 싸게 볼 수 있는 중간 budget 검증이 필요하다.
- 또는 `h4`에 맞춘 `batch_size`, `lr`, `drop_out` 재조정이 필요할 수 있다.
