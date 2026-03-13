# metadata input product type plus garment 1.2 fast

## Hypothesis

`product_type_scale = 1.5` 위에 `garment_group_scale = 1.2`를 추가하면, `1.5`가 과했던 `garment_group` 신호를 완만하게 넣어 fast-scout benchmark를 더 올릴 수 있다.

## Mutation

- `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features_product_type15_garment12.json`

## Baseline

- `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features_product_type15.json`
- `B_HR 0.0190`
- `B_NDCG 0.0080`
- `T_HR 0.0024`
- `T_MAP 0.0003`

## Result

- `BPRLoss 50.6852`
- `B_HR 0.0184`
- `B_NDCG 0.0080`
- `T_HR 0.0030`
- `T_MAP 0.0004`

## Verdict

- `FAIL`

## Learning

- primary metric은 동률이고 `B_HR`가 내려가서 full validation로 올릴 근거가 부족하다.
- `garment_group`를 `1.5`로 올리는 것도, `1.2`로 완만하게 올리는 것도 benchmark ranking 기준으론 `product_type15` baseline을 넘지 못했다.
- 현재 `metadata-input` 축에서는 `product_type_scale = 1.5`가 가장 안정적인 champion 방향이다.
