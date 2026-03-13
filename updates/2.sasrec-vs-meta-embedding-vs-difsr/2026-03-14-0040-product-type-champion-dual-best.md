# product type champion dual best

## 대상

- benchmark champion:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`

## 목적

- 새 champion도 `benchmark-best`와 `test-best`가 갈리는지 직접 checkpoint 평가로 확인한다.

## 평가 checkpoint

- epoch `1`
- epoch `2`

## 결과

- benchmark-best:
  - epoch `2`
  - `B_HR 0.0309`
  - `B_NDCG 0.0139`
  - `T_HR 0.0042`
  - `T_MAP 0.0015`
- test-best:
  - epoch `1`
  - `B_HR 0.0286`
  - `B_NDCG 0.0136`
  - `T_HR 0.0059`
  - `T_MAP 0.0021`

## 해석

- 새 champion도 이전과 동일하게 `benchmark-best`와 `test-best`가 다르다.
- 즉 `product_type_scale = 1.5` 승격은 benchmark 기준으로는 타당하지만, test-oriented inference 관점에서는 epoch `1` checkpoint도 같이 관리할 가치가 있다.
- 현재 운영 결론은:
  - research champion: epoch `2`
  - test-best companion checkpoint: epoch `1`

## 산출물

- `data/metrics/dual_best_report_difsr_product_type15.json`
