# bs32 h4 all-features validation

## Hypothesis

`all_features + h4`는 `batch_size 32`에서 runtime과 quality의 균형이 좋아져 full champion에 더 가까워질 수 있다.

## Mutation

- treatment config:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs32_seq30_do01_concat_lr2e4_hms15_all_features_h4.json`

## Baseline

- `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
- best full metrics:
  - `B_HR 0.0291`
  - `B_NDCG 0.0137`
  - `T_HR 0.0054`
  - `T_MAP 0.0016`

## Result

- start benchmark:
  - `B_HR 0.0119`
  - `B_NDCG 0.0054`
- epoch 0:
  - `BPRLoss 21.7379`
  - `B_HR 0.0220`
  - `B_NDCG 0.0092`
  - `T_HR 0.0024`
  - `T_MAP 0.0005`

## Verdict

- `FAIL`

## Learning

- `bs32`로 줄이면서 `bs16 + h4`의 runtime stall은 피했지만, current champion과의 quality gap은 여전히 큼
- `h4`는 fast-scout에서는 강했지만 full-budget에 가까워질수록 과한 capacity 혹은 runtime penalty가 더 크게 드러남
- 다음 축은 `attention-capacity`가 아니라 `metadata-input`의 feature-specific weighting이 더 적절함
