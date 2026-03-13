# metadata input department weighting

## Hypotheses

1. `department_scale = 1.5`는 all-features에서 `department` 신호를 강화해 fast-scout benchmark quality를 높일 수 있다.
2. `department_scale = 0.5`는 `department`의 과한 비중을 줄여 fast-scout과 full validation 모두에서 더 나은 ranking quality를 낼 수 있다.

## Mutations

- fast-scout upweight:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features_department15.json`
- fast-scout downweight:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_all_features_department05.json`
- full validation:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_department05.json`

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

### `department_scale = 1.5`

- fast-scout:
  - `BPRLoss 51.8061`
  - `B_HR 0.0167`
  - `B_NDCG 0.0066`
  - `T_HR 0.0006`
  - `T_MAP 0.0002`
- verdict:
  - `FAIL`

### `department_scale = 0.5`

- fast-scout:
  - `BPRLoss 45.4546`
  - `B_HR 0.0190`
  - `B_NDCG 0.0091`
  - `T_HR 0.0018`
  - `T_MAP 0.0003`
- fast-scout verdict:
  - `PASS`

- full validation epoch 0:
  - `BPRLoss 9.6818`
  - `B_HR 0.0178`
  - `B_NDCG 0.0082`
  - `T_HR 0.0036`
  - `T_MAP 0.0011`
- full validation verdict:
  - `FAIL`
  - current champion `B_NDCG 0.0137`과 격차가 커서 early-stop

## Learning

- `department`는 단일 metadata feature로는 강하지만, all-features 조합에서 비중을 키우면 benchmark ranking이 약해짐
- `department` 비중을 줄이면 fast-scout에선 좋아지지만 full validation에선 benchmark 기준으로 유지되지 않음
- 대신 `T_HR`, `T_MAP`은 개선돼서 `evaluation-gap`이 다시 드러남
- 다음 weighting 축은 `department`보다 `product_type` 또는 `garment_group` 쪽이 더 타당함
