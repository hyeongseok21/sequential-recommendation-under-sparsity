# Current Champion Dual-Best Report

## 목적

- current champion에서 `benchmark-best`와 `test-best`를 한 번에 읽을 수 있는 운영 리포트를 남긴다.

## 대상

- checkpoint dir:
  - [`hm_refactored/hm_out/checkpoints/hm_m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15`](../../hm_refactored/hm_out/checkpoints/hm_m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15)
- report:
  - [`data/metrics/dual_best_report_difsr_hms15.json`](../../data/metrics/dual_best_report_difsr_hms15.json)

## 요약

### benchmark-best

- epoch `1`
- `B_HR 0.0291`
- `B_NDCG 0.0137`
- `T_HR 0.0054`
- `T_MAP 0.0016`

### test-best

- epoch `2`
- `B_HR 0.0250`
- `B_NDCG 0.0120`
- `T_HR 0.0042`
- `T_MAP 0.0020`

## 해석

- current champion은 benchmark 기준과 test 기준이 다른 epoch를 선호한다.
- benchmark ranking을 우선하면 epoch `1`, test MAP을 우선하면 epoch `2`가 더 낫다.
- 따라서 운영 판단은 이제 “어느 epoch가 최고인가”가 아니라 “어떤 목적 함수로 checkpoint를 고를 것인가”를 명시해야 한다.

## 현재 판단

- 연구용 champion은 그대로 benchmark-best epoch `1`
- 운영 리포트는 앞으로 benchmark-best / test-best를 같이 보여주는 방식이 맞다.
