# Dual Best Epoch Summary 추가

## 목적

- `evaluation-gap` 때문에 benchmark-best epoch와 test-best epoch가 다를 수 있으므로, 학습 종료 시 두 기준을 모두 summary로 남긴다.

## 변경 내용

- [`hm_refactored/train.py`](../../hm_refactored/train.py)
  - `epoch_summary.json` 저장 추가
  - 저장 항목:
    - checkpoint policy 기준 best epoch
    - benchmark-best epoch
    - test-best epoch

## 검증

- config:
  - [`hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json`](../../hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json)
- output:
  - [`hm_refactored/hm_out/checkpoints/hm_m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15/epoch_summary.json`](../../hm_refactored/hm_out/checkpoints/hm_m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15/epoch_summary.json)

## 해석

- current fast verification run에서는 benchmark-best와 test-best가 모두 epoch `0`으로 같았다.
- 하지만 이 summary 구조가 들어갔기 때문에 이후 full run에서 두 기준이 갈려도 artifact를 바로 비교할 수 있다.

## 현재 판단

- `evaluation-gap` 분석 결과를 운영 artifact 수준으로 흡수했다.
- 다음부터는 run 종료 후 `loss.txt`를 다시 파싱하지 않고도 dual-best 기준을 바로 확인할 수 있다.
