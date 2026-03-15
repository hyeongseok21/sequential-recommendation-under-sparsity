# Eval Checkpoint Direct Compare

## 목적

- 저장된 checkpoint를 재학습 없이 바로 평가해서, `benchmark-best`와 `test-best` 차이를 artifact 수준에서 확인한다.

## 변경 내용

- [`hm_refactored/train.py`](../../hm_refactored/train.py)
  - `--eval_checkpoint` 추가
  - 지정한 epoch checkpoint를 로드해 benchmark/test만 실행
  - 결과를 `eval_checkpoint_<epoch>.json`으로 저장
  - PyTorch 2.6+ 호환을 위해 `torch.load(..., weights_only=False)` 명시

## 검증 대상

- config:
  - [`hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`](../../hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json)
- 비교 checkpoint:
  - epoch `1`
  - epoch `2`

## 직접 평가 결과

### checkpoint 1

- artifact:
  - [`hm_refactored/hm_out/checkpoints/hm_m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15/eval_checkpoint_000001.json`](../../hm_refactored/hm_out/checkpoints/hm_m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15/eval_checkpoint_000001.json)
- `B_HR 0.0291`
- `B_NDCG 0.0137`
- `T_HR 0.0054`
- `T_MAP 0.0016`

### checkpoint 2

- artifact:
  - [`hm_refactored/hm_out/checkpoints/hm_m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15/eval_checkpoint_000002.json`](../../hm_refactored/hm_out/checkpoints/hm_m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15/eval_checkpoint_000002.json)
- `B_HR 0.0250`
- `B_NDCG 0.0120`
- `T_HR 0.0042`
- `T_MAP 0.0020`

## 해석

- current champion run에서 benchmark-best는 epoch `1`, test-best는 epoch `2`다.
- 따라서 이제는 `loss.txt`를 다시 파싱하지 않고도 저장된 checkpoint를 직접 비교해 운영 판단을 내릴 수 있다.
- 이는 `evaluation-gap`을 직접 다루는 `P4` 도구로 충분히 유효하다.
