# DIF-SR concat full lr=2e-4 재검증

## 가설

- fast-scout에서 개선된 `lr=2e-4`는 full training budget에서도 현재 `concat DIF-SR` champion보다 더 나은 benchmark 성능을 낼 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat.json`
- 구조: `concat DIF-SR`
- 조건: `batch_size=16`, `train_epoch=3`, `lr=1e-4`

### Treatment

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4.json`
- 구조: 동일
- 변경 축: `lr=2e-4`

### 고정 조건

- `seed=42`
- `drop_out=0.1`
- `seq_len=30`
- `fusion_type=concat`
- 동일 dataset / 동일 benchmark / 동일 test

## 결과

### Baseline

- best epoch: `2`
- best benchmark: `B_HR 0.0220`, `B_NDCG 0.0103`
- best test: `T_HR 0.0042`, `T_MAP 0.0013`

### Treatment

- best epoch: `1`
- epoch 0:
  - `BPRLoss 10.0109`
  - `B_HR 0.0226`
  - `B_NDCG 0.0103`
  - `T_HR 0.0024`
  - `T_MAP 0.0015`
- epoch 1:
  - `BPRLoss 6.4559`
  - `B_HR 0.0303`
  - `B_NDCG 0.0135`
  - `T_HR 0.0036`
  - `T_MAP 0.0012`
- epoch 2:
  - `BPRLoss 5.6105`
  - `B_HR 0.0238`
  - `B_NDCG 0.0104`
  - `T_HR 0.0030`
  - `T_MAP 0.0010`

## 해석

- `lr=2e-4`는 epoch 1에서 benchmark 기준으로 기존 champion을 분명하게 넘어섰다.
- `B_HR`와 `B_NDCG` 개선 폭이 크고, fast-scout에서 보였던 방향성과도 일치한다.
- test 지표는 baseline best보다 약간 낮지만, 현재 실험 흐름에서는 benchmark ranking quality를 주 지표로 보고 있으므로 treatment를 승격할 근거가 충분하다.
- 다만 higher learning rate에서는 epoch 1 이후 다시 일부 하락하므로, early-stop 관점에서는 `best epoch=1` 관리가 중요하다.

## 현재 판단

- gate verdict: PASS
- champion 교체: `concat DIF-SR + lr=2e-4`

## 다음 후보

1. 새 champion 기준으로 `weight_decay`를 소폭 낮춰 over-regularization 여부 확인
2. 또는 `save/best epoch=1` 기준으로 실제 inference checkpoint를 고정
3. 이후에는 hyperparameter보다 candidate scoring 또는 metadata fusion 세부 구조 개선으로 이동
