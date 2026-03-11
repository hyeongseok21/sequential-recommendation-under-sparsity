# DIF-SR concat lr2e-4 weight decay 축소 예비 비교

## 가설

- 새 champion `concat DIF-SR + lr=2e-4`에서 `weight_decay`를 `0.001`에서 `0.0005`로 낮추면 over-regularization이 줄어 성능이 더 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4.json`
- 구조: `concat DIF-SR`
- 조건: `lr=2e-4`, `weight_decay=0.001`

### Treatment

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_wd5e4.json`
- 변경 축: `weight_decay=0.0005`

### 고정 조건

- `seed=42`
- `batch_size=16`
- `train_epoch=3`
- `drop_out=0.1`
- `seq_len=30`
- `fusion_type=concat`
- 동일 dataset / 동일 benchmark / 동일 test

## 중간 결과

### Baseline

- best epoch: `1`
- `B_HR 0.0303`
- `B_NDCG 0.0135`
- `T_HR 0.0036`
- `T_MAP 0.0012`

### Treatment

- start benchmark: `B_HR 0.0125`, `B_NDCG 0.0057`
- epoch 0:
  - `BPRLoss 10.0975`
  - `B_HR 0.0220`
  - `B_NDCG 0.0101`
  - `T_HR 0.0024`
  - `T_MAP 0.0013`

## 해석

- epoch 0 시점에서 treatment는 baseline champion의 best 성능보다 이미 낮다.
- `B_HR`, `B_NDCG` 모두 개선 신호가 없고, benchmark 기준으로는 소폭 열세다.
- gain signal이 약한 반면 runtime 비용은 그대로 크기 때문에, 이 축은 조기 중단하는 것이 합리적이다.
- 따라서 이번 결과는 "weight decay를 낮추면 좋아진다"는 가설을 지지하지 않는다.

## 현재 판단

- gate verdict: FAIL
- champion 유지: `concat DIF-SR + lr=2e-4`
- 이번 run은 epoch 0 이후 추가 gain 가능성이 낮다고 보고 조기 중단했다.

## 다음 후보

1. `weight_decay` 대신 `clip_grad_ratio`를 완화해 optimization 병목 확인
2. 또는 hyperparameter보다 candidate scoring / fusion 세부 구조로 다시 이동
3. 현재 champion의 best epoch checkpoint를 기준 모델로 고정
