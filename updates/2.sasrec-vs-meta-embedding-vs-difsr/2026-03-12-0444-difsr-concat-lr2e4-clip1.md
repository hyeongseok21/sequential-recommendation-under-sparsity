# DIF-SR concat lr2e-4 clip_grad_ratio 완화 예비 비교

## 가설

- 새 champion `concat DIF-SR + lr=2e-4`에서 `clip_grad_ratio`를 `0.5`에서 `1.0`으로 완화하면 optimization bottleneck이 줄어 성능이 더 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4.json`
- 구조: `concat DIF-SR`
- 조건: `lr=2e-4`, `clip_grad_ratio=0.5`

### Treatment

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_clip1.json`
- 변경 축: `clip_grad_ratio=1.0`

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
  - `BPRLoss 10.0082`
  - `B_HR 0.0220`
  - `B_NDCG 0.0101`
  - `T_HR 0.0024`
  - `T_MAP 0.0013`

## 해석

- epoch 0 시점에서 treatment는 baseline champion의 best benchmark보다 낮다.
- `B_HR`, `B_NDCG` 모두 상승 신호가 없고, 수치상으로는 `weight_decay` 축소 treatment와 거의 같은 수준이다.
- 따라서 현재 champion 기준으로 gradient clipping이 주요 병목일 가능성은 낮다.
- 추가 epoch를 더 보더라도 승격 가능성이 낮다고 판단해 조기 중단했다.

## 현재 판단

- gate verdict: FAIL
- champion 유지: `concat DIF-SR + lr=2e-4`

## 다음 후보

1. hyperparameter 축보다 candidate scoring / metadata fusion 세부 구조로 다시 이동
2. 또는 best epoch checkpoint를 기준 모델로 고정하고 추론 품질 검증
3. 추가 튜닝을 한다면 `n_heads`나 `seq_len`보다 inference-side scoring 개선이 우선
