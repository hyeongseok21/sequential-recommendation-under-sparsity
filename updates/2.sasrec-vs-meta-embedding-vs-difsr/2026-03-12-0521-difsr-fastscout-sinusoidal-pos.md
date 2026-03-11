# DIF-SR fast-scout sinusoidal position 비교

## 가설

- current fast-scout champion 조건에서 `learnable_pos=false`로 바꾸면, positional regularization이 더 안정적으로 작동해 ranking quality가 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4.json`
- 구조: `concat DIF-SR`
- 조건: `batch_size=64`, `train_epoch=1`, `lr=2e-4`, `learnable_pos=true`

### Treatment

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_sinusoidal.json`
- 변경 축: `learnable_pos=false`

### 고정 조건

- `seed=42`
- `drop_out=0.1`
- `seq_len=30`
- `fusion_type=concat`
- 동일 dataset / 동일 benchmark / 동일 test

## 결과

### Baseline

- best epoch: `0`
- `BPRLoss 48.4609`
- `B_HR 0.0137`
- `B_NDCG 0.0063`
- `T_HR 0.0018`
- `T_MAP 0.0004`

### Treatment

- best epoch: `0`
- `BPRLoss 47.9495`
- `B_HR 0.0119`
- `B_NDCG 0.0056`
- `T_HR 0.0018`
- `T_MAP 0.0003`

## 해석

- sinusoidal positional encoding은 loss는 소폭 낮췄지만 ranking quality는 baseline보다 떨어졌다.
- 특히 `B_NDCG`와 `B_HR`가 모두 낮아, 현재 `concat DIF-SR` fast-scout 조건에서는 learnable positional embedding이 더 적합하다.
- 이는 현 champion이 위치 정보까지 학습적으로 맞추는 쪽에서 이득을 보고 있음을 시사한다.

## 현재 판단

- gate verdict: FAIL
- champion 유지: `concat DIF-SR + lr=2e-4`

## 다음 후보

1. positional encoding보다 sequence-side metadata interaction을 직접 조정하는 구조 실험으로 이동
2. fast-scout는 유지하되, 구조 실험은 epoch 0/1 signal 위주로 early-stop 판단
3. current champion checkpoint를 기준으로 inference-side 검증도 병행
