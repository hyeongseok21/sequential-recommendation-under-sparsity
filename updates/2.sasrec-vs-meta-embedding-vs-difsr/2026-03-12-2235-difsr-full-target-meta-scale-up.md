# DIF-SR full target metadata scale 상향 검증

## 가설

- history-side metadata를 강화한 current champion에서 target item 쪽 metadata 비중도 같이 높이면 full training budget에서도 ranking quality가 더 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
- 구조: current champion `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`

### Treatment

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_tms15.json`
- 변경 축: `target_meta_scale=1.5`

### 고정 조건

- `seed=42`
- `batch_size=16`
- `train_epoch=3`
- `drop_out=0.1`
- `seq_len=30`
- `fusion_type=concat`

## 중간 결과

### Baseline

- best epoch: `1`
- `B_HR 0.0291`
- `B_NDCG 0.0137`
- `T_HR 0.0054`
- `T_MAP 0.0016`

### Treatment

- epoch 0:
  - `BPRLoss 10.7538`
  - `B_HR 0.0244`
  - `B_NDCG 0.0110`
  - `T_HR 0.0036`
  - `T_MAP 0.0017`
- epoch 1:
  - `BPRLoss 6.4554`
  - `B_HR 0.0286`
  - `B_NDCG 0.0127`
  - `T_HR 0.0042`
  - `T_MAP 0.0008`

## 해석

- fast-scout에서는 좋아졌지만, full run에서는 current champion의 `B_NDCG`를 넘지 못했다.
- epoch 1 기준으로도 benchmark와 test가 모두 baseline보다 약간 낮거나 비슷한 수준이다.
- 따라서 target-side metadata scaling은 short-budget signal로는 유망해 보여도, full budget에서는 current champion보다 강하지 않다.

## 현재 판단

- gate verdict: FAIL
- champion 유지: `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`
- baseline을 넘지 못해 epoch 1에서 조기 중단했다.

## 다음 후보

1. target metadata scale보다 sequence-side attention interaction을 더 직접 바꾸는 구조 실험
2. 또는 current champion의 best epoch checkpoint를 기준 모델로 고정하고 inference-side 검증
3. fast-scout와 full run의 차이가 큰 축은 후보군에서 우선순위를 낮춤
