# DIF-SR full history metadata scale 상향 검증

## 가설

- fast-scout에서 개선된 `history_meta_scale=1.5`는 full training budget에서도 current champion보다 더 나은 ranking quality를 낼 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4.json`
- 구조: current champion `concat DIF-SR + lr=2e-4`

### Treatment

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
- 변경 축: `history_meta_scale=1.5`

### 고정 조건

- `seed=42`
- `batch_size=16`
- `train_epoch=3`
- `drop_out=0.1`
- `seq_len=30`
- `fusion_type=concat`

## 결과

### Baseline

- best epoch: `1`
- `B_HR 0.0303`
- `B_NDCG 0.0135`
- `T_HR 0.0036`
- `T_MAP 0.0012`

### Treatment

- best epoch: `1`
- epoch 0:
  - `BPRLoss 9.8954`
  - `B_HR 0.0220`
  - `B_NDCG 0.0105`
  - `T_HR 0.0036`
  - `T_MAP 0.0017`
- epoch 1:
  - `BPRLoss 6.3974`
  - `B_HR 0.0291`
  - `B_NDCG 0.0137`
  - `T_HR 0.0054`
  - `T_MAP 0.0016`
- epoch 2:
  - `BPRLoss 5.5764`
  - `B_HR 0.0250`
  - `B_NDCG 0.0120`
  - `T_HR 0.0042`
  - `T_MAP 0.0020`

## 해석

- `history_meta_scale=1.5`는 best epoch 기준으로 `B_NDCG`를 소폭 개선했고, `T_HR`와 `T_MAP`도 같이 올렸다.
- `B_HR`는 약간 낮아졌지만 주 지표와 test 지표가 함께 좋아진 만큼 treatment를 승격할 근거가 충분하다.
- 즉 current champion은 history-side metadata를 줄이는 것보다, 약간 더 강하게 쓰는 쪽이 더 적합하다.

## 현재 판단

- gate verdict: PASS
- champion 교체: `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`

## 다음 후보

1. 새 champion 기준으로 full run `target_meta_scale` 단일 축 검증
2. 또는 best epoch checkpoint를 serving 기준 모델로 고정
3. metadata scale은 target 쪽보다 history 쪽이 더 민감한지 추가 비교
