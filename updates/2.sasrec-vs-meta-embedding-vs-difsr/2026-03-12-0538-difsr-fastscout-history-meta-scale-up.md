# DIF-SR fast-scout history metadata scale 상향 비교

## 가설

- history token 쪽 metadata 비중을 더 높이면 sequence-side interaction이 강화되어 ranking quality가 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4.json`
- 조건: `history_meta_scale=1.0`

### Treatment

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json`
- 변경 축: `history_meta_scale=1.5`

### 고정 조건

- `seed=42`
- `batch_size=64`
- `train_epoch=1`
- `lr=2e-4`
- `drop_out=0.1`
- `seq_len=30`
- `fusion_type=concat`

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
- `BPRLoss 47.8306`
- `B_HR 0.0178`
- `B_NDCG 0.0075`
- `T_HR 0.0012`
- `T_MAP 0.0002`

## 해석

- history-side metadata 비중을 높이자 `B_HR`, `B_NDCG`가 함께 개선됐다.
- 즉 current fast-scout champion은 metadata를 약하게 쓰는 것보다 조금 더 강하게 쓰는 방향에서 이득이 있다.
- 다만 test 지표는 소폭 하락했으므로, full validation 전에는 champion을 fast-scout 후보로만 올리는 편이 맞다.

## 현재 판단

- gate verdict: PASS on fast-scout
- full champion 유지: `concat DIF-SR + lr=2e-4`
- next candidate: `history_meta_scale=1.5` full validation

## 다음 후보

1. `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4` 기준으로 `history_meta_scale=1.5` full run
2. full run에서도 유지되면 champion 승격
3. 유지되지 않으면 fast-scout 전용 heuristic으로만 남김
