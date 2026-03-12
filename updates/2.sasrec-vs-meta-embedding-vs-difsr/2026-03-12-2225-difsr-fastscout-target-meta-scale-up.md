# DIF-SR fast-scout target metadata scale 상향 비교

## 가설

- history-side metadata를 강화한 상태에서 target item 쪽 metadata 비중도 함께 높이면 ranking quality가 더 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json`
- 조건: `history_meta_scale=1.5`, `target_meta_scale=1.0`

### Treatment

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_tms15.json`
- 변경 축: `target_meta_scale=1.5`

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
- `BPRLoss 47.8306`
- `B_HR 0.0178`
- `B_NDCG 0.0075`
- `T_HR 0.0012`
- `T_MAP 0.0002`

### Treatment

- best epoch: `0`
- `BPRLoss 54.1179`
- `B_HR 0.0196`
- `B_NDCG 0.0079`
- `T_HR 0.0030`
- `T_MAP 0.0005`

## 해석

- target-side metadata 비중을 같이 높이자 benchmark와 test 지표가 모두 개선됐다.
- loss는 올라갔지만 ranking metric은 좋아졌기 때문에, 이 축은 optimization보다 representation alignment 측면에서 이득을 준 것으로 해석할 수 있다.
- 따라서 다음 단계는 full run에서 이 조합이 유지되는지 검증하는 것이다.

## 현재 판단

- gate verdict: PASS on fast-scout
- full champion 유지: `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`
- next candidate: `target_meta_scale=1.5` full validation

## 다음 후보

1. `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15` 기준으로 `target_meta_scale=1.5` full run
2. full run에서도 유지되면 champion 승격
3. 유지되지 않으면 target-side scaling은 fast-scout 전용 신호로만 남김
