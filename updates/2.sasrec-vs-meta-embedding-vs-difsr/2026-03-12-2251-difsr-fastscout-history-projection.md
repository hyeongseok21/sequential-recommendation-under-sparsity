# DIF-SR fast-scout history 전용 projection 비교

## 가설

- history path가 target path와 같은 metadata projection을 공유하는 대신, history 전용 projection을 따로 쓰면 sequence-side metadata interaction이 더 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json`
- 조건: `history_meta_scale=1.5`

### Treatment

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_hproj.json`
- 변경 축: `use_history_meta_projection=true`

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
- `BPRLoss 50.4051`
- `B_HR 0.0143`
- `B_NDCG 0.0074`
- `T_HR 0.0042`
- `T_MAP 0.0014`

## 해석

- history 전용 projection은 test 지표는 좋아졌지만, primary metric인 `B_NDCG`는 baseline보다 소폭 낮았다.
- 현재 fast-scout 기준으로는 sequence-side representation을 분리하는 것보다, shared projection + strong history scale 조합이 더 안정적이다.
- 따라서 이번 mutation은 승격하지 않는다.

## 현재 판단

- gate verdict: FAIL
- champion 유지: `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`

## 다음 후보

1. projection 분리보다 history-side metadata 강도 자체를 조금 더 올리는 축 (`history_meta_scale=2.0`)
2. sequence-side attention interaction을 직접 바꾸는 구조 실험
3. current champion checkpoint 기준 inference 검증
