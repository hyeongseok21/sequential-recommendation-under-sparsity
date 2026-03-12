# DIF-SR fast-scout history metadata scale 2.0 비교

## 가설

- current fast-scout champion보다 history-side metadata를 조금 더 강하게 쓰면 ranking quality가 더 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json`
- 조건: `history_meta_scale=1.5`

### Treatment

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms20.json`
- 변경 축: `history_meta_scale=2.0`

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
- `BPRLoss 47.8903`
- `B_HR 0.0173`
- `B_NDCG 0.0076`
- `T_HR 0.0030`
- `T_MAP 0.0004`

## 해석

- `B_NDCG`는 소폭 개선됐고 `T_HR`, `T_MAP`도 좋아졌다.
- `B_HR`는 조금 내려갔지만 전체적으로는 current fast-scout champion보다 나은 방향으로 보인다.
- 따라서 이 축은 full validation으로 올려볼 가치가 있다.

## 현재 판단

- gate verdict: PASS on fast-scout
- full champion 유지: `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`
- next candidate: `history_meta_scale=2.0` full validation

## 다음 후보

1. `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms20.json` full run
2. 유지되면 champion 승격
3. 유지되지 않으면 `history_meta_scale=1.5`를 최종 유지
