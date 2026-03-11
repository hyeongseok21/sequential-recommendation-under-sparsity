# DIF-SR fast-scout history metadata scale 비교

## 가설

- sequence-side metadata interaction이 과하면 ranking이 흔들릴 수 있다.
- history token 쪽 metadata 비중을 `1.0`에서 `0.5`로 낮추면 fast-scout 기준 성능이 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4.json`
- 조건: `history_meta_scale=1.0`

### Treatment

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms05.json`
- 변경 축: `history_meta_scale=0.5`

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
- `BPRLoss 51.1679`
- `B_HR 0.0131`
- `B_NDCG 0.0063`
- `T_HR 0.0018`
- `T_MAP 0.0003`

## 해석

- `B_NDCG`는 동률이지만 `B_HR`와 `T_MAP`가 모두 소폭 하락했다.
- 즉 history-side metadata 비중을 낮추는 것은 적어도 현재 fast-scout 조건에서는 이득이 없다.
- 현재 champion은 sequence token에서 metadata를 충분히 강하게 쓰는 쪽이 더 적합해 보인다.

## 현재 판단

- gate verdict: FAIL
- champion 유지: `concat DIF-SR + lr=2e-4`

## 다음 후보

1. metadata 강도를 줄이는 대신, sequence-side interaction 형태 자체를 바꾸는 구조 실험
2. fast-scout loop는 유지하되 동률이면 secondary metric으로 바로 FAIL 처리
3. 이후 구조 탐색은 feature scaling보다 attention interaction 쪽으로 이동
