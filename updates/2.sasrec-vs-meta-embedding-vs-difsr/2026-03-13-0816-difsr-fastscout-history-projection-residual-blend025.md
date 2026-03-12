# DIF-SR fast-scout history projection residual blend 0.25 비교

## 가설

- residual blend `0.5`가 benchmark보다 test 쪽으로 치우쳤으니, blend를 `0.25`로 낮추면 benchmark ranking이 더 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json`
- 조건: `history_meta_scale=1.5`

### Treatment

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_hprojblend025.json`
- 변경 축: `use_history_meta_projection=true`, `history_meta_residual_blend=0.25`

## 결과

### Baseline

- `B_HR 0.0178`
- `B_NDCG 0.0075`
- `T_HR 0.0012`
- `T_MAP 0.0002`

### Treatment

- `BPRLoss 48.1436`
- `B_HR 0.0107`
- `B_NDCG 0.0055`
- `T_HR 0.0024`
- `T_MAP 0.0005`

## 해석

- blend를 `0.25`로 낮추자 benchmark ranking이 크게 무너졌다.
- test 쪽은 조금 올라갔지만, `B_NDCG` 기준으로는 명확한 FAIL이다.
- 따라서 residual blend 축은 `0.5`가 local optimum에 가깝고, 더 낮추는 방향은 유효하지 않다.

## 현재 판단

- gate verdict: FAIL on fast-scout
- champion 유지: `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`
- next focus: `evaluation-gap`
