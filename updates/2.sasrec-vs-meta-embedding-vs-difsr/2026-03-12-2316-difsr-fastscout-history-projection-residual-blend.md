# DIF-SR fast-scout history projection residual blend 비교

## 가설

- history metadata 전용 projection을 shared projection의 대체가 아니라 residual blend로 넣으면 sequence-side interaction이 더 안정적으로 작동할 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json`
- 조건: `history_meta_scale=1.5`

### Treatment

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_hprojblend05.json`
- 변경 축: `use_history_meta_projection=true`, `history_meta_residual_blend=0.5`

### 고정 조건

- `seed=42`
- `batch_size=64`
- `train_epoch=1`
- `lr=2e-4`
- `drop_out=0.1`
- `seq_len=30`
- `fusion_type=concat`
- `history_meta_scale=1.5`

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
- `BPRLoss 47.5063`
- `B_HR 0.0167`
- `B_NDCG 0.0083`
- `T_HR 0.0030`
- `T_MAP 0.0013`

## 해석

- 이전 pure history projection은 `B_NDCG`가 약간 내려갔지만, residual blend로 바꾸자 `B_NDCG`가 baseline을 넘었다.
- `B_HR`는 조금 내려갔지만 `T_HR`, `T_MAP`가 함께 좋아졌고, primary metric도 개선됐다.
- 따라서 문제는 history projection 아이디어 자체보다는 shared path를 완전히 대체했던 구현 방식에 더 가까웠다.

## 현재 판단

- gate verdict: PASS on fast-scout
- full champion 유지: `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`
- next candidate: residual blend full validation

## 다음 후보

1. `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_hprojblend05.json` full run
2. full에서도 유지되면 sequence-side 구조 승격 후보
3. 유지되지 않으면 blend 계수 축을 따로 보지 않고 현 champion으로 복귀
