# DIF-SR fast-scout target metadata scale 0.5 비교

## 가설

- current fast-scout champion에서 target-side metadata 비중을 낮추면 history-side signal과 충돌이 줄어 ranking quality가 더 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json`
- 조건: `history_meta_scale=1.5`

### Treatment

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15_tms05.json`
- 변경 축: `target_meta_scale=0.5`

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
- `BPRLoss 44.5393`
- `B_HR 0.0155`
- `B_NDCG 0.0064`
- `T_HR 0.0018`
- `T_MAP 0.0004`

## 해석

- 손실은 더 낮아졌지만 runbook 기준 1순위 지표인 `B_NDCG`가 baseline보다 내려갔다.
- `T_HR`, `T_MAP`는 약간 좋아졌지만 benchmark ranking quality가 악화됐기 때문에 승격 후보가 될 수 없다.
- 즉 current champion에서는 target metadata를 줄이는 방향이 history-target alignment를 개선하지 못한다.

## 현재 판단

- gate verdict: FAIL on fast-scout
- full champion 유지: `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`
- failure class: `scoring`

## 다음 후보

1. scoring 축은 한 번 냉각하고 sequence-side 구조 실험으로 돌아간다.
2. 다음은 scale 조정보다 history metadata가 attention으로 들어가는 경로를 더 정교하게 바꾸는 쪽이 적절하다.
3. 후보 1순위는 `history metadata projection`을 단순 치환이 아니라 residual blend 형태로 바꾸는 것이다.
