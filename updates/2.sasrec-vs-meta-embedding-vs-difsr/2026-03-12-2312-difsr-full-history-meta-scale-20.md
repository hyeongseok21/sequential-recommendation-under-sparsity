# DIF-SR full history metadata scale 2.0 비교

## 가설

- current champion보다 history-side metadata를 조금 더 강하게 쓰면 full training budget에서도 ranking quality가 더 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
- 조건: `history_meta_scale=1.5`

### Treatment

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms20.json`
- 변경 축: `history_meta_scale=2.0`

### 고정 조건

- `seed=42`
- `batch_size=16`
- `train_epoch=3`
- `lr=2e-4`
- `drop_out=0.1`
- `seq_len=30`
- `fusion_type=concat`

## 결과

### Baseline

- best epoch: `1`
- `B_HR 0.0291`
- `B_NDCG 0.0137`
- `T_HR 0.0054`
- `T_MAP 0.0016`

### Treatment

- best epoch: `1`
- `BPRLoss 6.3932`
- `B_HR 0.0262`
- `B_NDCG 0.0132`
- `T_HR 0.0042`
- `T_MAP 0.0016`

## 해석

- fast-scout에서는 `history_meta_scale=2.0`가 소폭 좋아 보였지만 full validation에서는 `B_NDCG`가 baseline보다 내려갔다.
- `epoch 2`에서 `T_MAP`는 `0.0020`까지 올라갔지만, runbook 기준 1순위 지표인 `B_NDCG`는 best epoch 기준 `0.0132`로 champion을 넘지 못했다.
- 즉 history-side metadata를 더 세게 넣는 방향은 `1.5`까지는 유효했지만 `2.0`에서는 과해진 것으로 보인다.

## 현재 판단

- gate verdict: FAIL on full validation
- full champion 유지: `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`
- failure class: `sequence-context`

## 다음 후보

1. `target_meta_scale`가 아니라 history-side interaction 구조를 바꾼다.
2. 단순 scale 증가 대신 history metadata 전용 attention projection을 다시 설계한다.
3. 다음 구조 실험은 `history metadata projection + scale 1.5` 조합처럼 sequence 경로만 더 정교하게 보는 쪽이 맞다.
