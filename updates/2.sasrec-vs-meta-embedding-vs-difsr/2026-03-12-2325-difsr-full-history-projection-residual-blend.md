# DIF-SR full history projection residual blend 비교

## 가설

- history metadata 전용 projection을 shared projection의 residual blend로 넣으면 full training budget에서도 sequence-side interaction이 더 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`
- 조건: `history_meta_scale=1.5`

### Treatment

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_hprojblend05.json`
- 변경 축: `use_history_meta_projection=true`, `history_meta_residual_blend=0.5`

### 고정 조건

- `seed=42`
- `batch_size=16`
- `train_epoch=3`
- `lr=2e-4`
- `drop_out=0.1`
- `seq_len=30`
- `fusion_type=concat`
- `history_meta_scale=1.5`

## 결과

### Baseline

- best epoch: `1`
- `B_HR 0.0291`
- `B_NDCG 0.0137`
- `T_HR 0.0054`
- `T_MAP 0.0016`

### Treatment

- best epoch: `2`
- `BPRLoss 5.5833`
- `B_HR 0.0268`
- `B_NDCG 0.0127`
- `T_HR 0.0059`
- `T_MAP 0.0021`

## 해석

- fast-scout에서는 residual blend가 좋아 보였지만 full validation에서는 primary metric인 `B_NDCG`가 baseline보다 낮았다.
- 대신 `T_HR`, `T_MAP`는 좋아졌기 때문에, 이 구조는 full benchmark ranking보다 test-side signal을 더 끌어올리는 방향으로 작동한 것으로 보인다.
- 따라서 현재 champion을 교체할 정도의 구조 개선으로 보기는 어렵다.

## 현재 판단

- gate verdict: FAIL on full validation
- full champion 유지: `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`
- failure class: `sequence-context`

## 다음 후보

1. sequence-side 구조 실험은 scale/projection 단일축보다 evaluation gap을 먼저 점검한다.
2. next hypothesis는 `benchmark`와 `test`의 괴리를 줄이는 checkpoint selection 또는 best-epoch policy 쪽이 더 적절하다.
3. 구조 축을 다시 보려면 history metadata가 attention score에 들어가는 경로를 더 명시적으로 바꾸는 실험이 필요하다.
