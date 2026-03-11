# DIF-SR concat fast-scout lr 비교

## 가설

- 현재 `concat DIF-SR` champion에서 `lr`를 `1e-4`에서 `2e-4`로 올리면, 짧은 학습 예산에서도 ranking quality가 더 빨리 올라갈 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast.json`
- 구조: `concat DIF-SR`
- 조건: `batch_size=64`, `train_epoch=1`, `lr=1e-4`

### Treatment

- `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4.json`
- 구조: 동일
- 변경 축: `lr=2e-4`

### 고정 조건

- `seed=42`
- `drop_out=0.1`
- `seq_len=30`
- `fusion_type=concat`
- 동일 dataset / 동일 benchmark / 동일 test

## 결과

### Baseline

- best epoch: `0`
- `BPRLoss 59.0298`
- best benchmark: `B_HR 0.0137`, `B_NDCG 0.0053`
- best test: `T_HR 0.0012`, `T_MAP 0.0002`

### Treatment

- best epoch: `0`
- `BPRLoss 48.4609`
- best benchmark: `B_HR 0.0137`, `B_NDCG 0.0063`
- best test: `T_HR 0.0018`, `T_MAP 0.0004`

## 해석

- 같은 1 epoch fast-scout 조건에서 `lr=2e-4`는 `B_HR`는 같았지만 `B_NDCG`와 `T_MAP`를 개선했다.
- `BPRLoss`도 더 낮아져, 현재 `concat DIF-SR`는 짧은 학습 구간에서 약간 더 큰 learning rate가 유리할 가능성이 보인다.
- 다만 이 결과는 fast-scout profile 기준이므로, full 3 epoch champion decision으로 바로 승격하기보다는 정식 run으로 재검증하는 편이 맞다.

## 현재 판단

- gate verdict: PASS on fast-scout
- full champion 유지: `concat DIF-SR (lr=1e-4)`
- next candidate: `concat DIF-SR (lr=2e-4)`를 full run으로 재검증

## 다음 후보

1. `config.m1_local_meta_difsr_bs16_seq30_do01_concat.json`의 `lr=2e-4` full run 추가
2. 결과가 유지되면 그때 full champion 교체
3. 유지되지 않으면 `lr=1e-4`를 champion으로 두고 `weight_decay` 또는 `clip_grad_ratio` 축으로 이동
