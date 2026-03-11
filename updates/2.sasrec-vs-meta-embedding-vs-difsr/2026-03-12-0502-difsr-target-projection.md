# DIF-SR target projection 예비 비교

## 가설

- history encoder와 target item scorer의 표현 공간을 분리하면 ranking quality가 좋아질 수 있다.
- 이를 위해 target item 쪽에 전용 projection layer를 추가한다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4.json`
- 구조: current champion `concat DIF-SR + lr=2e-4`

### Treatment

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_targetproj.json`
- 변경 축: `use_target_projection=true`

### 고정 조건

- `seed=42`
- `batch_size=16`
- `train_epoch=3`
- `drop_out=0.1`
- `seq_len=30`
- `fusion_type=concat`
- 동일 dataset / 동일 benchmark / 동일 test

## 중간 결과

### Baseline

- best epoch: `1`
- `B_HR 0.0303`
- `B_NDCG 0.0135`
- `T_HR 0.0036`
- `T_MAP 0.0012`

### Treatment

- start benchmark: `B_HR 0.0059`, `B_NDCG 0.0025`
- epoch 0:
  - `BPRLoss 11.2484`
  - `B_HR 0.0190`
  - `B_NDCG 0.0096`
  - `T_HR 0.0030`
  - `T_MAP 0.0006`

## 해석

- target projection을 추가하면 구조적으로는 더 유연해지지만, 현재 로컬 조건에서는 benchmark가 baseline champion보다 낮다.
- `T_HR`는 크게 나쁘지 않지만 `B_HR`, `B_NDCG`, `T_MAP`가 모두 밀린다.
- 즉 현재는 history-target 공간 분리보다, 기존 공유 표현이 더 안정적으로 작동하고 있는 것으로 보인다.
- gain signal이 약해 epoch 0 이후는 조기 중단했다.

## 현재 판단

- gate verdict: FAIL
- champion 유지: `concat DIF-SR + lr=2e-4`

## 다음 후보

1. target projection 같은 추가 분리보다, existing item representation 내부의 metadata fusion 방식을 더 세밀하게 다루는 쪽이 유망
2. 또는 best epoch checkpoint를 기준으로 inference/serving 관점 검증으로 넘어가기
3. 이후 구조 실험은 candidate scoring보다 sequence-side metadata interaction 쪽으로 이동
