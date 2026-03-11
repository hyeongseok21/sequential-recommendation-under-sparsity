# DIF-SR gate fusion 예비 비교

## 가설

- metadata fusion을 `gate` 방식으로 바꾸면, 속성별 attention score를 학습적으로 조절해 `concat`보다 더 나은 ranking 성능을 낼 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat.json`
- 구조: metadata-aware item representation + `fusion_type=concat`

### Treatment

- `config.m1_local_meta_difsr_bs16_seq30_do01_gate.json`
- 구조: metadata-aware item representation + `fusion_type=gate`

### 고정 조건

- `seed=42`
- `batch_size=16`
- `seq_len=30`
- `drop_out=0.1`
- `embed_metadata=true`
- 동일 dataset / 동일 benchmark / 동일 test

## 구현 메모

- 기존 코드에는 `gate` 경로가 비어 있었기 때문에 `VanillaAttention`을 추가해 source-wise learned fusion이 가능하도록 최소 구현했다.
- 첫 시도에서는 source 축 정렬이 잘못되어 shape mismatch가 발생했고, 이를 수정한 뒤 재실행했다.

## 중간 결과

### Baseline (`concat`)

- best epoch: `2`
- best benchmark: `B_HR 0.0220`, `B_NDCG 0.0103`
- best test: `T_HR 0.0042`, `T_MAP 0.0013`

### Treatment (`gate`)

- start benchmark: `B_HR 0.0071`, `B_NDCG 0.0030`
- epoch 0:
  - `BPRLoss 11.8137`
  - `B_HR 0.0137`
  - `B_NDCG 0.0059`
  - `T_HR 0.0018`
  - `T_MAP 0.0002`

## 해석

- `gate`는 epoch 0 시점에서 이미 `concat` baseline보다 benchmark/test가 모두 낮다.
- 동시에 runtime 비용이 크게 증가했다. 동일 조건에서 `concat`은 수 분 내 3 epoch를 마쳤지만, `gate`는 1 epoch 진행만으로도 현저히 느렸다.
- 다만 이번 결과는 `gate` 아이디어 자체의 실패라기보다, 현재 최소 구현의 실패로 해석하는 편이 맞다.
- 현재 구현은 source-wise learned fusion을 빠르게 붙인 수준이라, 논문식 또는 더 정교한 `gate` 설계와는 차이가 있다.
- 따라서 이번 판단은 "gate concept FAIL"이 아니라 "minimal gate implementation FAIL"이다.

## 현재 판단

- gate verdict: FAIL
- champion 유지: `DIF-SR + metadata-aware item representation + fusion_type=concat`
- 이번 run은 epoch 0 이후 runtime 비용이 과도해 조기 중단했다.
- 후속 해석: `gate`는 필요하면 더 충실한 구현으로 나중에 재평가할 수 있다.

## 다음 후보

1. `gate`를 유지하려면 현재 dense attention 대신 더 가벼운 source-wise MLP 또는 scalar gating으로 축소
2. `concat champion` 기준으로 `lr` 또는 `drop_out`을 추가 미세조정
3. 이후 실험은 quality 우선이라면 `gate`보다 `concat` 기반의 scoring/fusion 세부 개선이 더 적합
