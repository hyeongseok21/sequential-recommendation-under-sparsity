# DIF-SR concat fusion 비교

## 가설

- metadata fusion 방식을 `sum`에서 `concat`으로 바꾸면, 속성 간 상호작용을 더 잘 보존해 `DIF-SR`의 benchmark/test 성능이 좋아질 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_difsr_bs16_seq30_do01.json`
- 구조: metadata-aware item representation + `fusion_type=sum`

### Treatment

- `config.m1_local_meta_difsr_bs16_seq30_do01_concat.json`
- 구조: metadata-aware item representation + `fusion_type=concat`

### 고정 조건

- `seed=42`
- `batch_size=16`
- `seq_len=30`
- `drop_out=0.1`
- `embed_metadata=true`
- 동일 dataset / 동일 benchmark / 동일 test

## 결과

### Baseline

- best epoch: `2`
- best benchmark: `B_HR 0.0220`, `B_NDCG 0.0101`
- best test: `T_HR 0.0030`, `T_MAP 0.0007`

### Treatment

- best epoch: `2`
- best benchmark: `B_HR 0.0220`, `B_NDCG 0.0103`
- best test: `T_HR 0.0042`, `T_MAP 0.0013`

## 해석

- `B_HR`는 동일했지만 `B_NDCG`가 소폭 상승했고, test 지표도 모두 개선됐다.
- 차이는 크지 않지만, 현재 로컬 조건에서는 `concat`이 `sum`보다 안정적으로 더 낫다.
- 초기 epoch부터 benchmark/test가 함께 올라간 점을 보면, 단순한 노이즈보다는 metadata 결합 방식 차이가 실제로 반영된 것으로 해석할 수 있다.

## 현재 판단

- gate verdict: PASS
- champion 승격: `DIF-SR + metadata-aware item representation + fusion_type=concat`

## 다음 후보

1. 현재 `concat DIF-SR`를 meta-embedding SASRec champion과 다시 정식 비교
2. `fusion_type=gate`를 구현해 학습형 fusion이 추가 이득을 주는지 확인
3. candidate scoring 쪽 metadata projection을 더 분리해 representation bottleneck을 줄이는 실험 진행
