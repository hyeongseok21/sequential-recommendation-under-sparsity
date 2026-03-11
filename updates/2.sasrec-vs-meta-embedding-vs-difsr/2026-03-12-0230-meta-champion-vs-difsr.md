# Meta champion vs DIF-SR

## 가설

- 메타데이터를 attention 내부에서 직접 반영하는 `DIF-SR`가 현재 meta champion보다 더 나은 benchmark/test 성능을 낼 수 있다.

## 비교 조건

### Baseline

- `config.m1_local_meta_transformer_hash_benchmark_bs16_seq30_do01.json`
- 구조: meta-embedding SASRec

### Treatment

- `config.m1_local_meta_difsr_bs16_seq30_do01.json`
- 구조: DIF-SR

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
- best benchmark: `B_HR 0.0113`, `B_NDCG 0.0054`
- best test: `T_HR 0.0012`, `T_MAP 0.0003`

### Treatment

- best epoch: `1`
- best benchmark: `B_HR 0.0083`, `B_NDCG 0.0038`
- best test: `T_HR 0.0006`, `T_MAP 0.0001`

## 해석

- 현재 최소 통합 버전의 `DIF-SR`는 meta champion보다 benchmark와 test 모두에서 낮다.
- 즉 "메타를 attention 안에 넣는다"는 방향 자체는 가능하지만, 지금 구현 상태에서는 성능 우위가 확인되지 않았다.
- 특히 candidate ranking 품질(`B_NDCG`)에서 차이가 분명하다.

## 현재 판단

- gate verdict: FAIL
- champion 유지: `meta-embedding SASRec`

## 다음 후보

1. DIF-SR의 attribute fusion 방식을 `sum` 외 다른 방식으로 확장
2. candidate item 쪽에도 metadata-aware representation을 더 직접 반영
3. 현재 최소 통합 버전을 논문 구현과 더 가깝게 보강한 뒤 재비교
