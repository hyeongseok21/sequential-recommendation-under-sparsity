# M1 로컬 Transformer 확장 검증

## 변경 내용

- `hm_refactored/configs/config.m1_local_transformer_hash_benchmark.json`
  - `embed_size`를 `32`로 상향
  - `train_epoch`를 `3`으로 상향
  - 전처리 재사용을 위해 `reset=false`로 조정
- `hm_refactored/configs/config.m1_local_meta_transformer_hash_benchmark.json`
  - `embed_metadata=true`인 로컬 benchmark 설정 추가
  - `embed_size=32`, `train_epoch=3`, `batch_size=32`
  - 전처리 재사용을 위해 `reset=false` 유지

## 실행 설정

### Non-meta benchmark

```bash
source .venv/bin/activate
python hm_refactored/train.py --config_path hm_refactored/configs/config.m1_local_transformer_hash_benchmark.json
```

### Meta benchmark

```bash
source .venv/bin/activate
python hm_refactored/train.py --config_path hm_refactored/configs/config.m1_local_meta_transformer_hash_benchmark.json
```

## 검증 결과

### Non-meta (`hm_m1_local_transformer_hash_benchmark_e3_d32`)

- 데이터 규모
  - 평가 유저: `164`
  - 평가 고유 아이템: `150`
- 학습 결과
  - start benchmark: `B_HR 0.0549`, `B_NDCG 0.0267`
  - epoch 0: `BPRLoss 113.4176`, `B_HR 0.0488`, `B_NDCG 0.0230`
  - epoch 1: `BPRLoss 100.3046`, `B_HR 0.0488`, `B_NDCG 0.0242`
  - epoch 2: `BPRLoss 91.5137`, `B_HR 0.0671`, `B_NDCG 0.0300`
- 결론
  - M1/MPS에서 3 epoch 완주
  - benchmark 기준으로는 epoch 2가 가장 양호

### Meta (`hm_m1_local_meta_transformer_hash_benchmark_e3_d32`)

- 데이터 규모
  - 유저: `22258`
  - 아이템: `29785`
  - 평가 유저: `1681`
  - benchmark 고유 아이템: `1332`
- 학습 결과
  - start benchmark: `B_HR 0.0125`, `B_NDCG 0.0053`
  - start test: `T_HR 0.0006`, `T_MAP 0.0001`
  - epoch 0: `BPRLoss 36.7532`, `B_HR 0.0107`, `B_NDCG 0.0049`
  - epoch 1: `BPRLoss 25.5797`, `B_HR 0.0101`, `B_NDCG 0.0047`
  - epoch 2: `BPRLoss 22.7188`, `B_HR 0.0101`, `B_NDCG 0.0043`, `T_HR 0.0012`, `T_MAP 0.0007`
- 결론
  - 메타데이터-on 설정도 M1/MPS에서 3 epoch 완주
  - 현재 샘플과 설정에서는 non-meta 대비 benchmark 지표가 낮음

## 해석

- `embed_size`를 `32`로 올리고 epoch를 `3`으로 늘린 non-meta 설정은 benchmark 기준으로 개선 여지가 확인됐다.
- 메타데이터-on 버전은 실행 안정성은 확보됐지만, 현 샘플링과 하이퍼파라미터에서는 성능 이득이 보이지 않았다.
- 따라서 현재 로컬 champion은 `config.m1_local_transformer_hash_benchmark.json`이다.

## 다음 후보

1. non-meta 기준 `train_epoch=5`로 추가 검증
2. meta 버전의 `lr`, `batch_size`, `seq_len`을 별도 탐색
3. user-hash 샘플 대신 평가 유저 밀도가 더 높은 샘플을 다시 구성
