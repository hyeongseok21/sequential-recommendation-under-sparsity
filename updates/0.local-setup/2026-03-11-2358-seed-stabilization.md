# 로컬 실험 시드 고정 및 재현성 검증

## 변경 내용

- `hm_refactored/train.py`
  - `random`, `numpy`, `torch` 시드 고정 로직 추가
  - `train_params.seed` 지원 추가
  - DataLoader `generator`와 `worker_init_fn`에 동일 시드 적용
- `hm_refactored/dataset.py`
  - `BenchmarkTotalDataset`에서 `set(...).difference(...)`를 제거
  - 후보 아이템 순서를 보존하는 방식으로 negative candidate 생성
- 설정 파일
  - `hm_refactored/configs/config.m1_local_transformer_hash_benchmark.json`
  - `hm_refactored/configs/config.m1_local_meta_transformer_hash_benchmark.json`
  - `hm_refactored/configs/config.m1_local_transformer_hash_benchmark_e5_d32.json`
  - 위 3개에 `train_params.seed=42` 추가

## 검증 방법

같은 설정을 두 번 연속 실행했다.

```bash
source .venv/bin/activate
python hm_refactored/train.py --config_path hm_refactored/configs/config.m1_local_transformer_hash_benchmark_e5_d32.json
python hm_refactored/train.py --config_path hm_refactored/configs/config.m1_local_transformer_hash_benchmark_e5_d32.json
```

## 검증 결과

- 두 실행의 metric 라인은 동일했다.
- 차이는 타임스탬프와 tqdm 진행 속도 출력뿐이었다.

### 재현된 핵심 수치

- start benchmark: `B_HR 0.0610`, `B_NDCG 0.0362`
- epoch 0: `BPRLoss 109.5299`, `B_HR 0.0610`, `B_NDCG 0.0358`, `T_HR 0.0061`, `T_MAP 0.0007`
- epoch 1: `BPRLoss 97.3351`, `B_HR 0.0671`, `B_NDCG 0.0374`, `T_HR 0.0061`, `T_MAP 0.0020`
- epoch 2: `BPRLoss 91.4396`, `B_HR 0.0610`, `B_NDCG 0.0346`, `T_HR 0.0061`, `T_MAP 0.0007`
- epoch 3: `BPRLoss 84.8356`, `B_HR 0.0549`, `B_NDCG 0.0325`, `T_HR 0.0061`, `T_MAP 0.0007`
- epoch 4: `BPRLoss 82.6757`, `B_HR 0.0610`, `B_NDCG 0.0335`
- best epoch: `1`, `GLOBAL_HR 0.0671`, `GLOBAL_NDCG 0.0374`

## 결론

- 로컬 M1 실험에서 가장 큰 분산 원인은 시드 미고정과 benchmark 후보 순서 비결정성이었다.
- 현재는 동일 config 재실행 시 metric 재현이 가능하다.
- 이후 `epoch`, `lr`, `metadata` 비교는 이제 반복 평균 없이도 1차 비교가 가능해졌다.
