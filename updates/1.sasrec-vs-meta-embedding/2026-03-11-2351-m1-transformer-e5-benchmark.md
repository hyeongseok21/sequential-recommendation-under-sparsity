# M1 로컬 Transformer epoch 5 검증

## 가설

- 현재 로컬 champion인 non-meta Transformer 설정에서 `train_epoch`를 `3`에서 `5`로 늘리면 benchmark 지표가 더 좋아질 수 있다.

## 변경 내용

- 새 설정 파일 추가
  - `hm_refactored/configs/config.m1_local_transformer_hash_benchmark_e5_d32.json`
- 주요 파라미터
  - `embed_size=32`
  - `train_epoch=5`
  - `batch_size=64`
  - `embed_metadata=false`
  - `reset=false`

## 실행 명령

```bash
source .venv/bin/activate
python hm_refactored/train.py --config_path hm_refactored/configs/config.m1_local_transformer_hash_benchmark_e5_d32.json
```

## 결과

- 데이터 규모
  - 평가 유저: `164`
  - 평가 고유 아이템: `150`
- 학습 로그
  - start benchmark: `B_HR 0.0854`, `B_NDCG 0.0357`
  - epoch 0: `BPRLoss 113.4898`, `B_HR 0.0793`, `B_NDCG 0.0351`
  - epoch 1: `BPRLoss 103.6363`, `B_HR 0.0854`, `B_NDCG 0.0372`
  - epoch 2: `BPRLoss 96.1035`, `B_HR 0.0793`, `B_NDCG 0.0410`
  - epoch 3: `BPRLoss 90.8198`, `B_HR 0.0854`, `B_NDCG 0.0413`
  - epoch 4: `BPRLoss 85.8531`, `B_HR 0.0915`, `B_NDCG 0.0420`
- 최종 판단
  - `epoch=5` 설정은 M1/MPS에서 안정적으로 완주
  - 이번 run에서는 epoch가 늘어날수록 benchmark 지표가 점진적으로 개선
  - test 지표(`T_HR`, `T_MAP`)는 여전히 `0.0000`

## 해석

- 학습 손실은 꾸준히 감소했고 benchmark 지표도 epoch 4까지 개선됐다.
- 다만 이 코드베이스는 시드 고정이 없어 이전 `e3` 결과와 직접적인 수치 비교에는 변동성이 있다.
- 현재로서는 `epoch=5`를 non-meta 로컬 실험의 다음 champion 후보로 볼 수 있다.

## 다음 후보

1. `epoch=5` 설정을 한 번 더 반복 실행해 분산 확인
2. `lr=5e-5` 또는 `2e-4`로 학습률 민감도 확인
3. meta 버전에 대해 `batch_size=64` 또는 `seq_len=30` 조합 탐색
