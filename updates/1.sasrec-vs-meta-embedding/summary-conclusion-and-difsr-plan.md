# 로컬 실험 요약 및 DIF-SR 후속 계획

## 이번 작업에서 정리한 기반

- M1 로컬 실행 환경 정리
  - `.venv` 기반 실행 환경 구성
  - `MPS` 사용 가능 여부 확인
  - 로컬 `mlruns` 기반으로 외부 의존성 없이 실행 가능하게 정리
- 학습 러너 안정화
  - `cuda / mps / cpu` 자동 선택 지원
  - 상대경로 config 처리
  - `mlflow` 미설치 시 no-op 처리
  - benchmark / test 평가의 빈 후보, top-k 예외 상황 방어
- 전처리/런타임 호환성 보강
  - pandas 2.x 호환 수정
  - M1 로컬 smoke / local / hash benchmark용 config 추가
- 재현성 확보
  - `seed` 고정 로직 추가
  - DataLoader / numpy / torch / random 시드 정렬
  - benchmark candidate 순서 비결정성 제거

## 실험 흐름 요약

### 1. non-meta Transformer 기준선 정리

- `embed_size=32`, `epoch=3`, benchmark 설정으로 안정적으로 로컬 검증 가능함을 확인
- `epoch=5`까지 늘려도 실행은 안정적이지만, 시드 고정 전에는 run-to-run 분산이 커서 결론이 흔들렸음
- 시드 고정 후에는 동일 config 재실행 시 metric 재현 가능함을 확인

### 2. meta Transformer 기본 비교

- 같은 seed 기준에서 `meta on`과 `non-meta`를 비교
- 결과적으로 현재 로컬 조건에서는 `non-meta`가 명확히 우세
- 즉, "메타데이터를 쓰는 것 자체"보다 "메타데이터를 어떻게 쓰는가"가 핵심 문제라는 가설이 강화됨

### 3. meta 내부 최적화 탐색

- `lr=2e-4`: FAIL
  - 손실은 더 빨리 줄었지만 `B_NDCG`와 `T_MAP` 악화
- `batch_size=16`: PASS
  - meta benchmark 기준 개선
- `seq_len=30`: PASS
  - benchmark와 test가 함께 소폭 개선
- `n_layers=2`: FAIL
  - 성능 크게 하락
- `n_heads=4`: FAIL
  - `B_HR`는 약간 올랐지만 `B_NDCG` 악화
- `drop_out=0.1`: PASS
  - 현재 메타 설정에서 가장 좋은 benchmark/test 균형
- `weight_decay_opt=true`: FAIL
  - test 일부는 올랐지만 benchmark가 크게 악화

## 현재 결론

### 전체 로컬 champion

- 여전히 `non-meta Transformer`
- 이유
  - 같은 seed 기준으로도 현재 meta 계열이 non-meta를 넘지 못했음

### meta 내부 champion

- [`hm_refactored/configs/config.m1_local_meta_transformer_hash_benchmark_bs16_seq30_do01.json`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.m1_local_meta_transformer_hash_benchmark_bs16_seq30_do01.json)
- 핵심 설정
  - `embed_metadata=true`
  - `batch_size=16`
  - `seq_len=30`
  - `drop_out=0.1`
  - `seed=42`

### 해석

- 지금까지의 결과는 "메타데이터가 쓸모없다"가 아니라,
  "현재 SASRec 기반 메타 결합 방식으로는 충분한 이득을 못 내고 있다"에 가깝다.
- 따라서 다음 단계는 단순 하이퍼파라미터 조정보다
  메타데이터를 attention 내부에서 직접 활용하는 구조적 변경이 더 타당하다.

## 후속 작업 방향: DIF-SR 구현

### 왜 DIF-SR인가

- 현재 meta 구현은 메타 임베딩을 보조 입력으로 사용하는 수준에 가깝다.
- `DIF-SR`는 아이템 메타데이터를 multi-head attention 내부에서 직접 반영하는 구조이므로,
  "메타를 attention 안에서 결합하는 방식 자체가 성능 차이를 만드는가"를 검증할 수 있다.
- 지금까지의 실험 결과상 가장 설득력 있는 다음 축이다.

### 구현 원칙

- 비교축은 하나만 유지
- baseline은 현재 meta champion으로 고정
- treatment는 동일 데이터 / 동일 seed / 동일 eval / 동일 sampler에서 `model_type`만 `DIFSR`로 교체
- first step은 full paper reproduction이 아니라, 현재 코드베이스에 들어가는 최소 통합 버전 구현

### 예상 작업 항목

1. `models/`에 `DIFSR` 모델 추가
2. `train.py`에 `model_type == "DIFSR"` 분기 추가
3. 현재 metadata input 포맷이 DIF-SR attention 입력과 맞는지 점검
4. 필요 시 meta dataset 출력 형식 최소 수정
5. 현재 BPR 학습 루프와 바로 연결 가능한지 확인
6. `config.m1_local_meta_difsr_*.json` 추가

### 1차 실험 설계

- Baseline
  - current meta champion
  - `config.m1_local_meta_transformer_hash_benchmark_bs16_seq30_do01.json`
- Treatment
  - 같은 설정에서 `model_type=DIFSR`
- 고정 항목
  - `embed_metadata=true`
  - `seed=42`
  - `batch_size=16`
  - `seq_len=30`
  - benchmark / test pipeline 동일
  - same dataset / same target week / same sampler
- 1차 판단 지표
  - `B_NDCG`
  - `B_HR`
  - `T_MAP`
  - epoch별 학습 시간
  - M1/MPS 메모리 안정성

### 기대 결과와 판단 기준

- `DIF-SR`가 meta champion보다 `B_NDCG`와 `T_MAP`를 동시에 개선하면 구조 변경 가치가 충분함
- `B_HR`만 오르고 `NDCG/MAP`가 약하면 champion 교체는 보류
- 구현 복잡도 대비 개선이 미미하면 현재 코드베이스에서는 meta Transformer champion 유지

## 최종 정리

- 로컬 실험 기반은 충분히 정리되었고, 재현성도 확보했다.
- 현재 최선의 해석은 "메타를 쓰는 방식"이 병목이라는 것이다.
- 따라서 다음 단계는 하이퍼파라미터 추가 탐색보다 `DIF-SR` 통합 및 baseline-vs-treatment 비교 실험이 맞다.
