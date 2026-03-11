# DIF-SR 최소 통합 및 smoke 검증

## 작업 목적

- 현재 meta champion은 메타데이터를 임베딩에 더하는 방식의 Transformer다.
- 다음 단계로, 아이템 메타데이터를 attention 내부에 직접 반영하는 `DIF-SR` 구조를 현재 코드베이스에 최소 통합한다.
- first goal은 논문 완전 재현이 아니라, 기존 학습/평가 파이프라인에 들어가는 실행 가능한 treatment 모델 확보이다.

## 구현 내용

### 1. `models/layers.py` 보정

- `DIFMultiHeadAttention`, `DIFTransformerLayer` 관련 코드가 이미 존재했지만 실제 사용 가능한 상태는 아니었다.
- `copy` import를 추가했다.
- `DIFTransformerLayer`의 생성자 인자 연결을 수정했다.
- `fusion_type='sum'` 기준으로 사용할 수 있게 기본 경로를 정리했다.

### 2. `models/Transformer.py`에 `CustomDIFSR` 추가

- 현재 `CustomMetaSASRec`와 동일한 `forward / recommend` 계약을 따르는 `CustomDIFSR` 모델을 추가했다.
- 학습 루프와 충돌하지 않도록 meta 모델의 입력 시그니처를 그대로 유지했다.
- positive / negative item scoring은 기존과 동일하게 item embedding 기반으로 유지했다.
- sequence encoder는 history item의 metadata를 attention 내부에서 사용하도록 구성했다.
- 사용한 item metadata는 현재 코드베이스 기준으로 다음 3개다.
  - `product_type_no`
  - `department_no`
  - `garment_group_no`

### 3. `train.py` 분기 추가

- `embed_metadata=true` 상태에서 `model_type == "DIFSR"`를 선택할 수 있게 분기 추가
- `train_df`에서 `item_id -> item metadata index` lookup table을 구성해 `CustomDIFSR`에 전달
- 기존 meta dataset / train loop / benchmark / test 코드는 그대로 재사용

### 4. config 추가

- smoke 검증용
  - `hm_refactored/configs/config.m1_local_meta_difsr_smoke.json`
- baseline 비교용 full config
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01.json`

## smoke 검증

### 실행 명령

```bash
source .venv/bin/activate
python hm_refactored/train.py --config_path hm_refactored/configs/config.m1_local_meta_difsr_smoke.json
```

### 결과

- M1 / MPS에서 초기화 통과
- benchmark / test start evaluation 통과
- 1 epoch 학습 완주
- checkpoint 저장까지 완료

### smoke 로그 기준 핵심 수치

- start benchmark: `B_HR 0.0042`, `B_NDCG 0.0018`
- epoch 0:
  - `BPRLoss 14.4792`
  - `B_HR 0.0059`
  - `B_NDCG 0.0028`
  - `T_HR 0.0006`
  - `T_MAP 0.0001`

## 현재 해석

- `DIF-SR` 최소 통합 자체는 가능하고, 현재 코드베이스에서 실제 학습/평가가 돈다.
- 즉 다음 단계는 "구현 가능 여부"가 아니라 "현재 meta champion 대비 실제로 이득이 있는가"를 측정하는 실험으로 넘어가면 된다.
- 현재 구현은 최소 통합 버전이므로, 필요하면 이후 단계에서 attribute fusion / scoring / auxiliary objective를 더 정교하게 맞출 수 있다.

## 다음 실험 설계

### Baseline

- current meta champion
- `config.m1_local_meta_transformer_hash_benchmark_bs16_seq30_do01.json`

### Treatment

- `DIF-SR`
- `config.m1_local_meta_difsr_bs16_seq30_do01.json`

### 고정 조건

- `seed=42`
- `batch_size=16`
- `seq_len=30`
- `drop_out=0.1`
- 동일 dataset / 동일 benchmark / 동일 test

### 1차 판단 지표

- `B_NDCG`
- `B_HR`
- `T_MAP`
- epoch당 실행 시간

## 결론

- `DIF-SR` 모델화는 가능했고, smoke 수준에서는 정상 동작을 확인했다.
- 다음 단계는 meta champion과의 정식 baseline-vs-treatment 비교 실험이다.
