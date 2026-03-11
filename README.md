# Personalized Fashion Recommendation

H&M 스타일의 구매 로그를 이용해 다음 아이템을 추천하는 실험용 추천시스템 코드베이스다.  
구현은 순차 추천 중심이며, matrix factorization과 SASRec 계열 Transformer를 비교할 수 있다.

## 프로젝트 목적

- 사용자 구매 이력으로 다음 구매 아이템을 예측
- `MF + BPR`와 `Transformer(SASRec 변형)` 비교
- 상품 메타데이터를 포함한 추천 실험 지원
- MLflow에 학습 및 평가 지표 기록

## 디렉터리 구조

```text
hm_refactored/
  configs/
    config.json              # 기본 실험 설정
  models/
    MF.py                    # MF + BPR 모델
    Transformer.py           # SASRec / MetaSASRec
    layers.py                # attention, encoder layer
    loss.py                  # InfoNCE 등 loss 유틸
    scheduler.py             # lr scheduler 유틸
  util/
    helper.py                # 로거, 시각화 보조 함수
    metric.py                # HR, NDCG, MAP, diversity
  visualizer/
    dosnes.py                # 임베딩 시각화 유틸
  dataset.py                 # train/test/benchmark dataset, sampler
  hm_preprocess.py           # 기본 전처리
  hm_preprocess_meta.py      # 메타데이터 포함 전처리
  train.py                   # 메인 학습 진입점
```

## 학습 흐름

1. `train.py`가 설정 파일을 읽는다.
2. 전처리된 pickle이 있으면 로드하고, 없으면 `hm_preprocess.py` 또는 `hm_preprocess_meta.py`로 생성한다.
3. `dataset.py`에서 사용자 히스토리 시퀀스를 만든다.
4. sampler가 negative item 또는 negative user-history view를 붙인다.
5. 모델이 history representation과 positive/negative item 간 점수를 학습한다.
6. 평가 시 `test`와 `benchmark` 두 방식으로 지표를 계산한다.
7. 결과를 checkpoint와 MLflow에 저장한다.

## 모델 개요

### 1. MF

- 파일: `hm_refactored/models/MF.py`
- 사용자 임베딩과 아이템 임베딩의 내적으로 점수를 계산한다.
- 학습 loss는 BPR 기반이다.

### 2. Transformer

- 파일: `hm_refactored/models/Transformer.py`
- 사용자 최근 구매 시퀀스를 attention encoder로 인코딩한다.
- 마지막 history embedding과 positive/negative item embedding의 점수 차이를 학습한다.
- 옵션:
  - learnable positional embedding
  - predefined gaussian attention mask
  - metadata embedding 확장

### 3. MetaSASRec

- 상품 메타데이터 일부를 item embedding과 결합한다.
- 현재 코드상 실제로 적극 사용되는 메타 특성은 `product_type`, `department`, `garment_group`, `age`다.

## 데이터 처리 방식

전처리 단계에서 다음 작업을 수행한다.

- `target_week` 기준으로 train/test 분리
- user/item categorical index 매핑 생성
- cold user, cold item 제거 옵션 적용
- user별 구매 시퀀스 생성
- 마지막 구매만 남긴 평가용 `unique_last_test_df` 생성
- 전처리 결과를 pickle로 저장

기본 전처리는 대략 다음 컬럼을 기대한다.

```text
user_id, item_id, timestamp, count, occurence
```

메타 전처리는 추가로 다음 컬럼을 기대한다.

```text
user_id, age, postal_code, item_id, price, timestamp, count, occurence,
product_code, product_type_no, graphical_appearance_no, colour_group_code,
perceived_colour_value_id, perceived_colour_master_id, department_no,
index_group_no, section_no, garment_group_no
```

## 평가 방식

### test

- 전체 item space에 대해 top-k 추천
- 지표: `HR`, `MAP`

### benchmark

- ground-truth positive item과 negative 후보들을 같이 두고 ranking
- 지표: `HR`, `NDCG`

## 현재 기본 설정

기본 설정 파일은 `hm_refactored/configs/config.json`이다.

현재 기본 실험은 다음 조합이다.

- `embed_metadata: true`
- `model_type: Transformer`
- `loss_type: BPR`
- `sampler_type: TwoView`
- `seq_len: 30`
- `target_week: 27`

## 실행 방법

먼저 의존성을 설치한다.

```bash
python3 -m pip install -r requirements.txt
```

학습 진입점:

```bash
python hm_refactored/train.py --config_path hm_refactored/configs/config.json
```

Apple Silicon에서 먼저 가볍게 확인하려면 smoke config로 시작하는 편이 낫다.

```bash
python3 hm_refactored/train.py --config_path hm_refactored/configs/config.m1_smoke.json
```

체크포인트에서 재시작:

```bash
python hm_refactored/train.py \
  --config_path hm_refactored/configs/config.json \
  --use_checkpoint 10
```

## 필요한 주요 의존성

코드에서 직접 import되는 주요 라이브러리는 다음과 같다.

- `torch`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `mlflow`
- `scikit-learn`
- `opencv-python`
- `Pillow`
- `tqdm`
- `six`

현재 저장소에는 `requirements.txt`나 `pyproject.toml`이 없다.

## Apple Silicon 메모

- 기본 코드 경로는 이제 `cuda`, `mps`, `cpu`를 자동 선택한다.
- M1/M2에서는 `torch`가 MPS를 지원하는 빌드여야 GPU를 사용할 수 있다.
- 가장 먼저는 `config.m1_smoke.json`으로 1 epoch smoke test를 권장한다.

## 바로 실행이 막히는 지점

이 저장소는 연구용 코드 성격이 강해서, 그대로는 다른 환경에서 바로 실행되기 어렵다.

### 1. 절대경로 하드코딩

`hm_refactored/configs/config.json` 안의 다음 경로들은 기존 개발 환경 기준 절대경로다.

- 원본 데이터 경로
- 전처리 저장 경로
- checkpoint 저장 경로
- attention map 저장 경로

현재 환경에 맞게 모두 수정해야 한다.

### 2. GPU 설정 하드코딩

`train.py` 내부에서 아래 환경변수를 직접 설정한다.

- `CUDA_VISIBLE_DEVICES=0`
- device는 config의 `device_num`을 다시 참조

즉, 실제 GPU 선택 로직이 일관적이지 않다.

### 3. 패키징 구조가 약함

- `sys.path.append(...)`로 import 경로를 우회한다.
- 루트 문서화와 패키지 설정이 없다.

### 4. 설정/코드 불일치 가능성

- scheduler 관련 코드는 일부 주석 처리되어 있다.
- metadata 모델은 정의상 많은 필드를 받지만 실제 학습 루프에서는 일부만 사용한다.

## 파일별 핵심 포인트

### `hm_refactored/train.py`

- 전체 실험 orchestration
- dataset/model/sampler 선택
- 학습, 평가, checkpoint, MLflow 기록

### `hm_refactored/dataset.py`

- train sample 생성
- user history / history mask 구성
- negative sampling
- benchmark/test dataset 생성

### `hm_refactored/hm_preprocess.py`

- 비메타 데이터 전처리
- week 기준 split
- user/item 인덱싱

### `hm_refactored/hm_preprocess_meta.py`

- 메타데이터 포함 전처리
- item/user 외 추가 categorical feature 인덱싱

### `hm_refactored/models/Transformer.py`

- SASRec 변형
- attention mask 및 positional encoding 처리
- recommendation score 계산

## 한 줄 요약

이 프로젝트는 H&M 구매 로그를 이용해 순차 추천 실험을 돌리는 연구용 코드이며, 현재 기준으로는 "실험 재현용 내부 코드"에 가깝고, 실행 전 경로/환경 정리가 먼저 필요하다.
