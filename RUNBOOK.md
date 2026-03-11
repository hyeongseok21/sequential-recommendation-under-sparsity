# Recsys Runbook

## 목적

- 이 저장소에서 추천 실험을 drift 없이 반복 실행하기 위한 표준 절차를 정의한다.
- 모든 실험은 `baseline vs treatment`, `one hypothesis per loop`를 기본으로 한다.

## Canonical Command

```bash
source .venv/bin/activate
python hm_refactored/train.py --config_path <CONFIG_PATH>
```

## 기본 원칙

1. 한 루프에서는 가설 하나만 검증한다.
2. baseline은 현재 champion 설정으로 고정한다.
3. treatment는 하나의 축만 바꾼다.
4. 비교 지표는 `B_NDCG`를 1순위로 본다.
5. 보조 지표는 `B_HR`, `T_MAP`, `T_HR` 순으로 본다.
6. seed는 항상 `42`로 고정한다.

## Phase

### `P0` Bring-up

- 목표: 환경, 데이터, 평가 루프가 끝까지 도는지 확인
- 허용 변경: 경로, 디바이스, 로깅, 데이터 로드, 안전장치
- pass 기준: end-to-end run 성공

### `P1` Baseline

- 목표: 재현 가능한 baseline/champion 설정 확보
- 허용 변경: baseline config 확정, seed 안정화, 평가 안정화
- pass 기준: 동일 config 재실행 시 동일 metric

### `P2` Hyperparameter

- 목표: 현재 champion 주변의 저위험 축 탐색
- 허용 변경: `lr`, `batch_size`, `seq_len`, `drop_out`, `weight_decay`, `clip_grad_ratio`
- pass 기준: `B_NDCG` 개선

### `P3` Architecture

- 목표: 구조적 mutation 검증
- 허용 변경: encoder, fusion, scoring, metadata interaction
- pass 기준: `B_NDCG` 개선이 baseline noise 수준을 넘음

### `P4` Inference

- 목표: checkpoint 선택, serving-safe scoring, latency 검증
- 허용 변경: inference path, best-epoch selection, candidate scoring path
- pass 기준: 품질 유지 또는 개선 + latency/risk 감소

### `P5` Finalization

- 목표: champion 고정, 결과 요약, 재실행 절차 확정

## Metric Priority

1. `B_NDCG`
2. `B_HR`
3. `T_MAP`
4. `T_HR`

## Mutation Scope

- `P2`에서는 config 위주로 바꾼다.
- `P3`에서만 모델 코드를 바꾼다.
- 구조 실험도 한 번에 한 mutation만 허용한다.

## Artifact Rules

- 새 treatment마다 config 파일을 만든다.
- 결과는 `updates/` 아래 phase 폴더에 남긴다.
- gate 결과는 `data/metrics/gate_result.json`
- memory는 `data/metrics/experiment_memory.csv`

## Stop Rules

- epoch 0 또는 초기 평가에서 baseline 대비 명확히 열세이고 runtime 비용이 크면 조기 중단 가능
- 실패 실험도 config와 log는 남긴다

## Champion Policy

- full run에서 `B_NDCG`가 baseline보다 좋아지면 승격 후보
- fast-scout에서만 좋아진 경우 full validation 전에는 승격하지 않는다
