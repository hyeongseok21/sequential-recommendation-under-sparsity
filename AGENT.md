# Recsys Experiment Agent

## 목적

- 이 저장소에서 추천 실험을 drift 없이 반복 실행하기 위한 에이전트 운영 규칙을 정의한다.
- 모든 실험은 `baseline vs treatment`, `one hypothesis per loop`를 기본으로 한다.
- 단순 benchmark 최고점보다, 실제 추천 시스템에 가까운 의사결정을 우선한다.

## 기본 원칙

1. 한 루프에서는 가설 하나만 검증한다.
2. baseline은 현재 champion 설정으로 고정한다.
3. treatment는 하나의 축만 바꾼다.
4. 연구용 비교 지표는 `B_NDCG`를 1순위로 본다.
5. 운영용 판단에서는 `T_MAP`, `T_HR`, `dual-best gap`도 함께 본다.
6. seed는 항상 `42`로 고정한다.
7. champion은 하나만 두지 않고, 필요하면 `research champion`과 `serving companion`을 함께 둔다.

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
- 허용 변경: inference path, best-epoch selection, candidate scoring path, reporting path
- pass 기준: 품질 유지 또는 개선 + latency/risk 감소

### `P5` Finalization

- 목표: champion 고정, 결과 요약, 재실행 절차 확정

## Metric Priority

### Research Priority

1. `B_NDCG`
2. `B_HR`
3. `T_MAP`
4. `T_HR`

### Serving Proxy Priority

1. `T_MAP`
2. `T_HR`
3. `dual-best gap`
4. checkpoint 안정성 / runtime

## Experiment Axes

### `system`

- 실행 가능성
- seed / 재현성
- 평가 안정화
- checkpoint 정책
- evaluation-gap

### `architecture`

- backbone (`SASRec`, `meta embedding`, `DIF-SR`)
- fusion (`sum`, `concat`, `gate`)
- encoder / scoring / projection
- history-side interaction
- target-side interaction

### `optimization`

- `lr`
- `batch_size`
- `drop_out`
- `weight_decay`
- `clip_grad_ratio`
- `train_epoch`

### `metadata-input`

- single feature
- pair feature
- all feature
- feature weighting
- feature-specific scale

### `attention-capacity`

- `n_heads`
- `n_layers`

### `evaluation-policy`

- `benchmark-best`
- `test-best`
- direct checkpoint evaluation
- dual-best report
- leaderboard / report policy

## Mutation Scope

- `P2`에서는 config 위주로 바꾼다.
- `P3`에서만 모델 코드를 바꾼다.
- 구조 실험도 한 번에 한 mutation만 허용한다.
- `P4`에서는 평가, 선택, 리포트, checkpoint 관련 경로만 바꾼다.
- 모든 실험은 위 taxonomy 중 하나의 `axis family`를 먼저 명시한다.

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
- 구조가 복잡해졌는데 gain이 작으면 champion 유지
- 필요하면 `benchmark-best`는 연구 champion으로, `test-best`는 serving companion으로 분리한다.
- `serving companion`은 benchmark champion을 대체하지 않지만 운영 가정의 기본 후보가 될 수 있다.

## Reporting Policy

- update log는 항상 한국어로 쓴다.
- `idea fail`과 `current implementation fail`은 구분해서 적는다.
- `benchmark-best`와 `test-best`가 다를 수 있으면 둘 다 남긴다.
- champion 승격 이후에는 `dual-best`를 기본 산출물로 만든다.
- 실험 방향 전환 시에는 benchmark 목표와 serving 목표를 분리해서 적는다.

## Multi-Agent Policy

- 멀티 에이전트 운영 시에도 champion, phase, gate 기준은 이 문서를 공통 헌장으로 사용한다.
- 역할 분리는 `execution`, `analysis`, `research`, `governance` 축으로 나눈다.
- 어떤 agent도 baseline 없이 treatment를 단독 제안하거나 실행하지 않는다.
- 최종 승격 판단은 항상 `Governor` 역할이 내린다.
- 상태 공유는 대화가 아니라 파일 기반으로 한다.
  - `data/metrics/experiment_memory.csv`
  - `data/metrics/gate_result.json`
  - `updates/...`
  - checkpoint metadata/json
- 세부 역할과 handoff 규약은 `agents/` 아래 문서를 따른다.
