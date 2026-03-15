# Experiment Protocol

## Naming

- config: `config.<topic>.json`
- update log: `updates/<phase>/<timestamp>-<topic>.md`
- checkpoint save name은 config name과 최대한 맞춘다

## Required Inputs Per Experiment

- `phase`
- `axis_family`
- `hypothesis`
- `baseline_config`
- `treatment_config`
- `primary_metric`
- `decision_mode`
  - `research`
  - `serving`
  - `hybrid`
  - `closure`

## Required Outputs Per Experiment

- gate result JSON
- experiment memory row
- update log
- promoted champion일 경우:
  - dual-best report JSON
- serving candidate를 다룰 경우:
  - serving note / companion summary
  - slice metric summary (있으면)
- slice-analysis 실험일 경우:
  - slice definition note
  - slice threshold / rule summary
- closure phase일 경우:
  - overall result table
  - slice result table
  - graph artifact list
  - README summary

## Gate Result Schema

```json
{
  "phase": "P2",
  "primary_metric": "B_NDCG",
  "baseline": 0.0103,
  "treatment": 0.0135,
  "delta": 0.0032,
  "verdict": "PASS",
  "reason": "treatment improved primary metric"
}
```

## Experiment Memory CSV Columns

- `timestamp`
- `phase`
- `axis`
- `hypothesis`
- `baseline_config`
- `treatment_config`
- `primary_metric`
- `baseline_value`
- `treatment_value`
- `delta`
- `verdict`
- `notes`

## Axis Taxonomy

- `system`
- `architecture`
- `optimization`
- `metadata-input`
- `attention-capacity`
- `evaluation-policy`
- `serving-proxy`
- `retrieval`
- `slice-analysis`
- `portfolio-closure`

## Notes Convention

- `notes` 첫 부분에 가능하면 `family=<axis_family>`를 넣는다.
- 가능하면 `mode=<research|serving|hybrid>`도 함께 넣는다.
- 예시:
  - `family=metadata-input; mode=research; single feature screening`
  - `family=attention-capacity; mode=research; head count fast-scout`
  - `family=evaluation-policy; mode=serving; direct checkpoint compare`
  - `family=slice-analysis; mode=serving; sparse-history user report`

## Fast-Scout Protocol

- 목적: low-cost signal capture
- 조건:
  - 보통 `train_epoch=1`
  - 필요 시 더 큰 `batch_size`
- 규칙:
  - PASS여도 full run 전에는 champion 교체 금지
  - closure mode에서는 새로운 fast-scout를 열지 않는다.

## Early Stop Heuristic

- epoch 0 기준으로 baseline champion 대비 `B_NDCG`가 뚜렷하게 낮고
- runtime 비용도 증가하면 조기 중단 가능

## Checkpoint Policy

- best epoch 기준 수치를 update log에 남긴다
- final epoch보다 best epoch를 우선한다
- champion 승격 시에는 `benchmark-best`뿐 아니라 `test-best` checkpoint도 같이 기록한다
- `different_epoch=true`면 dual-best report를 필수 artifact로 본다
- `serving companion`을 유지하는 경우, update log에 benchmark champion과 구분해서 적는다
- 실무형 phase에서는 `benchmark-best only`를 최종 단일 truth로 보지 않는다

## Metric Naming Policy

- top-k reporting은 `20`으로 통일한다.
- 기존 코드/로그의 `HR`는 closure 문서에서 `Recall@20`으로 표기한다.
- `NDCG`는 `NDCG@20`으로 표기한다.
- `MRR@20`은 closure report에서 secondary metric으로 추가 계산한다.

## Slice Evaluation Policy

- 실제 추천 가정의 phase에서는 overall metric만으로 결론내리지 않는다.
- 가능한 경우 다음 slice를 기본 대상으로 본다.
  - `sparse-history user`
  - `multi-interest user`
- slice metric은 overall metric과 함께 해석해야 하며, slice 이득만 있는 후보는 `serving companion` 후보가 될 수 있다.
- 기본 초기 정의는 [`SLICE_EVALUATION.md`](../../SLICE_EVALUATION.md)를 따른다.
- multi-interest 정의는 category transition backbone을 우선한다.
  - `product_type`
  - `department`
  - `section` 또는 `index_group`
- style feature는 slice backbone이 고정된 뒤에 추가한다.

## Closure Policy

- closure week에는 scope를 동결한다.
- 허용:
  - baseline 재실행
  - slice report
  - result table / graph / README summary
- 금지:
  - 새 모델
  - 새 feature
  - 새 fusion
  - 새 데이터셋

## Doc Policy

- 모든 update log는 한국어로 쓴다
- “idea fail”과 “current implementation fail”은 구분해서 적는다
