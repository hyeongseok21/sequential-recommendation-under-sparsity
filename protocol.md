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

## Required Outputs Per Experiment

- gate result JSON
- experiment memory row
- update log
- promoted champion일 경우:
  - dual-best report JSON

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

## Notes Convention

- `notes` 첫 부분에 가능하면 `family=<axis_family>`를 넣는다.
- 예시:
  - `family=metadata-input; single feature screening`
  - `family=attention-capacity; head count fast-scout`
  - `family=evaluation-policy; direct checkpoint compare`

## Fast-Scout Protocol

- 목적: low-cost signal capture
- 조건:
  - 보통 `train_epoch=1`
  - 필요 시 더 큰 `batch_size`
- 규칙:
  - PASS여도 full run 전에는 champion 교체 금지

## Early Stop Heuristic

- epoch 0 기준으로 baseline champion 대비 `B_NDCG`가 뚜렷하게 낮고
- runtime 비용도 증가하면 조기 중단 가능

## Checkpoint Policy

- best epoch 기준 수치를 update log에 남긴다
- final epoch보다 best epoch를 우선한다
- champion 승격 시에는 `benchmark-best`뿐 아니라 `test-best` checkpoint도 같이 기록한다
- `different_epoch=true`면 dual-best report를 필수 artifact로 본다

## Doc Policy

- 모든 update log는 한국어로 쓴다
- “idea fail”과 “current implementation fail”은 구분해서 적는다
