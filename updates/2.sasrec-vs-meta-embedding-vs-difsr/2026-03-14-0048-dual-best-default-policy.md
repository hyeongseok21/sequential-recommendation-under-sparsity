# dual best default policy

## 배경

- 새 benchmark champion인 `product_type_scale = 1.5` 설정도 `benchmark-best`와 `test-best` checkpoint가 다르게 나왔다.
- 즉 dual-best는 예외적인 분석이 아니라, 현재 이 프로젝트의 기본 운영 현상이다.

## 결정

- 앞으로 champion 승격 직후에는 `direct checkpoint evaluation`과 `dual-best report`를 기본 artifact로 생성한다.
- update log에는 가능하면 아래 두 값을 같이 적는다.
  - `benchmark-best`
  - `test-best`

## 반영 문서

- `RUNBOOK.md`
- `protocol.md`
- `EXPERIMENT_PLAN.md`

## 운영 의미

- research champion은 `benchmark-best` 기준으로 유지한다.
- serving 또는 test-oriented 해석이 필요하면 `test-best` checkpoint를 companion artifact로 같이 본다.
