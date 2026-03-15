# Multi-Agent Handoff Protocol

## 기본 순서

1. `Analyst`가 다음 hypothesis 1개를 제안한다.
2. `Governor`가 채택 여부와 phase 적합성을 확인한다.
3. `Research`가 필요한 최소 mutation을 구현한다.
4. `Operator`가 baseline/treatment를 실행한다.
5. `Analyst`가 결과를 해석한다.
6. `Governor`가 champion 유지/승격과 다음 step을 확정한다.

## 공통 입력 파일

- `../framework/AGENT.md`
- `../framework/RUNBOOK.md`
- `../framework/SELF_EVOLUTION_LOOP.md`
- `../framework/protocol.md`
- `data/metrics/experiment_memory.csv`
- `data/metrics/gate_result.json`

## handoff 단위

- hypothesis
- baseline config path
- treatment config path
- phase
- primary metric
- current champion

## handoff 규칙

- handoff는 항상 파일 경로와 metric 수치를 포함한다.
- “좋아 보인다” 같은 표현만 넘기지 않는다.
- 실패도 다음 agent에 넘긴다. 실패 로그는 다음 가설 생성에 필요하다.
- handoff 본문 형식은 `../templates/agent_handoff_template.md`를 따른다.

## 권장 handoff 예시

- `Analyst -> Governor`: hypothesis, baseline, phase 적합성, expected win condition
- `Governor -> Research`: 승인된 mutation axis, 금지 변경, 승격 조건
- `Research -> Operator`: 코드 변경 요약, treatment config path, runtime 주의사항
- `Operator -> Analyst`: raw metrics, checkpoint/report path, update log path
- `Analyst -> Governor`: verdict, bottleneck classification, next hypothesis
