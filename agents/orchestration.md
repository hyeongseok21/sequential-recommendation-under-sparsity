# Multi-Agent Orchestration

## 추천 구성

### 최소 구성

- `Operator`
- `Analyst`
- `Governor`

구현 변경이 적고 실험 운영이 중심일 때 사용한다.

### 확장 구성

- `Operator`
- `Analyst`
- `Research`
- `Governor`

모델 구조 실험이 많아질 때 사용한다.

## 현재 저장소에 맞는 운용 방식

- `P0`~`P1`: `Operator + Analyst + Governor`
- `P2`~`P3`: `Operator + Analyst + Research + Governor`
- `P4`: `Operator + Analyst + Governor`
- `P5`: `Analyst + Governor`를 기본으로 하고, slice 산출물이 필요할 때만 `Operator`
- `serving-oriented phase`: `Operator + Analyst + Governor`를 기본으로 하고, 구조 실험이 다시 커질 때만 `Research`를 재투입한다

## 파일 기반 공유 상태

- champion config path
- serving companion checkpoint / summary
- `experiment_memory.csv`
- `gate_result.json`
- `updates/...`
- checkpoint report json
- slice summary/json

## 성공 조건

- 모든 agent가 동일한 baseline/champion을 본다.
- 한 루프에서 hypothesis는 하나만 돈다.
- 커밋은 Governor가 의미 단위 종료를 확인한 뒤 진행한다.
- research champion과 serving companion이 다를 경우, 둘 다 같은 handoff payload 안에 명시한다.
- slice-analysis phase에서는 overall conclusion과 slice conclusion을 함께 넘긴다.

## Prompt Assembly

- 각 agent 프롬프트 초안은 `templates/agent_prompt_template.md`를 기반으로 만든다.
- 시스템 규칙은 `AGENT.md`, 실행 절차는 `RUNBOOK.md`, 역할 규칙은 `agents/*.md`에서 가져온다.
- handoff payload는 `templates/agent_handoff_template.md` 형식으로 전달한다.
