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

- `P0`~`P2`: `Operator + Analyst + Governor`
- `P3`: `Operator + Analyst + Research + Governor`
- `P4`~`P5`: `Operator + Analyst + Governor`

## 파일 기반 공유 상태

- champion config path
- `experiment_memory.csv`
- `gate_result.json`
- `updates/...`
- checkpoint report json

## 성공 조건

- 모든 agent가 동일한 baseline/champion을 본다.
- 한 루프에서 hypothesis는 하나만 돈다.
- 커밋은 Governor가 의미 단위 종료를 확인한 뒤 진행한다.

## Prompt Assembly

- 각 agent 프롬프트 초안은 `templates/agent_prompt_template.md`를 기반으로 만든다.
- 시스템 규칙은 `AGENT.md`, 실행 절차는 `RUNBOOK.md`, 역할 규칙은 `agents/*.md`에서 가져온다.
- handoff payload는 `templates/agent_handoff_template.md` 형식으로 전달한다.
