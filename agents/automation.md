# Multi-Agent Automation Guide

## 목적

- 멀티 에이전트 문서를 recurring task, 외부 scheduler, automation prompt에 바로 연결하기 위한 가이드를 정의한다.

## 권장 자동화 단위

### 1. Analyst Loop

- 입력: current champion, recent memory, recent gate result
- 출력: 다음 hypothesis 1개, baseline/treatment 제안, phase 적합성 판단

### 2. Operator Loop

- 입력: 승인된 baseline/treatment, phase, expected output path
- 출력: update log, gate result, experiment memory row

### 3. Governor Loop

- 입력: Analyst verdict, Operator result
- 출력: champion 유지/승격, 다음 축, 커밋 승인 여부

## 자동화 원칙

- 한 automation은 하나의 역할만 수행한다.
- schedule 정보는 prompt에 넣지 않고 automation 메타데이터로 분리한다.
- prompt는 자기완결적이어야 하며, 최소한 아래 항목을 포함한다.
  - current champion
  - phase
  - baseline config
  - treatment config 또는 hypothesis
  - expected artifact

## 권장 실행 순서

1. `Analyst` automation
2. `Governor` review
3. `Research` 또는 `Operator`
4. `Governor` closeout

## prompt 구성

- 공통 규칙은 `AGENT.md`
- 실행 절차는 `RUNBOOK.md`
- loop 정책은 `SELF_EVOLUTION_LOOP.md`
- 역할 규칙은 `agents/*.md`
- handoff 형식은 `templates/agent_handoff_template.md`
- prompt 골격은 `templates/agent_prompt_template.md`

## 보조 도구

- `scripts/render_agent_bundle.py`로 역할별 prompt bundle 초안을 생성할 수 있다.
