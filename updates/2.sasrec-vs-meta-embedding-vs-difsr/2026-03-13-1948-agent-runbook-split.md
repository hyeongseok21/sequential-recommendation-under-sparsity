# AGENT.md / RUNBOOK.md 역할 분리

## 변경 배경
- 기존 `RUNBOOK.md`는 실행 절차보다 에이전트의 행동 규칙, 게이트 기준, champion 정책을 더 많이 담고 있었다.
- 문서 이름과 실제 역할이 어긋나 있어, 에이전트 운영 규칙과 실행 절차를 분리하는 편이 더 명확하다고 판단했다.

## 적용 내용
- `AGENT.md`를 신설하고 에이전트 운영 규칙을 이 문서로 이동했다.
- `RUNBOOK.md`는 실제 실험 실행 절차, 체크포인트 평가, 리포트 생성 명령 중심으로 재정리했다.
- `scripts/lint_experiment_docs.py`가 `AGENT.md` 존재 여부도 검증하도록 수정했다.
- repo-local skill인 `skills/recsys-self-evolution-runbook/SKILL.md`가 `AGENT.md`를 우선 읽도록 갱신됐다.
- 전역 skill도 같은 방식으로 로컬 환경에 동기화했다.

## 현재 문서 역할
- `AGENT.md`: phase, gate, mutation scope, champion policy, stop rule 등 에이전트 행동 계약
- `RUNBOOK.md`: canonical command, 실행 순서, checkpoint evaluation, report 생성 등 운영 절차
- `SELF_EVOLUTION_LOOP.md`: exploit/explore loop와 recursive 실험 흐름
- `protocol.md`: 산출물, 로그, 메트릭 기록 규약

## 기대 효과
- 에이전트 규칙과 운영 절차가 섞이지 않아 문서 탐색성이 좋아진다.
- skill과 lint 기준이 같은 문서 구조를 바라보게 되어 재사용성이 높아진다.
- 이후 자동화나 agent 확장 시 `AGENT.md`를 중심 문서로 삼기 쉬워진다.
