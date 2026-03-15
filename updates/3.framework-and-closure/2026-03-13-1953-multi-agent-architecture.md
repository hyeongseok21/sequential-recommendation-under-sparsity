# 멀티 에이전트 운영 구조 추가

## 배경
- 기존에는 단일 agent 문서와 runbook으로 실험을 운영했지만, 역할이 늘어나면서 실행/분석/구현/판단 책임을 분리할 필요가 생겼다.
- 특히 recursive experiment loop를 장기적으로 운영하려면 handoff와 상태 공유 규약이 문서로 고정돼 있어야 한다.

## 추가된 구성
- `agents/operator.md`
- `agents/analyst.md`
- `agents/research.md`
- `agents/governor.md`
- `agents/handoff.md`
- `agents/orchestration.md`

## 역할 정의
- `Operator`: 승인된 baseline/treatment 실행, 로그/메트릭/업데이트 기록
- `Analyst`: 결과 해석, bottleneck 분류, 다음 hypothesis 제안
- `Research`: 구조 변경 설계와 최소 구현
- `Governor`: gate 판단, champion 승격, phase 전환, 커밋 시점 결정

## 운영 원칙
- 공통 헌장은 `AGENT.md`를 따른다.
- handoff는 파일 경로와 metric 수치를 포함해야 한다.
- 상태 공유는 `experiment_memory.csv`, `gate_result.json`, `updates/...`, checkpoint metadata로 한다.
- 최소 구성은 `Operator + Analyst + Governor`, 구조 실험이 많아지면 `Research`를 추가한다.

## 보강 사항
- `AGENT.md`에 multi-agent policy를 추가했다.
- `scripts/lint_experiment_docs.py`가 `agents/` 문서 존재 여부도 검증하도록 확장됐다.
- experiment skill이 `agents/orchestration.md`, `agents/handoff.md`를 함께 읽도록 갱신됐다.

## 기대 효과
- 실험 수행과 실험 해석을 분리해 편향을 줄일 수 있다.
- 구조 실험이 늘어나도 책임 경계가 유지된다.
- 나중에 automation이나 외부 orchestrator를 붙일 때 바로 역할 기반으로 확장할 수 있다.
