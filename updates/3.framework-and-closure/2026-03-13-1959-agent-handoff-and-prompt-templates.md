# 멀티 에이전트 handoff / prompt 템플릿 추가

## 변경 배경
- 역할 문서만으로는 실제 멀티 에이전트 운용 시 handoff payload와 프롬프트 구성이 사람마다 달라질 수 있다.
- agent 간 상태 전달 형식을 고정하면 automation이나 외부 orchestrator에 연결하기 쉬워진다.

## 추가한 파일
- `templates/agent_handoff_template.md`
- `templates/agent_prompt_template.md`

## 반영한 규약
- `agents/handoff.md`가 handoff 템플릿을 따르도록 명시했다.
- `agents/orchestration.md`에 prompt assembly 원칙을 추가했다.
- docs lint가 새 템플릿 두 개도 검증하도록 확장됐다.
- experiment skill이 새 템플릿까지 함께 읽도록 갱신됐다.

## 기대 효과
- agent 간 전달 형식이 일정해진다.
- hypothesis, baseline, metric, artifact 경로가 빠지지 않는다.
- 나중에 자동화 프롬프트를 생성할 때도 재사용할 수 있다.
