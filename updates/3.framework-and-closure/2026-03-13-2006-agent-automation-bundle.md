# 멀티 에이전트 automation bundle 추가

## 변경 배경
- handoff/prompt 템플릿까지는 준비됐지만, automation이 바로 사용할 수 있는 self-contained prompt bundle 생성 경로가 없었다.
- recurring task나 외부 orchestrator에 붙이려면 역할별 prompt를 빠르게 뽑아내는 도구가 필요했다.

## 추가한 파일
- `agents/automation.md`
- `templates/automation_prompt_bundle_template.md`
- `scripts/render_agent_bundle.py`

## 반영 내용
- automation 단위를 `Analyst`, `Operator`, `Governor` 중심으로 나눴다.
- prompt bundle에 들어가야 할 공유 문서, 입력 상태, 필수 산출물을 고정했다.
- `scripts/render_agent_bundle.py`로 역할별 prompt 초안을 파일 또는 stdout으로 생성할 수 있게 했다.
- docs lint와 experiment skill도 새 automation 문서를 인지하도록 갱신했다.

## 기대 효과
- 반복 실험용 automation prompt를 수작업 없이 만들 수 있다.
- 역할별 prompt 구성이 문서와 일치하게 유지된다.
- 추후 scheduler나 automation UI에 연결할 때 재사용성이 높다.
