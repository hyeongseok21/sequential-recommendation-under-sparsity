# 역할별 automation 초안 추가

## 변경 배경
- handoff/prompt 템플릿과 bundle 생성 스크립트는 준비됐지만, 실제로 바로 등록 가능한 역할별 automation 초안은 아직 없었다.
- 운영 관점에서는 Analyst, Operator, Governor 세 역할의 prompt를 바로 꺼내 쓸 수 있는 문서가 필요했다.

## 추가한 파일
- `automations/README.md`
- `automations/analyst.md`
- `automations/operator.md`
- `automations/governor.md`

## 반영 내용
- 각 automation 초안에 read-first 문서, current champion, primary metric, required output을 고정했다.
- schedule은 포함하지 않고 prompt 본문만 저장소에서 관리하도록 분리했다.
- docs lint가 `automations/` 문서 존재도 함께 검증하도록 확장됐다.
- 임시 sample bundle 산출물은 저장소 관리 대상에서 제거했다.

## 기대 효과
- 실제 recurring automation 등록 시 바로 prompt 본문으로 재사용할 수 있다.
- 현재 champion과 metric 기준이 일관되게 유지된다.
- multi-agent 운영을 사람이 수동으로도, automation으로도 같은 규약 아래 실행할 수 있다.
