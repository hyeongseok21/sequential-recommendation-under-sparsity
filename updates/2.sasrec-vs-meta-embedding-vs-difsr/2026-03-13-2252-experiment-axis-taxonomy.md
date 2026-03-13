# 실험 축 taxonomy 정리

## 배경
- 실험이 누적되면서 `구조`, `하이퍼파라미터`, `metadata 입력`, `head 수`, `평가 정책`이 서로 섞여 보이기 시작했다.
- 앞으로는 어떤 실험이 어떤 family에 속하는지 공통 문서에서 먼저 정의하는 편이 관리에 유리하다.

## 정리한 축
- `system`
- `architecture`
- `optimization`
- `metadata-input`
- `attention-capacity`
- `evaluation-policy`

## 문서 반영
- `AGENT.md`
  - 공통 실험 축 taxonomy 추가
- `SELF_EVOLUTION_LOOP.md`
  - `axis family` 우선순위와 family 단위 냉각 규칙 추가
- `protocol.md`
  - required input에 `axis_family` 추가
  - `notes`에 `family=...`를 붙이는 규칙 추가

## 기대 효과
- update log와 memory를 읽을 때 실험 목적이 더 빨리 드러난다.
- 같은 family에서 반복 실패하는 패턴을 더 쉽게 찾을 수 있다.
- 다음 가설을 고를 때도 “어느 family를 지금 파고 있는가”를 더 명확히 유지할 수 있다.
