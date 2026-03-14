# slice-analysis framework

## 가설

- serving pivot 이후에는 overall metric과 dual-best만으로는 실제 추천 해석이 부족하고, `sparse-history user`와 `multi-interest user`를 기본 slice로 고정해야 한다.

## 비교 조건

### Baseline

- `evaluation-policy`와 `serving-proxy` 중심 문서 체계
- slice는 언급만 있고 정의/산출물 규칙이 약한 상태

### Treatment

- `SLICE_EVALUATION.md` 신설
- phase / plan / protocol / runbook / skill에 `slice-analysis`를 기본 축으로 승격

### 고정 조건

- serving pivot 방향 유지
- research champion / serving companion 분리 유지

## 결과

### Baseline

- `P4`까지는 정리됐지만 slice 정의가 독립 문서로 고정되지 않음
- sparse-history, multi-interest가 해석 포인트 수준에 머묾

### Treatment

- `sparse-history user`, `multi-interest user` 정의와 해석 규칙을 별도 문서로 고정
- `P5`를 slice / serving finalization으로 분리
- 이후 retrieval phase 전 기본 slice backbone을 먼저 확정하는 흐름으로 정리

## 해석

- 지금 pivot의 다음 핵심은 모델 튜닝이 아니라 `serving-safe interpretation`이다.
- slice가 기본 산출물로 올라와야 `metadata는 sparse user에서 더 효과적` 같은 실무형 해석이 가능하다.
- multi-interest를 보기 위한 backbone은 우선 `product_type`, `department`, `section/index_group` 같은 category transition feature가 되어야 한다.

## 현재 판단

- gate verdict: `PASS`
- champion: unchanged

## 다음 후보

1. current champion 기준 `sparse-history user` / `multi-interest user` slice report 생성
2. slice 정의 스크립트 초안 추가
3. retrieval phase backlog를 slice 기준으로 재작성
