# closure preflight checklist

## 가설

- final run 전에 closure preflight를 문서와 스크립트로 고정하면, 일주일 내 마감 과정에서 비교 불공정이나 산출물 누락을 줄일 수 있다.

## 비교 조건

### Baseline

- canonical model set은 있었지만 실행 직전 점검 절차가 명시적으로 고정되지 않은 상태

### Treatment

- `CLOSURE_PREFLIGHT.md` 추가
- `scripts/check_closure_readiness.py` 추가
- runbook에 preflight 선행 규칙 추가

### 고정 조건

- closure model set 4개 유지
- scope freeze 유지

## 결과

### Baseline

- 실행 직전 검토 항목이 암묵적이었고, split parity나 save 경로 충돌을 사람이 수동으로 확인해야 했음

### Treatment

- config 존재 여부
- split parity
- `seed=42`
- `top_k=20`
- dataset 경로 존재
- save_name 충돌
- `DIF-SR` baseline의 `metadata_features=[]`
  를 자동 점검하는 preflight 스크립트 추가

## 해석

- closure 단계에서는 새 실험보다 실행 실수 방지가 더 중요하다.
- preflight를 선행하면 같은 질문에 대한 4모델 비교를 더 안전하게 닫을 수 있다.

## 현재 판단

- gate verdict: `PASS`
- champion: unchanged

## 다음 후보

1. `MRR@20` 계산 스크립트 추가
2. overall / slice 결과 테이블 스캐폴드 생성
3. canonical model set 실행
