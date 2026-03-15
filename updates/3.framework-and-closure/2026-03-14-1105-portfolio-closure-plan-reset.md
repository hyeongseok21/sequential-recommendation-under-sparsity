# portfolio closure plan reset

## 가설

- 남은 1주에서는 추가 탐색보다 baseline/metadata 비교를 닫고, overall + slice 결과물 패키지를 완성하는 편이 프로젝트 가치가 더 크다.

## 비교 조건

### Baseline

- serving pivot 이후에도 문서가 여전히 exploration backlog 중심으로 읽히는 상태

### Treatment

- closure mode를 system 문서에 반영
- `EXPERIMENT_PLAN.md`를 마감용 queue와 결과물 기준으로 재작성

### 고정 조건

- 모델 family 고정
- metadata feature set 고정
- H&M dataset 고정

## 결과

### Baseline

- 다음 실험 아이디어는 있었지만, 1주 closure 기준의 MUST/NICE/OUT-OF-SCOPE 구분이 약함

### Treatment

- `Closure Plan`
- `Experiment Queue`
- `Results Summary`
- `Remaining Risks`
- `Packaging Outputs`
  형식으로 plan을 재작성
- `P6`를 portfolio closure phase로 명시

## 해석

- 지금부터는 좋은 아이디어보다 제출 가능한 결과물이 더 중요하다.
- baseline 4개와 slice 2개를 닫는 것이 포트폴리오 가치에 직접 연결된다.
- retrieval, new feature, new fusion은 전부 이후 작업으로 미룬다.

## 현재 판단

- gate verdict: `PASS`
- champion: unchanged

## 다음 후보

1. closure용 canonical config 4개 확정
2. `MRR@20` 산출 경로 추가
3. overall / slice 결과 테이블 생성
