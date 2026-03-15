# closure canonical model set

## 가설

- portfolio closure에서는 각 backbone 안에서 metadata on/off만 바뀌는 canonical config 4개를 고정하는 편이 결과 해석에 가장 유리하다.

## 비교 조건

### Baseline

- strongest run은 있었지만 closure용 canonical model set이 명시적으로 고정되지 않은 상태

### Treatment

- `SASRec`
- `SASRec + metadata`
- `DIF-SR`
- `DIF-SR + metadata`
  4개를 closure config로 고정

### 고정 조건

- same split
- seed `42`
- top-k `20`
- 추가 탐색 금지

## 결과

### Baseline

- 모델별 strongest run은 있었지만, closure 질문에 맞는 apples-to-apples config set은 없었음

### Treatment

- closure config 4개를 추가
- `CLOSURE_MODELS.md`에서 canonical path와 비교 원칙을 고정
- `DIF-SR` baseline은 `metadata_features=[]`로 정의해 metadata injection을 끈 비교 설정으로 명시

## 해석

- 이제부터는 새 tuning이 아니라 이 4개 모델을 재실행하고 결과 표를 만드는 것이 우선이다.
- 특히 `DIF-SR` baseline을 명시적으로 고정해두면서, metadata 효과를 backbone 내부 비교로 설명할 수 있게 됐다.

## 현재 판단

- gate verdict: `PASS`
- champion: unchanged

## 다음 후보

1. `MRR@20` 계산 경로 추가
2. current canonical model set 4개 final run 실행
3. overall / slice 결과 테이블 생성
