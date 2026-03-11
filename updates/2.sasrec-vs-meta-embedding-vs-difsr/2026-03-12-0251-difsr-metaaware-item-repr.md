# DIF-SR metadata-aware item representation

## 가설

- 현재 최소 통합 `DIF-SR`는 attention 안에서만 metadata를 쓰고 item/candidate representation은 약하다.
- history token과 candidate scoring에도 metadata-aware item representation을 직접 반영하면 성능이 개선될 수 있다.

## 변경 내용

- `hm_refactored/models/Transformer.py`
  - `CustomDIFSR`에 `meta_project` 추가
  - `item embedding + projected metadata embedding`을 합친 item representation 사용
  - 적용 범위
    - history token
    - positive / negative item
    - recommend candidate item

## 비교 조건

### Baseline

- 현재 최소 통합 `DIF-SR`
- config: `config.m1_local_meta_difsr_bs16_seq30_do01.json`

### Treatment

- metadata-aware item representation이 추가된 `DIF-SR`
- 동일 config 재실행

## 결과

### Baseline

- best epoch: `1`
- best benchmark: `B_HR 0.0083`, `B_NDCG 0.0038`
- best test: `T_HR 0.0006`, `T_MAP 0.0001`

### Treatment

- best epoch: `2`
- best benchmark: `B_HR 0.0220`, `B_NDCG 0.0101`
- best test: `T_HR 0.0030`, `T_MAP 0.0007`

## 해석

- item/candidate 쪽 metadata representation을 보강하자 `DIF-SR`가 크게 개선됐다.
- 이는 이전 실패 원인이 "메타를 attention 안에 넣는 아이디어 자체"보다
  "item representation과 scoring 쪽 구현이 약했던 점"에 더 가까웠다는 것을 시사한다.

## 현재 판단

- gate verdict: PASS
- `DIF-SR` 내부 champion 교체
- 수치상 현재 meta-embedding SASRec champion도 넘어섰음

## 다음 후보

1. 현재 개선된 `DIF-SR`를 meta champion과 정식 champion decision으로 확정
2. `fusion_type`을 `sum` 외 방식으로 확장
3. non-meta SASRec 전체 champion과도 직접 비교
