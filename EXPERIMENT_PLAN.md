# Experiment Plan

## Current Serving Candidate

- config: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`
- benchmark-best epoch: `2`
- test-best epoch: `1`
- current benchmark-best full score:
  - `B_HR 0.0309`
  - `B_NDCG 0.0139`
  - `T_HR 0.0042`
  - `T_MAP 0.0015`
- current test-best companion score:
  - `B_HR 0.0286`
  - `B_NDCG 0.0136`
  - `T_HR 0.0059`
  - `T_MAP 0.0021`

## Product Goal

- 단순 benchmark 최고점보다, 실제 추천 시스템에 가까운 의사결정을 만드는 것이 우선이다.
- 모델 선택은 `benchmark-best`만으로 끝내지 않고, `test-best`, 안정성, 운영 복잡도까지 함께 본다.
- 새 실험은 “실서비스 후보로 채택할 가치가 있는가”를 기준으로 평가한다.

## Decision Policy

1. 연구 champion:
   - 1순위는 `B_NDCG`
   - 논문/모델 비교, 구조 실험 판단에 사용
2. serving candidate:
   - `dual-best` 기준으로 본다
   - `benchmark-best`와 `test-best`를 동시에 유지
   - `T_MAP`, `T_HR`, checkpoint 안정성을 운영 지표로 본다
3. mutation acceptance:
   - fast-scout PASS만으로는 채택하지 않는다
   - full validation에서 benchmark 또는 serving 관점 이득이 있어야 한다

## Success Metrics

### Research

1. `B_NDCG`
2. `B_HR`

### Serving Proxy

1. `T_MAP`
2. `T_HR`
3. `dual-best gap` 크기
4. runtime / checkpoint 운용 복잡도

## Backlog

### `evaluation-policy`

1. `benchmark-best` / `test-best` 요약을 기본 serving artifact로 고정
2. champion 승격 시 dual-best report, direct checkpoint evaluation, serving note를 항상 생성
3. evaluation-gap이 큰 후보는 연구 champion과 serving candidate를 분리해서 관리

### `serving-proxy`

1. current champion의 `test-best` checkpoint를 운영 후보로 명시
2. `T_MAP`, `T_HR` 개선이 있으나 `B_NDCG`가 근소 열세인 후보를 companion으로 둘지 규칙화
3. candidate export / summary report를 serving 관점으로 정리

### `slice-analysis`

1. current champion에 대해 `sparse-history user`, `multi-interest user` slice report를 기본 산출물로 추가
2. overall에서는 근소 차이지만 특정 slice에서 일관된 이득이 있는 후보를 `serving companion` 후보로 해석
3. retrieval 실험으로 넘어가기 전, slice 정의를 먼저 고정

### `architecture`

1. `dual-best gap`을 줄일 수 있는 구조 실험
2. history-side metadata interaction 개선
3. checkpoint마다 benchmark/test가 크게 갈리지 않는 구조 우선

### `metadata-input`

1. 현재 축은 냉각
2. 새 metadata source를 추가할 때만 재개
3. 단순 scale 추가 탐색은 우선순위 낮음

## Immediate Next Experiment

- family: `slice-analysis`
- hypothesis: current champion에 slice report를 붙이면 serving candidate 판단이 더 명확해진다.
- baseline:
  - overall metric + dual-best report만 있는 상태
- treatment:
  - dual-best + `sparse-history user` + `multi-interest user` slice report 동시 운용
- expected gate:
  - research champion / serving companion 해석에 slice 근거 추가
  - 이후 retrieval, metadata 확장 실험에서 slice를 기본 축으로 사용

## Latest Result

- family: `metadata-input`
- treatment:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type17.json`
- verdict: `FAIL`
- observed:
  - fast-scout:
    - `B_HR 0.0226`
    - `B_NDCG 0.0095`
    - `T_HR 0.0024`
    - `T_MAP 0.0004`
  - full best epoch `1`:
    - `B_HR 0.0274`
    - `B_NDCG 0.0129`
    - `T_HR 0.0054`
    - `T_MAP 0.0017`
- decision:
  - `product_type` 가중치를 더 올리면 fast-scout은 좋아 보이지만 full에서는 current champion을 넘지 못함
  - `metadata-input` scale 탐색은 여기서 일단 닫고 운영 관점 정리로 이동

## Latest Metadata-Input Result

- `department_scale = 1.5`
  - fast-scout verdict: `FAIL`
  - observed:
    - `B_NDCG 0.0066`
    - `B_HR 0.0167`
- `department_scale = 0.5`
  - fast-scout verdict: `PASS`
  - fast-scout observed:
    - `B_NDCG 0.0091`
    - `B_HR 0.0190`
  - full verdict: `FAIL`
  - full observed:
    - `[0 epoch] B_NDCG 0.0082`
    - `[0 epoch] B_HR 0.0178`
    - `[0 epoch] T_HR 0.0036`
    - `[0 epoch] T_MAP 0.0011`
- interpretation:
  - `department` weighting은 benchmark보다 test 쪽 metric에 더 민감하게 작동하는 경향이 있음
  - `product_type_scale = 1.5`는 full validation까지 PASS해서 새 benchmark champion이 됨
  - `garment_group` 추가 weighting은 `1.5`, `1.2` 모두 champion을 넘지 못함
  - `product_type15 + department05`는 fast-scout `PASS`였지만 full best `B_NDCG 0.0108`로 실패
  - `product_type17`도 fast-scout `PASS`였지만 full best `B_NDCG 0.0129`로 실패
  - 현재까지 `metadata-input` 축에서 full champion으로 남은 건 `product_type_scale = 1.5` 단일 weighting 뿐
  - 이 축은 당분간 냉각하고 `evaluation-policy`와 `dual-best` 운영 정리로 이동
