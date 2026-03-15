# Serving-Oriented Plan Reset

## 가설

- 지금 단계에서는 benchmark champion을 더 미세하게 튜닝하는 것보다, 실제 추천 운영 가정에 맞게 실험 목적과 의사결정 기준을 재설정하는 것이 더 효율적이다.

## 비교 조건

### Baseline

- benchmark champion 중심의 계획
- `B_NDCG` 최적화 우선
- `metadata-input`, `attention-capacity` 추가 탐색 중심

### Treatment

- 실제 추천 운영 가정 중심의 계획
- `research champion`과 `serving companion` 분리
- `dual-best`, `T_MAP`, `T_HR`, 운영 복잡도까지 포함한 의사결정

### 고정 조건

- current benchmark champion 유지
- `DIFSR + concat + history_meta_scale 1.5 + product_type_scale 1.5`

## 결과

### Baseline

- fast-scout에서는 여러 후보가 좋아 보였음
- full validation에서는 대부분 current champion을 넘지 못함
- benchmark-best와 test-best가 반복적으로 갈림

### Treatment

- 실험 계획을 `serving-oriented`로 재작성
- `evaluation-policy`를 현재 active phase로 승격
- `dual-best`를 기본 운영 artifact로 사용
- `research champion`과 `serving companion` 분리 원칙을 명시

## 해석

- 지금 병목은 단순 scale/head 튜닝이 아니라 benchmark와 test의 괴리를 어떻게 해석하고 운영에 반영할지에 있다.
- 따라서 실험 목표를 “benchmark score 하나 더 올리기”에서 “실제 추천 의사결정에 맞는 모델/체크포인트 체계 만들기”로 전환하는 것이 맞다.

## 현재 판단

- gate verdict: `PASS`
- champion:
  - research champion: `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`
  - serving companion: current dual-best report의 `test-best` epoch

## 다음 후보

1. `product_type15` champion에 대해 serving note / dual-best summary를 별도 문서로 고정
