# DIF-SR evaluation-gap 분석

## 목적

- current champion과 최근 실패 run에서 `benchmark-best epoch`와 `test-best epoch`가 실제로 얼마나 갈리는지 확인한다.

## 사용 스크립트

- [`/Users/conan/projects/personalized-fashion-recommendation/scripts/report_eval_gap.py`](/Users/conan/projects/personalized-fashion-recommendation/scripts/report_eval_gap.py)

## 분석 대상

1. current champion
   - `hm_m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15`
2. representative failed variant
   - `hm_m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_hprojblend05`

## 결과

### Current champion

- artifact:
  - [`/Users/conan/projects/personalized-fashion-recommendation/data/metrics/eval_gap_difsr_hms15.json`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/eval_gap_difsr_hms15.json)
- `benchmark-best`:
  - epoch `1`
  - `B_NDCG 0.0137`
  - `T_MAP 0.0016`
- `test-best`:
  - epoch `2`
  - `B_NDCG 0.0120`
  - `T_MAP 0.0020`

해석:

- champion에서도 이미 `benchmark`와 `test`가 다른 epoch를 선호한다.
- benchmark 기준 champion을 유지하는 건 맞지만, test 기준으로는 후속 분석이 필요하다.

### Residual blend failed variant

- artifact:
  - [`/Users/conan/projects/personalized-fashion-recommendation/data/metrics/eval_gap_difsr_hprojblend05.json`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/eval_gap_difsr_hprojblend05.json)
- `benchmark-best`:
  - epoch `2`
  - `B_NDCG 0.0127`
  - `T_MAP 0.0021`
- `test-best`:
  - epoch `1`
  - `B_NDCG 0.0119`
  - `T_MAP 0.0022`

해석:

- 실패 run도 benchmark-best와 test-best가 다르다.
- 즉 최근 구조 실험 일부가 benchmark ranking은 약하게 만들면서 test-side metric을 올리는 패턴을 반복하고 있다.

## 종합 결론

- current bottleneck은 단순 구조 개선보다 `evaluation-gap` 관리에 더 가깝다.
- 다음 실험은 구조 mutation보다:
  1. benchmark-best / test-best dual reporting
  2. checkpoint selection report 고정
  3. 필요 시 test-oriented checkpoint 분석
  순으로 가는 게 맞다.
