# DIF-SR Eval-Gap Leaderboard

## 목적

- `DIF-SR` run 전체에서 benchmark-best epoch와 test-best epoch가 얼마나 자주 갈리는지 집계한다.

## 산출물

- JSON:
  - [`data/metrics/eval_gap_leaderboard_difsr.json`](../../data/metrics/eval_gap_leaderboard_difsr.json)
- CSV:
  - [`data/metrics/eval_gap_leaderboard_difsr.csv`](../../data/metrics/eval_gap_leaderboard_difsr.csv)

## 핵심 결과

- 집계 대상: `23` runs
- top benchmark run:
  - `hm_m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15`
  - `benchmark-best epoch 1`
  - `B_NDCG 0.0137`
  - `test-best epoch 2`
  - `T_MAP 0.0020`
  - `epoch_gap = 1`

## 상위권 관찰

1. current champion
   - benchmark-best와 test-best가 다름
2. `concat lr2e4` baseline
   - benchmark-best epoch `1`
   - test-best epoch `0`
3. `hms20`
   - benchmark-best와 test-best가 같음
   - 하지만 절대 성능은 champion보다 낮음
4. `hprojblend05`
   - benchmark-best와 test-best가 다름

## 해석

- `DIF-SR` 상위권 run 다수가 epoch gap을 갖는다.
- 즉 current champion의 `evaluation-gap`은 우연이 아니라 이 실험군에서 반복되는 패턴이다.
- 따라서 이후 실험 보고는 single best epoch만 적는 방식보다:
  - benchmark-best
  - test-best
  를 함께 적는 방식이 더 적절하다.

## 현재 판단

- current research champion은 그대로 benchmark 기준 champion 유지
- 운영/분석 관점에서는 dual-best reporting을 기본으로 간다
