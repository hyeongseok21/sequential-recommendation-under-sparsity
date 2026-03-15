# Experiment Plan

## Closure Plan

- 목표:
  - 일주일 안에 sequential recommendation 포트폴리오 프로젝트를 완결한다.
- 핵심 질문:
  1. item metadata가 sequential recommendation 성능을 개선하는가
  2. 어떤 user segment가 metadata의 이득을 가장 크게 받는가
- 데이터/태스크:
  - H&M purchase sequence
  - next-item prediction
  - 동일 chronological split 유지
- closure 원칙:
  - completion을 optimization보다 우선한다
  - comparative evidence를 먼저 확보한다
  - 결과 표, slice 표, 그래프, README summary까지 닫는다
- freeze rules:
  - 새 모델 금지
  - 새 metadata feature 금지
  - 새 fusion 전략 금지
  - 새 데이터셋 금지
  - exploratory tuning 재개 금지

## Experiment Queue

### MUST

1. final comparison model set 4개 고정
   - `SASRec`
   - `SASRec + metadata`
   - `DIF-SR`
   - `DIF-SR + metadata`
2. 같은 split / 같은 seed / 같은 `top_k=20`으로 final run artifact 정리
3. overall metric 표 생성
   - `Recall@20`
   - `NDCG@20`
   - `MRR@20`
4. user slice metric 표 생성
   - `sparse-history user`
   - `multi-interest user`
   - 각 slice별 `Recall@20`, `NDCG@20`
5. 그래프 2개 생성
   - model comparison bar chart
   - slice comparison chart
6. final write-up 작성
   - concise findings
   - interpretation
   - limitations
   - README-ready summary

### NICE-TO-HAVE

1. `MRR@20` slice report
2. metadata representation 또는 attention visualization 1개
3. appendix 성격의 dual-best companion summary
4. optional metadata ablation chart

### OUT-OF-SCOPE

1. retrieval / two-tower
2. 새 metadata source 탐색
3. 새 architecture mutation
4. 새 hyperparameter sweep
5. serving policy 추가 확장

## Standardized Model Set

- metadata feature set은 현재 고정값을 유지한다.
- final comparison에는 아래 4개만 포함한다.

1. `SASRec`
2. `SASRec + metadata`
3. `DIF-SR`
4. `DIF-SR + metadata`

- canonical config manifest:
  - [`CLOSURE_MODELS.md`](CLOSURE_MODELS.md)

- 현재 strongest metadata candidate:
  - [`../hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`](../hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json)

## Metric Convention

- reporting top-k는 `20`으로 통일한다.
- 기존 코드/로그의 `HR`는 closure 문서에서 `Recall@20`으로 표기한다.
- `NDCG`는 `NDCG@20`으로 표기한다.
- `MRR@20`은 closure report에서 secondary metric으로 추가 계산한다.

## Slice Definition

### `sparse-history user`

- 초기 기준:
  - train history length `<= 5`

### `multi-interest user`

- 초기 기준:
  - 최근 `10`개 구매 기준
  - `department` unique count `>= 3` 또는 adjacent transition count `>= 4`

## Execution Order

1. closure용 canonical config 4개 확정
2. final overall metric 재실행
3. slice metric 산출
4. result table 작성
5. graph 작성
6. findings / limitations / README summary 작성

## Results Summary

- current strongest metadata run:
  - [`../hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`](../hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json)
- benchmark-best epoch `2`:
  - `Recall@20` 대응값: `0.0309`
  - `NDCG@20`: `0.0139`
- test-best companion epoch `1`:
  - `T_HR` 대응값: `0.0059`
  - `MAP` 기준 strongest companion
- current interpretation:
  - metadata는 일부 설정에서 유효했다
  - 추가 weighting 탐색은 대부분 full validation에서 유지되지 않았다
  - 이제는 exploration보다 comparative closure가 우선이다

## Remaining Risks

1. `SASRec + metadata`가 현재 same-format closure artifact로 아직 정리되지 않았을 수 있다.
2. `MRR@20`은 현재 코드 기본 산출물이 아니라 closure report 단계에서 추가 계산이 필요하다.
3. slice threshold는 once fixed 후 closure 주간에는 바꾸지 않는다.
4. benchmark-best와 test-best가 달라 본문/appendix를 분리해 설명해야 한다.

## Packaging Outputs

1. overall results table
2. sparse-history results table
3. multi-interest results table
4. model comparison bar chart
5. slice comparison chart
6. concise findings
7. limitations
8. README-ready summary
