# Slice Evaluation

## 목적

- 실제 추천 운영 가정에서 overall metric만으로 결론내리지 않기 위한 slice 평가 기준을 정의한다.
- `sparse-history user`와 `multi-interest user`를 기본 slice로 사용한다.
- slice metric은 `research champion` 승격보다 `serving companion` 해석에 우선적으로 활용한다.

## 기본 Slice

### `sparse-history user`

- 정의 목적:
  - 행동 로그가 적어 sequence-only signal이 약한 사용자를 분리한다.
- 기본 정의:
  - train history 길이가 하위 구간에 속하는 사용자
- 권장 초기 기준:
  - history length `<= 5`
- 해석 포인트:
  - metadata injection이 semantic 보완 역할을 하는지 확인한다.
  - retrieval / ranking 분리 실험에서 retrieval quality 차이가 더 커지는지 본다.

### `multi-interest user`

- 정의 목적:
  - 최근 행동에서 카테고리 전환이 잦은 사용자를 분리한다.
- 기본 정의:
  - 최근 행동 구간에서 category diversity 또는 transition count가 높은 사용자
- 권장 초기 기준:
  - 최근 `10`개 구매 기준
  - `department` unique count `>= 3` 또는
  - adjacent category transition count `>= 4`
- 해석 포인트:
  - `DIF-SR` 같은 multi-interest 친화 구조가 실제로 이 slice에서 강한지 본다.
  - feature weighting이나 metadata source가 전환 포착에 도움되는지 본다.

## Feature Backbone

- slice 정의와 metadata injection은 같은 feature backbone을 우선 공유한다.
- 현재 우선순위:
  1. `product_type`
  2. `department`
  3. `section` 또는 `index_group`
- style 계열 feature:
  - `colour_group_code`
  - `graphical_appearance_no`
- 원칙:
  - multi-interest 전환을 먼저 드러내는 category feature를 본다.
  - style drift feature는 그 다음 phase에서 추가한다.

## Metric Policy

- overall metric:
  - `B_NDCG`
  - `B_HR`
  - `T_MAP`
  - `T_HR`
- slice metric:
  - slice별 `HR`
  - slice별 `NDCG`
  - slice별 `MAP` 또는 `Recall@K`
- 기본 해석 규칙:
  - overall 개선이 있으면 우선 승격 후보
  - overall 차이가 작아도 `sparse-history user` 또는 `multi-interest user`에서 일관된 개선이 있으면 `serving companion` 후보가 될 수 있다.
  - slice 이득이 한쪽에만 있고 다른 slice에서 크게 악화되면 `slice-instability`로 기록한다.

## Report Contract

- serving phase의 update log에는 가능하면 다음을 포함한다.
  - overall metric 요약
  - `sparse-history user` metric 요약
  - `multi-interest user` metric 요약
  - 어떤 slice에서 개선/악화가 있었는지 한 줄 해석
- 추천 문장 예시:
  - `metadata는 sparse-history user에서 상대적으로 더 효과적이었다.`
  - `DIF-SR 계열은 multi-interest user에서 category transition을 더 잘 따라갔다.`

## Phase Usage

- `P4`:
  - serving companion 판단 시 기본 slice 리포트를 붙인다.
- `P5`:
  - final summary에는 overall + slice 관찰을 함께 적는다.
- 이후 retrieval phase:
  - candidate retrieval quality도 같은 slice 기준으로 본다.

## Immediate Next Slice Work

1. `sparse-history user` 정의 함수를 스크립트로 고정한다.
2. `multi-interest user` 정의를 `department` 기반 transition count로 먼저 고정한다.
3. current champion에 대해 overall + slice dual-best report를 만든다.
