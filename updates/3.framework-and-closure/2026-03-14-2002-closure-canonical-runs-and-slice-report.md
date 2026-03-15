# Closure Canonical Runs And Slice Report

## 요약

- portfolio closure 기준 4개 canonical model run을 모두 완료했다.
- 기존 `SASRec` closure baseline은 non-meta prep split을 읽고 있어서 평가 유저 수가 `164`로 줄어 있었고, 나머지 3개 모델은 `1681`명이었다.
- 최종 비교의 공정성을 위해 `SASRec` baseline도 metadata-preprocessed `userhash32` split을 읽도록 정렬한 뒤 다시 실행했다.
- 최종 overall / slice report는 모두 동일한 평가 유저 `1681`명 기준으로 생성했다.

## Canonical Model Set

1. `SASRec`
2. `SASRec + metadata`
3. `DIF-SR`
4. `DIF-SR + metadata`

## Overall Result

benchmark-best checkpoint 기준 test metric:

| model | checkpoint | Recall@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| SASRec | 1 | 0.0006 | 0.0002 | 0.0001 |
| SASRec + metadata | 2 | 0.0012 | 0.0005 | 0.0002 |
| DIF-SR | 2 | 0.0077 | 0.0024 | 0.0010 |
| DIF-SR + metadata | 2 | 0.0083 | 0.0032 | 0.0018 |

## Slice Result

### sparse-history

| model | users | Recall@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| SASRec | 483 | 0.0000 | 0.0000 | 0.0000 |
| SASRec + metadata | 483 | 0.0041 | 0.0016 | 0.0009 |
| DIF-SR | 483 | 0.0062 | 0.0024 | 0.0014 |
| DIF-SR + metadata | 483 | 0.0124 | 0.0052 | 0.0033 |

### multi-interest

| model | users | Recall@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| SASRec | 1136 | 0.0000 | 0.0000 | 0.0000 |
| SASRec + metadata | 1136 | 0.0000 | 0.0000 | 0.0000 |
| DIF-SR | 1136 | 0.0079 | 0.0026 | 0.0012 |
| DIF-SR + metadata | 1136 | 0.0070 | 0.0025 | 0.0013 |

## 해석

- Q1. item metadata는 sequential recommendation 성능을 개선하는가?
  - 그렇다. 두 backbone 모두 metadata를 붙였을 때 overall test metric이 개선됐다.
  - 개선 폭은 `DIF-SR` 쪽에서 더 컸다.
- Q2. 어떤 user segment가 metadata의 이득을 가장 많이 받는가?
  - `sparse-history` user에서 가장 크다.
  - `DIF-SR + metadata`는 `sparse-history` slice에서 `DIF-SR` 대비 `Recall@20`, `NDCG@20`, `MRR@20`이 모두 상승했다.
  - 반면 `multi-interest` slice에서는 `DIF-SR`와 `DIF-SR + metadata` 차이가 크지 않았고, `NDCG@20` 기준으로는 순수 `DIF-SR`가 약간 높았다.

## 한계

- local `userhash32` split 기준 결과라 full-scale production setting을 직접 대체하지는 않는다.
- main table은 benchmark-best checkpoint를 기준으로 고정했기 때문에, test-best checkpoint와는 차이가 있을 수 있다.
- next-item single-positive setting이므로 `MRR@20`은 reciprocal-rank 성격과 `AP@20` 성격이 거의 같은 값으로 해석된다.

## 산출물

- overall table: [`/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/overall_results.json`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/overall_results.json)
- slice table: [`/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/slice_results.json`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/slice_results.json)
- markdown summary: [`/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/RESULTS.md`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/RESULTS.md)
- README summary: [`/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/README_summary.md`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/README_summary.md)
- overall chart: [`/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/plots/model_comparison.png`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/plots/model_comparison.png)
- slice chart: [`/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/plots/slice_comparison.png`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/plots/slice_comparison.png)
