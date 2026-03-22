# Product-Type Switch Analysis

최근 5회 구매에서 `product_type` 전환 횟수를 기준으로 `0`, `1-2`, `3+` switch bucket을 구성했습니다.
이 분석의 목적은 `DIF-SR + Metadata`가 과도하게 넓은 mixed-intent slice가 아니라, 보다 국소적인 intent switching 구간에서 추가 이득을 보이는지 확인하는 것입니다.

| Switch Bucket | Model | Users | Recall@20 | NDCG@20 | MRR@20 | Mean History | Mean Unique Product Types |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 switches | Corrected SASRec | 421 | 0.0048 | 0.0024 | 0.0017 | 1.03 | 0.34 |
| 0 switches | DIF-SR | 421 | 0.0048 | 0.0013 | 0.0004 | 1.03 | 0.34 |
| 0 switches | DIF-SR + Metadata | 421 | 0.0119 | 0.0036 | 0.0014 | 1.03 | 0.34 |
| 0 switches | TopPopular | 421 | 0.0451 | 0.0151 | 0.0072 | 1.03 | 0.34 |
| 1-2 switches | Corrected SASRec | 397 | 0.0176 | 0.0065 | 0.0033 | 9.63 | 2.46 |
| 1-2 switches | DIF-SR | 397 | 0.0101 | 0.0035 | 0.0018 | 9.63 | 2.46 |
| 1-2 switches | DIF-SR + Metadata | 397 | 0.0126 | 0.0051 | 0.0033 | 9.63 | 2.46 |
| 1-2 switches | TopPopular | 397 | 0.0302 | 0.0128 | 0.0082 | 9.63 | 2.46 |
| 3+ switches | Corrected SASRec | 863 | 0.0116 | 0.0037 | 0.0015 | 16.80 | 3.95 |
| 3+ switches | DIF-SR | 863 | 0.0081 | 0.0024 | 0.0009 | 16.80 | 3.95 |
| 3+ switches | DIF-SR + Metadata | 863 | 0.0046 | 0.0021 | 0.0014 | 16.80 | 3.95 |
| 3+ switches | TopPopular | 863 | 0.0185 | 0.0066 | 0.0034 | 16.80 | 3.95 |

## Reading Guide

- `0 switches`는 최근 구매 맥락이 매우 안정적인 경우입니다.
- `1-2 switches`는 locally coherent한 intent switching을 나타내는 후보 구간입니다.
- `3+ switches`는 최근 구매 맥락이 더 복잡하거나 noisy한 경우에 가깝습니다.
