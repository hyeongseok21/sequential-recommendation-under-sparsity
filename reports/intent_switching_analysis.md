# Intent Switching Analysis

## Key Checks

- `high_entropy`가 실제 intent-switching proxy로도 읽히는지 확인
- `high_entropy`와 `history length`가 얼마나 강하게 결합되어 있는지 확인
- `DIF-SR + Metadata`가 switch-heavy user에서 정말 더 강한지 확인

## Descriptor Sanity Check

- low entropy: users `408`, mean history `3.52`, mean switch rate `0.004`
- mid entropy: users `166`, mean history `7.99`, mean switch rate `0.732`
- high entropy: users `1107`, mean history `19.15`, mean switch rate `0.862`

## Slice Summary

| Slice | Model | Users | Recall@20 | NDCG@20 | MRR@20 | Mean History | Mean Transitions | Mean Switch Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| high-entropy | Corrected SASRec | 1107 | 0.0145 | 0.0047 | 0.0021 | 15.95 | 6.29 | 0.862 |
| high-entropy | DIF-SR | 1107 | 0.0081 | 0.0027 | 0.0012 | 15.95 | 6.29 | 0.862 |
| high-entropy | DIF-SR + Metadata | 1107 | 0.0063 | 0.0020 | 0.0009 | 15.95 | 6.29 | 0.862 |
| high-entropy | TopPopular | 1107 | 0.0226 | 0.0075 | 0.0036 | 15.95 | 6.29 | 0.862 |
| switch-heavy | Corrected SASRec | 941 | 0.0159 | 0.0050 | 0.0021 | 18.17 | 6.97 | 0.861 |
| switch-heavy | DIF-SR | 941 | 0.0074 | 0.0022 | 0.0008 | 18.17 | 6.97 | 0.861 |
| switch-heavy | DIF-SR + Metadata | 941 | 0.0064 | 0.0021 | 0.0010 | 18.17 | 6.97 | 0.861 |
| switch-heavy | TopPopular | 941 | 0.0181 | 0.0066 | 0.0035 | 18.17 | 6.97 | 0.861 |
| high-entropy & short<=5 | Corrected SASRec | 58 | 0.0172 | 0.0086 | 0.0057 | 3.47 | 2.36 | 0.966 |
| high-entropy & short<=5 | DIF-SR | 58 | 0.0172 | 0.0109 | 0.0086 | 3.47 | 2.36 | 0.966 |
| high-entropy & short<=5 | DIF-SR + Metadata | 58 | 0.0172 | 0.0041 | 0.0010 | 3.47 | 2.36 | 0.966 |
| high-entropy & short<=5 | TopPopular | 200 | 0.0450 | 0.0121 | 0.0038 | 4.12 | 2.88 | 0.930 |
| high-entropy & mid6-15 | Corrected SASRec | 508 | 0.0197 | 0.0065 | 0.0029 | 7.71 | 5.31 | 0.859 |
| high-entropy & mid6-15 | DIF-SR | 508 | 0.0098 | 0.0029 | 0.0010 | 7.71 | 5.31 | 0.859 |
| high-entropy & mid6-15 | DIF-SR + Metadata | 508 | 0.0039 | 0.0010 | 0.0003 | 7.71 | 5.31 | 0.859 |
| high-entropy & mid6-15 | TopPopular | 491 | 0.0204 | 0.0077 | 0.0043 | 9.95 | 6.49 | 0.840 |
| high-entropy & long>15 | Corrected SASRec | 541 | 0.0092 | 0.0026 | 0.0009 | 25.02 | 7.63 | 0.854 |
| high-entropy & long>15 | DIF-SR | 541 | 0.0055 | 0.0016 | 0.0006 | 25.02 | 7.63 | 0.854 |
| high-entropy & long>15 | DIF-SR + Metadata | 541 | 0.0074 | 0.0027 | 0.0014 | 25.02 | 7.63 | 0.854 |
| high-entropy & long>15 | TopPopular | 416 | 0.0144 | 0.0052 | 0.0027 | 28.72 | 7.69 | 0.855 |

## Interpretation

- `high_entropy`는 mixed-intent proxy로는 유효하지만, 동시에 long-history user가 많이 섞여 있어 pure switching slice로 보기 어렵습니다.
- `switch-heavy`로 다시 봐도 `Corrected SASRec`가 가장 강하면, 현재 레짐에서는 intent switching보다 sequence signal과 popularity alignment가 더 크게 작동한다고 볼 수 있습니다.
- `DIF-SR + Metadata`가 switch-heavy slice에서 약하면, metadata가 switching signal을 넓게 포착하기보다 candidate space를 더 좁히는 방향으로 작동했을 가능성이 큽니다.
