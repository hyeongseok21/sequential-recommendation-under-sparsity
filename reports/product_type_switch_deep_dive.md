# Product-Type Switch Deep Dive

Main slice: `history<=8 & 1-2 product_type switches`

- median price std within main slice: `0.013507`
- median mean interval within main slice: `65.67`

| Slice | Model | Users | Recall@20 | NDCG@20 | MRR@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| history<=8 & 1 switch | Corrected SASRec | 104 | 0.0000 | 0.0000 | 0.0000 |
| history<=8 & 1 switch | DIF-SR | 104 | 0.0096 | 0.0025 | 0.0007 |
| history<=8 & 1 switch | DIF-SR + Metadata | 104 | 0.0192 | 0.0118 | 0.0101 |
| history<=8 & 2 switches | Corrected SASRec | 100 | 0.0200 | 0.0100 | 0.0067 |
| history<=8 & 2 switches | DIF-SR | 100 | 0.0100 | 0.0063 | 0.0050 |
| history<=8 & 2 switches | DIF-SR + Metadata | 100 | 0.0100 | 0.0024 | 0.0006 |
| history<=8 & 1-2 switches | Corrected SASRec | 204 | 0.0098 | 0.0049 | 0.0033 |
| history<=8 & 1-2 switches | DIF-SR | 204 | 0.0098 | 0.0043 | 0.0028 |
| history<=8 & 1-2 switches | DIF-SR + Metadata | 204 | 0.0147 | 0.0072 | 0.0054 |
| history<=8 & 1-2 switches & non-repeat | Corrected SASRec | 204 | 0.0098 | 0.0049 | 0.0033 |
| history<=8 & 1-2 switches & non-repeat | DIF-SR | 204 | 0.0098 | 0.0043 | 0.0028 |
| history<=8 & 1-2 switches & non-repeat | DIF-SR + Metadata | 204 | 0.0147 | 0.0072 | 0.0054 |
| history<=8 & 1-2 switches & low price variance | Corrected SASRec | 103 | 0.0000 | 0.0000 | 0.0000 |
| history<=8 & 1-2 switches & low price variance | DIF-SR | 103 | 0.0000 | 0.0000 | 0.0000 |
| history<=8 & 1-2 switches & low price variance | DIF-SR + Metadata | 103 | 0.0097 | 0.0097 | 0.0097 |
| history<=8 & 1-2 switches & high price variance | Corrected SASRec | 101 | 0.0198 | 0.0099 | 0.0066 |
| history<=8 & 1-2 switches & high price variance | DIF-SR | 101 | 0.0198 | 0.0088 | 0.0057 |
| history<=8 & 1-2 switches & high price variance | DIF-SR + Metadata | 101 | 0.0198 | 0.0046 | 0.0011 |
| history<=8 & 1-2 switches & dense sequence | Corrected SASRec | 102 | 0.0098 | 0.0049 | 0.0033 |
| history<=8 & 1-2 switches & dense sequence | DIF-SR | 102 | 0.0098 | 0.0062 | 0.0049 |
| history<=8 & 1-2 switches & dense sequence | DIF-SR + Metadata | 102 | 0.0196 | 0.0120 | 0.0103 |
| history<=8 & 1-2 switches & sparse sequence | Corrected SASRec | 102 | 0.0098 | 0.0049 | 0.0033 |
| history<=8 & 1-2 switches & sparse sequence | DIF-SR | 102 | 0.0098 | 0.0025 | 0.0007 |
| history<=8 & 1-2 switches & sparse sequence | DIF-SR + Metadata | 102 | 0.0098 | 0.0024 | 0.0006 |

## Candidate Overlap within Main Slice

| Pair | Users | Mean Overlap | Median Overlap |
| --- | ---: | ---: | ---: |
| Corrected SASRec vs DIF-SR + Metadata | 204 | 0.0103 | 0.0100 |
| DIF-SR vs DIF-SR + Metadata | 204 | 0.0475 | 0.0400 |
