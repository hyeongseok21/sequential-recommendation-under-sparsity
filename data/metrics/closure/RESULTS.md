# Closure Results

## Overall Results

| Model | Checkpoint | Users | Recall@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SASRec | 1 | 1681 | 0.0006 | 0.0002 | 0.0001 |
| SASRec + metadata | 2 | 1681 | 0.0012 | 0.0005 | 0.0002 |
| DIF-SR | 2 | 1681 | 0.0077 | 0.0024 | 0.0010 |
| DIF-SR + metadata | 2 | 1681 | 0.0083 | 0.0032 | 0.0018 |

## Slice Results

| Slice | Model | Users | Recall@20 | NDCG@20 | MRR@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| sparse-history | SASRec | 483 | 0.0000 | 0.0000 | 0.0000 |
| multi-interest | SASRec | 1136 | 0.0000 | 0.0000 | 0.0000 |
| sparse-history | SASRec + metadata | 483 | 0.0041 | 0.0016 | 0.0009 |
| multi-interest | SASRec + metadata | 1136 | 0.0000 | 0.0000 | 0.0000 |
| sparse-history | DIF-SR | 483 | 0.0062 | 0.0024 | 0.0014 |
| multi-interest | DIF-SR | 1136 | 0.0079 | 0.0026 | 0.0012 |
| sparse-history | DIF-SR + metadata | 483 | 0.0124 | 0.0052 | 0.0033 |
| multi-interest | DIF-SR + metadata | 1136 | 0.0070 | 0.0025 | 0.0013 |

## Notes

- Main table uses the benchmark-best checkpoint for each canonical model.
- In this single-positive next-item setting, MRR@20 is numerically aligned with reciprocal-rank style evaluation and matches AP@20 behavior.

## History Length Buckets

| History Bucket | Model | Users | Recall@20 | NDCG@20 | MRR@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| 21+ | SASRec | 395 | 0.0000 | 0.0000 | 0.0000 |
| 6-10 | SASRec | 395 | 0.0025 | 0.0007 | 0.0002 |
| 11-20 | SASRec | 408 | 0.0000 | 0.0000 | 0.0000 |
| 1-5 | SASRec | 483 | 0.0000 | 0.0000 | 0.0000 |
| 21+ | SASRec + metadata | 395 | 0.0000 | 0.0000 | 0.0000 |
| 6-10 | SASRec + metadata | 395 | 0.0000 | 0.0000 | 0.0000 |
| 11-20 | SASRec + metadata | 408 | 0.0000 | 0.0000 | 0.0000 |
| 1-5 | SASRec + metadata | 483 | 0.0041 | 0.0016 | 0.0009 |
| 21+ | DIF-SR | 395 | 0.0051 | 0.0016 | 0.0006 |
| 6-10 | DIF-SR | 395 | 0.0152 | 0.0043 | 0.0015 |
| 11-20 | DIF-SR | 408 | 0.0049 | 0.0013 | 0.0004 |
| 1-5 | DIF-SR | 483 | 0.0062 | 0.0024 | 0.0014 |
| 21+ | DIF-SR + metadata | 395 | 0.0076 | 0.0030 | 0.0017 |
| 6-10 | DIF-SR + metadata | 395 | 0.0076 | 0.0028 | 0.0015 |
| 11-20 | DIF-SR + metadata | 408 | 0.0049 | 0.0014 | 0.0004 |
| 1-5 | DIF-SR + metadata | 483 | 0.0124 | 0.0052 | 0.0033 |

## Category Entropy Buckets

| Entropy Bucket | Model | Users | Recall@20 | NDCG@20 | MRR@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| high | SASRec | 1107 | 0.0000 | 0.0000 | 0.0000 |
| low | SASRec | 408 | 0.0000 | 0.0000 | 0.0000 |
| mid | SASRec | 166 | 0.0060 | 0.0017 | 0.0005 |
| high | SASRec + metadata | 1107 | 0.0000 | 0.0000 | 0.0000 |
| low | SASRec + metadata | 408 | 0.0049 | 0.0019 | 0.0010 |
| mid | SASRec + metadata | 166 | 0.0000 | 0.0000 | 0.0000 |
| high | DIF-SR | 1107 | 0.0081 | 0.0027 | 0.0012 |
| low | DIF-SR | 408 | 0.0025 | 0.0007 | 0.0002 |
| mid | DIF-SR | 166 | 0.0181 | 0.0046 | 0.0013 |
| high | DIF-SR + metadata | 1107 | 0.0063 | 0.0020 | 0.0009 |
| low | DIF-SR + metadata | 408 | 0.0123 | 0.0037 | 0.0015 |
| mid | DIF-SR + metadata | 166 | 0.0120 | 0.0098 | 0.0090 |

## Bucket Definitions

- History buckets: `1-5`, `6-10`, `11-20`, `21+` based on train history length.
- Category entropy uses recent `department` history with Shannon entropy.
- Entropy buckets: `low < 0.5`, `mid < 1.0`, `high >= 1.0`.
