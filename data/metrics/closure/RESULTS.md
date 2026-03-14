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
