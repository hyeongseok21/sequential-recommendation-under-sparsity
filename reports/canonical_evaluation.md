# Canonical Evaluation

Primary artifact using the clean research-style dataset.

| Model | Recall@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: |
| TopPopular | 0.0280 | 0.0102 | 0.0055 |
| Corrected SASRec | 0.0113 | 0.0040 | 0.0020 |
| DIF-SR | 0.0155 | 0.0061 | 0.0036 |
| DIF-SR + Metadata | 0.0184 | 0.0075 | 0.0045 |

Interpretation:
- `TopPopular` remains the strongest model, which confirms strong popularity dominance.
- `DIF-SR + Metadata` is the strongest personalized model under aligned conditions.
- `Corrected SASRec` is the verified transformer baseline after causal masking and training fixes.
