# Service-Style Supplementary Evaluation

Supplementary robustness analysis under relaxed production-like filtering:
- cold-like users allowed
- zero-history users allowed
- repeat purchases allowed

| Model | Canonical Recall@20 | Service Recall@20 | Canonical NDCG@20 | Service NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| TopPopular | 0.0280 | 0.0293 | 0.0102 | 0.0111 |
| Corrected SASRec | 0.0113 | 0.0147 | 0.0040 | 0.0099 |
| DIF-SR | 0.0155 | 0.0169 | 0.0061 | 0.0086 |
| DIF-SR + Metadata | 0.0184 | 0.0202 | 0.0075 | 0.0093 |

## Key Slices

| Slice | Best Model | Recall@20 | NDCG@20 |
| --- | --- | ---: | ---: |
| cold-like users | DIF-SR + Metadata | 0.0233 | 0.0116 |
| short history users (<=5) | DIF-SR + Metadata | 0.0239 | 0.0107 |
| repeat purchase cases | Corrected SASRec | 0.3194 | 0.2640 |

Interpretation:
- `DIF-SR + Metadata` is best on cold-like and short-history users.
- `Corrected SASRec` is best on repeat purchase cases, which is consistent with memorization-heavy behavior.
- Service-style metrics rise overall because repeat purchase cases make the task easier.
