# Service-Style Evaluation

This supplementary evaluation keeps the canonical experiment untouched and re-runs a second dataset variant with:
- `remove_cold_user = false`
- `remove_zero_history = false`
- `remove_recent_bought = false`
- `remove_cold_item = true`

The goal is to simulate a more realistic production environment where cold-like users remain, zero-history cases exist, and repeat purchases are allowed.

## Canonical vs Service-Style Comparison

| Model | Canonical Recall@20 | Service Recall@20 | Delta | Canonical NDCG@20 | Service NDCG@20 | Delta | Canonical MRR@20 | Service MRR@20 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TopPopular | 0.0280 | 0.0293 | +0.0013 | 0.0102 | 0.0111 | +0.0009 | 0.0055 | 0.0061 | +0.0006 |
| Corrected SASRec | 0.0113 | 0.0147 | +0.0034 | 0.0040 | 0.0099 | +0.0058 | 0.0020 | 0.0085 | +0.0066 |
| DIF-SR | 0.0155 | 0.0169 | +0.0015 | 0.0061 | 0.0086 | +0.0025 | 0.0036 | 0.0063 | +0.0027 |
| DIF-SR + Metadata | 0.0184 | 0.0202 | +0.0017 | 0.0075 | 0.0093 | +0.0019 | 0.0045 | 0.0063 | +0.0018 |

## Slice Metrics

### cold-like users

| Model | Users | Recall@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| TopPopular | 430 | 0.0163 | 0.0066 | 0.0040 |
| Corrected SASRec | 430 | 0.0070 | 0.0019 | 0.0006 |
| DIF-SR | 430 | 0.0116 | 0.0051 | 0.0034 |
| DIF-SR + Metadata | 430 | 0.0233 | 0.0116 | 0.0081 |

### short history users (<=5)

| Model | Users | Recall@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| TopPopular | 922 | 0.0304 | 0.0105 | 0.0051 |
| Corrected SASRec | 922 | 0.0087 | 0.0050 | 0.0040 |
| DIF-SR | 922 | 0.0174 | 0.0092 | 0.0070 |
| DIF-SR + Metadata | 922 | 0.0239 | 0.0107 | 0.0070 |

### repeat purchase cases

| Model | Users | Recall@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| TopPopular | 72 | 0.1528 | 0.0808 | 0.0609 |
| Corrected SASRec | 72 | 0.3194 | 0.2640 | 0.2473 |
| DIF-SR | 72 | 0.1111 | 0.0800 | 0.0713 |
| DIF-SR + Metadata | 72 | 0.1389 | 0.0869 | 0.0720 |

## Analysis

1. Including cold/zero-history users does not collapse all sequential models equally. `Corrected SASRec` changed by +0.0034 Recall@20, `DIF-SR` by +0.0015, and `DIF-SR + Metadata` by +0.0017.
2. Allowing repeat purchases creates an easier recommendation path for popularity- and repeat-heavy behavior, so slice performance on `repeat purchase cases` should be interpreted alongside `TopPopular`.
3. Under relaxed filters, metadata still helps the diffusion backbone: `DIF-SR + Metadata` improves Recall@20 over `DIF-SR` by +0.0032. The same claim is not made for SASRec because the supplementary run did not include a metadata SASRec branch.

Observed service-style best epochs:
- `Corrected SASRec`: epoch `2`
- `DIF-SR`: epoch `0`
- `DIF-SR + Metadata`: epoch `0`
