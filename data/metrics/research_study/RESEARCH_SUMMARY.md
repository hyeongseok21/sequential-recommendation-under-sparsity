# Sequential Recommendation with Metadata: Research Summary

## Problem
This study evaluates sequential recommendation under extreme sparsity on H&M purchase sequences. The primary question is whether metadata helps next-item prediction when behavioral signal is weak.

## Dataset
- Users: 22,258
- Items: 29,785
- Interactions: 135,412
- Average sequence length: 9.35
- Median sequence length: 6
- Sparsity: 99.98%
- Metadata: product_type, department, garment_group

## Baseline Sanity Check
The initial SASRec baseline was not trustworthy. Investigation found:
- missing causal masking because `torch.tril(...)` was disabled
- severe undertraining due to a weak configuration

After restoring causal masking and rerunning a stronger sanity configuration, SASRec improved from:
- `Recall@20 0.0006 -> 0.0113`
- `NDCG@20 0.0002 -> 0.0040`
- `MRR@20 0.0001 -> 0.0020`

This corrected run is the canonical transformer baseline for the final artifact.

## Canonical Aligned Results
- TopPopular: `Recall@20 0.0280`, `NDCG@20 0.0102`, `MRR@20 0.0055`
- Corrected SASRec: `Recall@20 0.0113`, `NDCG@20 0.0040`, `MRR@20 0.0020`
- DIF-SR: `Recall@20 0.0155`, `NDCG@20 0.0061`, `MRR@20 0.0036`
- DIF-SR + Metadata: `Recall@20 0.0184`, `NDCG@20 0.0075`, `MRR@20 0.0045`

## Service-Style Supplementary Results
- TopPopular: `Recall@20 0.0293`, `NDCG@20 0.0111`, `MRR@20 0.0061`
- Corrected SASRec: `Recall@20 0.0147`, `NDCG@20 0.0099`, `MRR@20 0.0085`
- DIF-SR: `Recall@20 0.0169`, `NDCG@20 0.0086`, `MRR@20 0.0063`
- DIF-SR + Metadata: `Recall@20 0.0202`, `NDCG@20 0.0093`, `MRR@20 0.0063`

## Slice Findings
- cold-like users: `DIF-SR + Metadata` is best
- short-history users (`<=5`): `DIF-SR + Metadata` is best
- repeat purchase cases: `Corrected SASRec` is best

## Additional Analyses
- candidate frontier overlap between SASRec and DIF-SR families is low, so DIF-SR changes the recommendation frontier rather than only reranking
- metadata improves DIF-SR accuracy but reduces recommendation diversity
- popularity exposure remains head-heavy even for the strongest personalized models

## Interpretation
- baseline verification was necessary before any model claim became credible
- the dataset is strongly popularity-dominated
- short histories materially limit sequential signal
- after aligned comparison, `DIF-SR + Metadata` is the strongest personalized model
- the final project should emphasize both baseline verification and behavior under sparse, popularity-heavy conditions

## Reproducibility
```bash
source .venv/bin/activate
python experiments/package_portfolio_artifact.py
```
