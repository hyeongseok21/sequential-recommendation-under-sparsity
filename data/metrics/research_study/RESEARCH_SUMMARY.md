# Sequential Recommendation with Metadata: Research Summary

## Problem
We study whether item metadata improves sequential recommendation on sparse H&M purchase sequences, and which user segments benefit most.

## Dataset
- Users: 22,258
- Items: 29,785
- Interactions: 135,412
- Average sequence length: 9.35
- Sparsity: 99.98%
- Metadata: product_type, department, garment_group

## Models
- SASRec
- SASRec + Metadata
- DIF-SR
- DIF-SR + Metadata

## Key Results
- Best overall model: DIF-SR + metadata (NDCG@20 0.0032, Recall@20 0.0083)
- Best cold-start slice (`history<=5`): DIF-SR + metadata (NDCG@20 0.0052)
- Best high-entropy slice: DIF-SR (NDCG@20 0.0027)

## Additional Analyses
- Frontier overlap is lowest for `SASRec vs DIF-SR + metadata` (mean overlap 0.0100), showing that DIF-SR family models retrieve substantially different candidates than SASRec.
- `DIF-SR + metadata` improves accuracy but sharply reduces recommendation diversity (mean category entropy 1.1027 vs 2.7170 for DIF-SR).
- `DIF-SR + metadata` remains head-heavy; tail exposure drops from 0.1209 in DIF-SR to 0.1081 after metadata injection.
- In high-entropy users, `DIF-SR` retains slightly better NDCG@20 than `DIF-SR + metadata`, suggesting multi-interest robustness mainly comes from the backbone.

## Interpretation
- Metadata consistently improves both backbones overall, but the gain is substantially larger on the DIF-SR backbone.
- Metadata is most helpful for short-history users, where sequence-only signal is weakest.
- DIF-SR changes the top-100 candidate frontier rather than merely reranking SASRec outputs, which supports the claim that multi-interest modeling changes candidate retrieval behavior.
- The accuracy gain from metadata comes with lower diversity and stronger concentration on popular categories, so metadata is acting more like a relevance sharpener than a diversity enhancer in this setting.

## Reproducibility
Run the full research analysis pipeline with:

```bash
python experiments/run_evaluation.py
```

Artifacts are written to `data/metrics/research_study/` and plots are written to `plots/`.
