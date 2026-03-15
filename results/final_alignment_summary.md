# Final Alignment Summary

Dataset: H&M local `userhash32` purchase sequences

Aligned conditions:
- same local split and seed (`42`)
- same leave-one-out evaluation
- same seen-item masking and candidate space
- `seq_len=50`
- `lr=1e-3`
- `embed_size=64`
- `n_heads=2`
- `n_layers=2`
- `drop_out=0.2`
- `batch_size=64`

## Final Comparison Table

| Model | Best Epoch | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | NDCG@50 | NDCG@100 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TopPopular | - | 0.0280 | 0.0428 | 0.0648 | 0.0102 | - | - | 0.0055 |
| Corrected SASRec | 2 | 0.0113 | 0.0238 | 0.0387 | 0.0040 | 0.0065 | 0.0089 | 0.0020 |
| SASRec + Metadata | 4 | 0.0101 | 0.0178 | 0.0274 | 0.0034 | 0.0049 | 0.0065 | 0.0015 |
| DIF-SR | 0 | 0.0155 | 0.0315 | 0.0464 | 0.0061 | 0.0093 | 0.0117 | 0.0036 |
| DIF-SR + Metadata | 3 | 0.0184 | 0.0333 | 0.0571 | 0.0075 | 0.0103 | 0.0142 | 0.0045 |

## Interpretation

`TopPopular` still dominates all personalized models at `Recall@20` and `NDCG@20`, which confirms that this dataset remains strongly popularity-heavy even after baseline correction. The corrected `SASRec` is now a minimally credible baseline, but sequential personalization is still constrained by short histories and extreme sparsity.

Under aligned conditions, `DIF-SR` outperforms corrected `SASRec`, and `DIF-SR + Metadata` is the strongest personalized model in this final pass. Metadata does not help the SASRec family here: `SASRec + Metadata` is slightly worse than corrected `SASRec` across all reported metrics. By contrast, metadata improves the `DIF-SR` family: `DIF-SR + Metadata` beats `DIF-SR` on every reported metric in this aligned pass.

## Final Answers

- Does `TopPopular` still dominate?
  - Yes. Popularity remains a stronger signal than any personalized sequential model in this local split.
- Does metadata help?
  - Yes for `DIF-SR`, no for `SASRec` in this aligned setup.
- Can we claim backbone superiority?
  - Yes, with narrower wording than before. Under aligned budget and evaluation conditions, `DIF-SR` is stronger than corrected `SASRec`, and `DIF-SR + Metadata` is the best personalized model.
- What should the project conclusion emphasize?
  - Both. The artifact should emphasize baseline verification first, then report that in a sparse and popularity-dominated environment, `DIF-SR + Metadata` is the strongest personalized approach under aligned conditions.

## Recommended Framing

The scientifically credible conclusion is not that diffusion universally wins, but that careful baseline verification materially changed the interpretation. After fixing the broken SASRec baseline and aligning training conditions, personalized models still lag `TopPopular`, which highlights the severity of sparsity and popularity bias in the H&M local split. Within that difficult setting, `DIF-SR + Metadata` provides the strongest personalized ranking quality, while metadata only helps when paired with the multi-interest backbone.
