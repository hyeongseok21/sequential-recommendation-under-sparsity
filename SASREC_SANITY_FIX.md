# SASRec Sanity Fix

## Corrected SASRec code change summary

- Restored proper causal masking in [`/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/models/Transformer.py`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/models/Transformer.py) by applying `torch.tril(...)` inside `CustomSASRec._get_sequence_mask`.
- Added one literature-style sanity config at [`/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.sanity_sasrec_fixed.json`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.sanity_sasrec_fixed.json):
  - `seq_len=50`
  - `train_epoch=20` nominal target
  - `lr=1e-3`
  - `embed_size=64`
  - `n_heads=2`
  - `n_layers=2`
  - `drop_out=0.2`
  - `batch_size=64`
- Same dataset and evaluation protocol were kept.
- The run was stopped after epoch 4 because benchmark performance peaked at epoch 2 and then degraded. The corrected baseline below uses the best observed checkpoint at epoch 2.

## Root-cause note

- The missing causal mask was not removed by a recent experiment.
- Git history shows it was already commented out in the first SASRec implementation commit:
  - `87ed268` `omnious personalized recommendation system`
- It was then propagated forward when the model was later split/refactored:
  - `509ab2c` `divide CustomMetaSASRec`
- The most plausible explanation is that the original author temporarily disabled the causal mask while experimenting with alternate attention masks or debugging training behavior, and the change remained in the codebase.

## Final metric table

All metrics below are test-side closure metrics on the same local `userhash32` evaluation set.

| Model | Checkpoint | Recall@20 | NDCG@20 | MRR@20 |
| --- | --- | ---: | ---: | ---: |
| TopPopular | n/a | 0.0280 | 0.0102 | 0.0055 |
| Old SASRec | closure best | 0.0006 | 0.0002 | 0.0001 |
| Corrected SASRec | epoch 2 | 0.0113 | 0.0040 | 0.0020 |

Reference benchmark-side metrics for corrected SASRec epoch 2:

| Model | Benchmark Recall@20 | Benchmark NDCG@20 |
| --- | ---: | ---: |
| Corrected SASRec | 0.0250 | 0.0099 |

## Interpretation

The corrected SASRec is now a minimally credible sanity baseline.

- It is no longer catastrophically weak.
- It improves substantially over the old baseline:
  - `Recall@20`: `0.0006 -> 0.0113`
  - `NDCG@20`: `0.0002 -> 0.0040`
  - `MRR@20`: `0.0001 -> 0.0020`
- The huge SASRec-vs-DIF-SR gap was partly inflated by an unreliable SASRec setup.

However, the corrected SASRec is still weaker than TopPopular on this split.

- That means the baseline is now credible enough for sanity comparison.
- It also suggests that extreme sparsity and short histories still limit sequence modeling on this dataset.
- So the remaining gap is likely a mix of true dataset difficulty and architecture advantage, not just a broken baseline.

## Recommended next use

- Use this corrected SASRec setup as the new sanity baseline when discussing DIF-SR gains.
- Do not use the old closure SASRec as a research baseline anymore.
- If more time is available later, only do a small follow-up around this corrected point, not a broad search.
