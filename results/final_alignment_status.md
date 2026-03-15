# Final Alignment Status

Dataset: H&M local `userhash32` purchase sequences

Aligned settings target:
- `seq_len=50`
- `train_epoch=20`
- `lr=1e-3`
- `embed_size=64`
- `n_heads=2`
- `n_layers=2`
- `drop_out=0.2`
- `batch_size=64`
- `seed=42`

Execution order:
1. Corrected SASRec
2. DIF-SR
3. SASRec + Metadata
4. DIF-SR + Metadata

## Runs

### 1. Corrected SASRec

- best epoch: `2`
- metrics:
  - `Recall@20 0.0113`
  - `Recall@50 0.0238`
  - `Recall@100 0.0387`
  - `NDCG@20 0.0040`
  - `NDCG@50 0.0065`
  - `NDCG@100 0.0089`
  - `MRR@20 0.0020`
- training time: `~4m 12s` to early stop observation window (`epoch 0-4`), best at epoch 2
- config path: [`/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.sanity_sasrec_fixed.json`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.sanity_sasrec_fixed.json)

### 2. DIF-SR

- best epoch: `0`
- metrics:
  - `Recall@20 0.0155`
  - `Recall@50 0.0315`
  - `Recall@100 0.0464`
  - `NDCG@20 0.0061`
  - `NDCG@50 0.0093`
  - `NDCG@100 0.0117`
  - `MRR@20 0.0036`
- training time: `~22m 56s`
- config path: [`/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.fair_difsr.json`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.fair_difsr.json)

### 3. SASRec + Metadata

- best epoch: `4`
- metrics:
  - `Recall@20 0.0101`
  - `Recall@50 0.0178`
  - `Recall@100 0.0274`
  - `NDCG@20 0.0034`
  - `NDCG@50 0.0049`
  - `NDCG@100 0.0065`
  - `MRR@20 0.0015`
- training time: `~22m 02s`
- config path: [`/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.fair_sasrec_meta.json`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.fair_sasrec_meta.json)

### 4. DIF-SR + Metadata

- best epoch: `3`
- metrics:
  - `Recall@20 0.0184`
  - `Recall@50 0.0333`
  - `Recall@100 0.0571`
  - `NDCG@20 0.0075`
  - `NDCG@50 0.0103`
  - `NDCG@100 0.0142`
  - `MRR@20 0.0045`
- training time: `~23m 29s` to best-epoch confirmation window (`epoch 0-6`), benchmark peaked at epoch 3 and degraded afterward
- config path: [`/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.fair_difsr_meta.json`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.fair_difsr_meta.json)
