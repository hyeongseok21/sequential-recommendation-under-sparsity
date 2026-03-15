# Sequential Recommendation Under Sparse Behavioral Signal

This repository packages a completed sequential recommendation study on sparse H&M purchase sequences. The artifact is organized as a portfolio-quality research report rather than an open-ended experiment workspace.

## 1. Problem

Sequential recommendation is difficult when user histories are short, item space is large, and popularity dominates interactions. This project studies whether metadata helps next-item prediction under that regime and whether a multi-interest backbone remains useful after baseline verification.

## 2. Dataset

Primary dataset: H&M local `userhash32` purchase sequences.

- Users: `22,258`
- Items: `29,785`
- Interactions: `135,412`
- Average sequence length: `9.35`
- Median sequence length: `6`
- Sparsity: `99.98%`
- Metadata:
  - `product_type`
  - `department`
  - `garment_group`

Two evaluation regimes are included.

Canonical evaluation:
- cold users removed
- cold items removed
- zero-history users removed
- repeat purchases removed

Service-style supplementary evaluation:
- cold-like users allowed
- zero-history users allowed
- repeat purchases allowed

The canonical evaluation remains the primary result. The service-style setting is a robustness analysis.

## 3. Baseline Verification

The original SASRec baseline was not reliable. Two issues were found:

1. the attention mask was not causal because `torch.tril(...)` was disabled in [`/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/models/Transformer.py`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/models/Transformer.py)
2. the original training recipe severely undertrained SASRec:
   - `lr=1e-4`
   - `batch_size=16`
   - `train_epoch=3`
   - `embed_size=32`
   - `n_layers=1`
   - `seq_len=30`

After restoring causal masking and rerunning a single stronger sanity configuration, SASRec improved from:

- `Recall@20 0.0006 -> 0.0113`
- `NDCG@20 0.0002 -> 0.0040`
- `MRR@20 0.0001 -> 0.0020`

That corrected run is the canonical SASRec baseline used in the final artifact.

Detailed note:
- [`/Users/conan/projects/personalized-fashion-recommendation/SASREC_SANITY_FIX.md`](/Users/conan/projects/personalized-fashion-recommendation/SASREC_SANITY_FIX.md)

## 4. Models

- `TopPopular`
- `Corrected SASRec`
- `DIF-SR`
- `DIF-SR + Metadata`

The final comparison uses aligned experimental conditions where possible:
- same split
- same seed
- same seen-item masking
- same leave-one-out evaluation
- `seq_len=50`
- `lr=1e-3`
- `embed_size=64`
- `n_heads=2`
- `n_layers=2`
- `dropout=0.2`
- `batch_size=64`

## 5. Canonical Results

Canonical aligned comparison:

| Model | Recall@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: |
| TopPopular | 0.0280 | 0.0102 | 0.0055 |
| Corrected SASRec | 0.0113 | 0.0040 | 0.0020 |
| DIF-SR | 0.0155 | 0.0061 | 0.0036 |
| DIF-SR + Metadata | 0.0184 | 0.0075 | 0.0045 |

Interpretation:
- `TopPopular` remains the strongest model, which confirms strong popularity dominance.
- `DIF-SR + Metadata` is the strongest personalized model.
- `DIF-SR` is stronger than corrected `SASRec` under aligned conditions.

Reference report:
- [`/Users/conan/projects/personalized-fashion-recommendation/reports/canonical_evaluation.md`](/Users/conan/projects/personalized-fashion-recommendation/reports/canonical_evaluation.md)

## 6. Supplementary Service-Style Evaluation

The service-style evaluation asks a different question: what changes when production-like conditions are allowed?

Service-style results:

| Model | Recall@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: |
| TopPopular | 0.0293 | 0.0111 | 0.0061 |
| Corrected SASRec | 0.0147 | 0.0099 | 0.0085 |
| DIF-SR | 0.0169 | 0.0086 | 0.0063 |
| DIF-SR + Metadata | 0.0202 | 0.0093 | 0.0063 |

Interpretation:
- aggregate metrics rise because repeat purchases make the task easier
- `DIF-SR + Metadata` is still the strongest personalized model on overall Recall@20
- `Corrected SASRec` becomes unusually strong in repeat-purchase cases

Reference report:
- [`/Users/conan/projects/personalized-fashion-recommendation/reports/service_style_evaluation.md`](/Users/conan/projects/personalized-fashion-recommendation/reports/service_style_evaluation.md)

## 7. Slice Analysis

Service-style slices were used to surface behavior under realistic conditions.

Cold-like users:
- best model: `DIF-SR + Metadata`
- `Recall@20 0.0233`
- `NDCG@20 0.0116`

Short-history users (`<=5`):
- best model: `DIF-SR + Metadata`
- `Recall@20 0.0239`
- `NDCG@20 0.0107`

Repeat purchase cases:
- best model: `Corrected SASRec`
- `Recall@20 0.3194`
- `NDCG@20 0.2640`

This suggests metadata is most useful when behavioral signal is weak, while repeat-heavy behavior rewards memorization more directly.

## 8. Key Insights

- Strong popularity dominance remains the central property of the dataset.
- Baseline verification materially changed the interpretation of the project.
- Metadata is useful under sparse behavioral signal, but most clearly when paired with `DIF-SR`.
- Research-style clean evaluation and service-style realistic evaluation tell different stories; both are useful.
- The most defensible conclusion is not “diffusion always wins,” but that after baseline correction and aligned comparison, `DIF-SR + Metadata` is the strongest personalized model in a sparse, popularity-heavy environment.

## 9. Reproducibility

Generate the final portfolio plots and curated reports from existing results:

```bash
source .venv/bin/activate
python experiments/package_portfolio_artifact.py
```

Regenerate the broader research analysis package:

```bash
source .venv/bin/activate
python experiments/run_evaluation.py
```

Primary artifact locations:
- configs index: [`/Users/conan/projects/personalized-fashion-recommendation/configs/README.md`](/Users/conan/projects/personalized-fashion-recommendation/configs/README.md)
- canonical report: [`/Users/conan/projects/personalized-fashion-recommendation/reports/canonical_evaluation.md`](/Users/conan/projects/personalized-fashion-recommendation/reports/canonical_evaluation.md)
- service-style report: [`/Users/conan/projects/personalized-fashion-recommendation/reports/service_style_evaluation.md`](/Users/conan/projects/personalized-fashion-recommendation/reports/service_style_evaluation.md)
- portfolio summary: [`/Users/conan/projects/personalized-fashion-recommendation/reports/research_summary.md`](/Users/conan/projects/personalized-fashion-recommendation/reports/research_summary.md)

Final plots:
- ![model comparison](/Users/conan/projects/personalized-fashion-recommendation/plots/final_model_comparison.png)
- ![canonical vs service](/Users/conan/projects/personalized-fashion-recommendation/plots/canonical_vs_service_comparison.png)
- ![service slice analysis](/Users/conan/projects/personalized-fashion-recommendation/plots/final_slice_analysis.png)

## Project Structure

```text
datasets/
configs/
experiments/
analysis/
plots/
reports/
README.md
```
