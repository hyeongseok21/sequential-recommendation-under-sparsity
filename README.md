# Sequential Recommendation With Metadata on H&M Purchase Sequences

This repository packages a research-style sequential recommendation study on sparse H&M purchase histories.
The central question is whether item metadata improves next-item prediction, and which user segments benefit most.

## Problem

Fashion purchase histories are sparse and users often switch across categories.
This makes next-item prediction difficult for:

- short-history users with weak sequence evidence
- multi-interest users whose recent category transitions are frequent

We study whether item metadata can compensate for this sparsity, and whether a multi-interest backbone such as DIF-SR is more robust than SASRec.

## Dataset

- Dataset: H&M local `userhash32` purchase sequences
- Users: 22,258
- Items: 29,785
- Interactions: 135,412
- Average sequence length: 9.35
- Sparsity: 99.98%
- Metadata:
  - `product_type`
  - `department`
  - `garment_group`

Evaluation is next-item prediction with the benchmark-best checkpoint from each canonical closure run.

## Models

- `SASRec`
- `SASRec + Metadata`
- `DIF-SR`
- `DIF-SR + Metadata`

Metadata injection is frozen for closure mode; no new features or fusion variants are introduced in the final study.

## Experimental Setup

Primary metrics:

- `Recall@20`
- `Recall@50`
- `Recall@100`
- `NDCG@20`
- `NDCG@50`
- `NDCG@100`

Secondary metric:

- `MRR@20`

User analysis slices:

- sequence length:
  - short history `<= 5`
  - medium history `6-15`
  - long history `> 15`
- category entropy:
  - low entropy `< 0.5`
  - medium entropy `< 1.0`
  - high entropy `>= 1.0`

Additional analyses:

- top-100 candidate overlap
- top-20 recommendation diversity
- popularity bias / long-tail exposure

## Main Results

| Model | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | NDCG@50 | NDCG@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SASRec | 0.0006 | 0.0018 | 0.0036 | 0.0002 | 0.0004 | 0.0007 |
| SASRec + Metadata | 0.0012 | 0.0048 | 0.0071 | 0.0005 | 0.0012 | 0.0015 |
| DIF-SR | 0.0077 | 0.0119 | 0.0190 | 0.0024 | 0.0032 | 0.0043 |
| DIF-SR + Metadata | 0.0083 | 0.0190 | 0.0321 | 0.0032 | 0.0053 | 0.0074 |

The best overall model is `DIF-SR + Metadata`.

## Slice Findings

### Cold-start / Short History

`DIF-SR + Metadata` is best for short-history users:

- `Recall@20 = 0.0124`
- `NDCG@20 = 0.0052`

This is the clearest evidence that metadata helps when sequence evidence is weak.

### Multi-interest / High Entropy

For high-entropy users:

- `DIF-SR`: `Recall@20 = 0.0081`, `NDCG@20 = 0.0027`
- `DIF-SR + Metadata`: `Recall@20 = 0.0063`, `NDCG@20 = 0.0020`

This suggests multi-interest robustness comes primarily from the `DIF-SR` backbone, while metadata improves relevance more consistently in short-history settings than in high-entropy settings.

## Frontier, Diversity, and Popularity

### Candidate Frontier Change

Top-100 candidate overlap is very low between SASRec and DIF-SR family models:

- `SASRec vs DIF-SR`: mean overlap `0.0109`
- `SASRec vs DIF-SR + Metadata`: mean overlap `0.0100`
- `DIF-SR vs DIF-SR + Metadata`: mean overlap `0.0461`

This indicates DIF-SR changes the candidate frontier rather than merely reranking SASRec outputs.

### Recommendation Diversity

`DIF-SR + Metadata` improves accuracy but produces more concentrated recommendation lists:

- mean category entropy
  - `DIF-SR`: `2.7170`
  - `DIF-SR + Metadata`: `1.1027`

So metadata acts as a relevance sharpener in this setting, not a diversity enhancer.

### Popularity Bias

Top-20 exposure is strongly head-biased for DIF-SR family models:

- tail share
  - `SASRec`: `0.4761`
  - `SASRec + Metadata`: `0.4569`
  - `DIF-SR`: `0.1209`
  - `DIF-SR + Metadata`: `0.1081`

Metadata slightly improves DIF-SR accuracy, but does not reduce popularity bias.

## Key Insights

1. Metadata improves overall next-item prediction for both backbones, but the gain is much larger on top of DIF-SR.
2. Metadata is most useful for short-history users, where semantic item signals compensate for weak sequence evidence.
3. DIF-SR captures multi-interest behavior better than SASRec; SASRec family models are nearly non-functional on the high-entropy slice.
4. DIF-SR family models change the recommendation frontier substantially, but at the cost of stronger concentration on head items.
5. In this setting, better relevance and better diversity do not move together.

## Reproducibility

Run the full research evaluation pipeline with:

```bash
source .venv/bin/activate
python experiments/run_evaluation.py
```

Artifacts produced by the study:

- research metrics: [`data/metrics/research_study`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/research_study)
- plots: [`plots`](/Users/conan/projects/personalized-fashion-recommendation/plots)
- closure comparison tables: [`data/metrics/closure/RESULTS.md`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/RESULTS.md)

Generated figures:

- ![overall metrics](/Users/conan/projects/personalized-fashion-recommendation/plots/overall_metric_comparison.png)
- ![candidate overlap](/Users/conan/projects/personalized-fashion-recommendation/plots/candidate_overlap_histogram.png)
- ![diversity](/Users/conan/projects/personalized-fashion-recommendation/plots/diversity_comparison.png)
- ![popularity exposure](/Users/conan/projects/personalized-fashion-recommendation/plots/popularity_exposure.png)
- ![slice analysis](/Users/conan/projects/personalized-fashion-recommendation/plots/slice_analysis.png)

## Project Structure

```text
analysis/
  candidate_overlap.py
  diversity_analysis.py
  popularity_bias.py
  overall_metrics.py
  slice_analysis.py
  common.py
experiments/
  run_evaluation.py
models/
  sasrec.py
  dif_sr.py
plots/
data/metrics/research_study/
hm_refactored/
```
