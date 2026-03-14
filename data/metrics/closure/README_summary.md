# Sequential Recommendation with Metadata

## Problem
H&M purchase data contains users with short histories and potentially changing interests across categories.

## Hypothesis
Item metadata can improve sequential recommendation quality, especially for sparse-history or multi-interest users.

## Models
- SASRec
- SASRec + metadata
- DIF-SR
- DIF-SR + metadata

## Key Findings
- Best overall NDCG@20: DIF-SR + metadata (0.0032)
- Best sparse-history NDCG@20: DIF-SR + metadata (0.0052)
- Best multi-interest NDCG@20: DIF-SR (0.0026)
- MRR@20 is computed on the single-positive next-item setting and is numerically aligned with reciprocal-rank style evaluation.

## Limitations
- This closure compares models on a local userhash split, not the full production-scale corpus.
- Benchmark-best and test-best epochs can differ, so the main table uses the benchmark-best checkpoint policy consistently.
