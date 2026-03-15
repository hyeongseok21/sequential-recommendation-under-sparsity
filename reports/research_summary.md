# Research Portfolio Summary

This repository studies sequential recommendation under sparse interaction regimes using H&M purchase sequences.

Main conclusions:
- baseline verification was necessary because the original SASRec baseline was invalid
- the dataset is sparse and strongly popularity-dominated
- `DIF-SR + Metadata` is the strongest personalized model in the canonical aligned comparison
- metadata is most useful when behavioral signal is weak, especially in cold-like and short-history service slices

Primary entry points:
- canonical summary: [canonical_evaluation.md](/Users/conan/projects/personalized-fashion-recommendation/reports/canonical_evaluation.md)
- service-style summary: [service_style_evaluation.md](/Users/conan/projects/personalized-fashion-recommendation/reports/service_style_evaluation.md)
- baseline sanity check: [SASREC_SANITY_FIX.md](/Users/conan/projects/personalized-fashion-recommendation/SASREC_SANITY_FIX.md)
