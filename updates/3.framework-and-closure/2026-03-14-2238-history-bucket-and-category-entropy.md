# History Bucket And Category Entropy

## Context

- closure canonical result set already existed for:
  - `SASRec`
  - `SASRec + metadata`
  - `DIF-SR`
  - `DIF-SR + metadata`
- user-level metrics already contained `history_len`
- additional user analysis artifacts were needed without expanding model scope

## Added

- history length bucket analysis
  - `1-5`
  - `6-10`
  - `11-20`
  - `21+`
- category entropy analysis
  - computed on recent `department` history
  - Shannon entropy
  - bucketed as:
    - `low < 0.5`
    - `mid < 1.0`
    - `high >= 1.0`

## Artifacts

- [`data/metrics/closure/history_bucket_results.json`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/history_bucket_results.json)
- [`data/metrics/closure/history_bucket_results.csv`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/history_bucket_results.csv)
- [`data/metrics/closure/category_entropy_results.json`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/category_entropy_results.json)
- [`data/metrics/closure/category_entropy_results.csv`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/category_entropy_results.csv)
- [`data/metrics/closure/plots/history_bucket_comparison.png`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/plots/history_bucket_comparison.png)
- [`data/metrics/closure/plots/category_entropy_comparison.png`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/closure/plots/category_entropy_comparison.png)

## Quick Read

- `DIF-SR + metadata` is strongest in the shortest-history bucket `1-5`
- `DIF-SR` is strongest in `6-10`
- `SASRec` family remains weak across most buckets
- for category entropy:
  - `DIF-SR + metadata` is especially strong in `mid` entropy users on `NDCG@20`
  - `DIF-SR` is slightly stronger than `DIF-SR + metadata` in `high` entropy users

## Interpretation

- metadata is most clearly helpful when history signal is limited
- very high category-switching users still look more like a backbone problem than a metadata-only problem
- this supports the current closure story:
  - metadata helps sparse-history users
  - multi-interest robustness mainly comes from the `DIF-SR` backbone
