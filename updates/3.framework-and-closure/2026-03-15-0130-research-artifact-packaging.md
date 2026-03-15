# Research Artifact Packaging

## Goal

- closure 결과를 포트폴리오 수준의 연구형 산출물로 재패키징한다.
- 추가 학습 없이 기존 canonical checkpoint를 재사용한다.

## Added

- reproducible evaluation entrypoint
  - [`experiments/run_evaluation.py`](/Users/conan/projects/personalized-fashion-recommendation/experiments/run_evaluation.py)
- analysis scripts
  - [`analysis/candidate_overlap.py`](/Users/conan/projects/personalized-fashion-recommendation/analysis/candidate_overlap.py)
  - [`analysis/diversity_analysis.py`](/Users/conan/projects/personalized-fashion-recommendation/analysis/diversity_analysis.py)
  - [`analysis/popularity_bias.py`](/Users/conan/projects/personalized-fashion-recommendation/analysis/popularity_bias.py)
  - [`analysis/overall_metrics.py`](/Users/conan/projects/personalized-fashion-recommendation/analysis/overall_metrics.py)
  - [`analysis/slice_analysis.py`](/Users/conan/projects/personalized-fashion-recommendation/analysis/slice_analysis.py)
  - [`analysis/common.py`](/Users/conan/projects/personalized-fashion-recommendation/analysis/common.py)
- model spec wrappers
  - [`models/sasrec.py`](/Users/conan/projects/personalized-fashion-recommendation/models/sasrec.py)
  - [`models/dif_sr.py`](/Users/conan/projects/personalized-fashion-recommendation/models/dif_sr.py)
- plots
  - [`plots/overall_metric_comparison.png`](/Users/conan/projects/personalized-fashion-recommendation/plots/overall_metric_comparison.png)
  - [`plots/candidate_overlap_histogram.png`](/Users/conan/projects/personalized-fashion-recommendation/plots/candidate_overlap_histogram.png)
  - [`plots/diversity_comparison.png`](/Users/conan/projects/personalized-fashion-recommendation/plots/diversity_comparison.png)
  - [`plots/popularity_exposure.png`](/Users/conan/projects/personalized-fashion-recommendation/plots/popularity_exposure.png)
  - [`plots/slice_analysis.png`](/Users/conan/projects/personalized-fashion-recommendation/plots/slice_analysis.png)
- research summary
  - [`data/metrics/research_study/RESEARCH_SUMMARY.md`](/Users/conan/projects/personalized-fashion-recommendation/data/metrics/research_study/RESEARCH_SUMMARY.md)
- root README research rewrite
  - [`README.md`](/Users/conan/projects/personalized-fashion-recommendation/README.md)

## Main Findings

- overall best는 `DIF-SR + metadata`
- short-history user에서는 metadata 이득이 가장 큼
- high-entropy user에서는 `DIF-SR`가 `DIF-SR + metadata`보다 약간 우세
- `DIF-SR` 계열은 `SASRec` 계열과 top-100 후보군이 거의 겹치지 않음
- metadata는 accuracy를 올리지만 diversity를 줄이고 인기 아이템 쏠림을 완화하지는 못함

## Interpretation

- 이 프로젝트의 strongest story는 `metadata always helps`가 아니라,
  - `metadata helps most under sparse history`
  - `multi-interest robustness comes mainly from DIF-SR backbone`
  - `accuracy gain can come with diversity / popularity tradeoffs`
  이다.
