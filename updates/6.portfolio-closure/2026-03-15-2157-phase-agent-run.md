# Phase Agent Run

## Goal

- active phase를 문서에서 자동 감지하고 phase 맞춤 명령을 실행한다.
- 감지된 phase: `P6`
- 가설: closure phase에서는 새 탐색보다 결과 패키징과 문서 일관성 검증이 더 중요하다.
- dry-run: `false`

## Commands

### lint framework docs

```bash
/Users/conan/projects/personalized-fashion-recommendation/.venv/bin/python scripts/lint_experiment_docs.py
```

- return code: `0`
- stdout tail:

```text
Experiment framework docs look consistent.
```

### run research analysis

```bash
/Users/conan/projects/personalized-fashion-recommendation/.venv/bin/python experiments/run_evaluation.py
```

- return code: `0`
- stdout tail:

```text
Num Unique Item : 1332
Num Negatives : 100
{
  "research_summary": "/Users/conan/projects/personalized-fashion-recommendation/data/metrics/research_study/RESEARCH_SUMMARY.md",
  "candidate_overlap_summary": "/Users/conan/projects/personalized-fashion-recommendation/data/metrics/research_study/candidate_overlap_summary.json",
  "overall_metric_table": "/Users/conan/projects/personalized-fashion-recommendation/data/metrics/research_study/overall_metric_table.json",
  "diversity_summary": "/Users/conan/projects/personalized-fashion-recommendation/data/metrics/research_study/diversity_summary.json",
  "popularity_exposure_summary": "/Users/conan/projects/personalized-fashion-recommendation/data/metrics/research_study/popularity_exposure_summary.json",
  "slice_study": "/Users/conan/projects/personalized-fashion-recommendation/data/metrics/research_study/slice_study.json",
  "plots_dir": "/Users/conan/projects/personalized-fashion-recommendation/plots"
}
```
- stderr tail:

```text
        "recent_ten": false
    },
    "mlflow_params": {
        "remote_server_uri": "/Users/conan/projects/personalized-fashion-recommendation/mlruns",
        "experiment_name": "hm_closure_sasrec"
    }
}
09:56:59:PM:INFO: Loading Dataset.
09:56:59:PM:INFO: Preprocessed data already exist. Loading...
09:56:59:PM:INFO: Number of users : 22258	Number of items : 29785
09:56:59:PM:INFO: Initialize Dataset & Negative Sampler.
09:56:59:PM:INFO: Initializing Model.
```

### run service-style evaluation

```bash
/Users/conan/projects/personalized-fashion-recommendation/.venv/bin/python scripts/generate_service_style_eval.py
```

- return code: `0`
- stdout tail:

```text
Num Unique Item : 1632
Num Negatives : 100
Num Unique Item : 1632
Num Negatives : 100
Num Unique Item : 1632
Num Negatives : 100
Num Unique Item : 1632
Num Negatives : 100
```
- stderr tail:

```text
        "recent_ten": false
    },
    "mlflow_params": {
        "remote_server_uri": "/Users/conan/projects/personalized-fashion-recommendation/mlruns",
        "experiment_name": "hm_service_sasrec"
    }
}
09:57:15:PM:INFO: Loading Dataset.
09:57:15:PM:INFO: Preprocessed data already exist. Loading...
09:57:15:PM:INFO: Number of users : 22258	Number of items : 29785
09:57:15:PM:INFO: Initialize Dataset & Negative Sampler.
09:57:15:PM:INFO: Initializing Model.
```

### package portfolio artifact

```bash
/Users/conan/projects/personalized-fashion-recommendation/.venv/bin/python experiments/package_portfolio_artifact.py
```

- return code: `0`

## Current Verdict

- overall verdict: `PASS`
- update log path: `updates/6.portfolio-closure/2026-03-15-2157-phase-agent-run.md`

## Notes

- 현재 구현은 active phase가 `P6`일 때 closure automation을 수행한다.
- 다른 phase는 추후 축별 baseline/treatment 루프로 확장할 수 있다.
