# Recsys Runbook

## 목적

- 이 저장소에서 실험을 실제로 실행하고 결과 artifact를 확인하는 절차를 짧게 정리한다.
- 정책과 판단 기준은 `AGENT.md`를 우선한다.

## Canonical Command

```bash
source .venv/bin/activate
python hm_refactored/train.py --config_path <CONFIG_PATH>
```

## 실행 절차

1. champion config 또는 treatment config를 고른다.
2. canonical command로 학습 또는 fast-scout를 실행한다.
3. 결과를 다음 artifact에서 확인한다.
   - `hm_refactored/hm_out/checkpoints/<save_name>/loss.txt`
   - `hm_refactored/hm_out/checkpoints/<save_name>/epoch_summary.json`
   - `data/metrics/gate_result.json`
   - `data/metrics/experiment_memory.csv`

## Checkpoint 평가

저장된 checkpoint를 재학습 없이 평가할 때:

```bash
source .venv/bin/activate
python hm_refactored/train.py --config_path <CONFIG_PATH> --eval_checkpoint <EPOCH>
```

결과는 다음 파일에 저장된다.

- `hm_refactored/hm_out/checkpoints/<save_name>/eval_checkpoint_<epoch>.json`

promoted champion은 기본적으로 `benchmark-best`와 `test-best`를 둘 다 확인한다.

권장 절차:

```bash
source .venv/bin/activate
python hm_refactored/train.py --config_path <CONFIG_PATH> --eval_checkpoint <BENCHMARK_BEST_EPOCH>
python hm_refactored/train.py --config_path <CONFIG_PATH> --eval_checkpoint <TEST_BEST_EPOCH>
python3 scripts/report_dual_best.py --checkpoint-dir <CHECKPOINT_DIR> --output <OUTPUT_JSON>
```

## 자주 쓰는 리포트

정책 비교:

```bash
python3 scripts/report_checkpoint_policy.py --loss-path <LOSS_PATH>
```

single run dual-best:

```bash
python3 scripts/report_dual_best.py --checkpoint-dir <CHECKPOINT_DIR>
```

운영 기본 규칙:

- champion 승격 직후에는 `single run dual-best`를 항상 생성한다.
- `benchmark-best`와 `test-best`가 다르면 update log에 둘 다 남긴다.
- serving phase에서는 가능하면 slice report를 함께 남긴다.
- slice 정의와 해석 기준은 [`SLICE_EVALUATION.md`](/Users/conan/projects/personalized-fashion-recommendation/SLICE_EVALUATION.md)를 따른다.

run set leaderboard:

```bash
python3 scripts/report_eval_gap_leaderboard.py --root hm_refactored/hm_out/checkpoints --pattern difsr --output-json <JSON_PATH> --output-csv <CSV_PATH>
```

## 문서 검사

```bash
python3 scripts/lint_experiment_docs.py
```
