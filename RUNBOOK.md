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

## 자주 쓰는 리포트

정책 비교:

```bash
python3 scripts/report_checkpoint_policy.py --loss-path <LOSS_PATH>
```

single run dual-best:

```bash
python3 scripts/report_dual_best.py --checkpoint-dir <CHECKPOINT_DIR>
```

run set leaderboard:

```bash
python3 scripts/report_eval_gap_leaderboard.py --root hm_refactored/hm_out/checkpoints --pattern difsr --output-json <JSON_PATH> --output-csv <CSV_PATH>
```

## 문서 검사

```bash
python3 scripts/lint_experiment_docs.py
```
