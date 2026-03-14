#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
HM_ROOT = REPO_ROOT / "hm_refactored"
if str(HM_ROOT) not in sys.path:
    sys.path.insert(0, str(HM_ROOT))

from train import Trainer  # noqa: E402
from util.metric import hit, map_, ndcg  # noqa: E402


CANONICAL_MODELS = [
    ("SASRec", REPO_ROOT / "hm_refactored/configs/config.closure_sasrec.json"),
    ("SASRec + metadata", REPO_ROOT / "hm_refactored/configs/config.closure_sasrec_meta.json"),
    ("DIF-SR", REPO_ROOT / "hm_refactored/configs/config.closure_difsr.json"),
    ("DIF-SR + metadata", REPO_ROOT / "hm_refactored/configs/config.closure_difsr_meta.json"),
]


def load_best_benchmark_epoch(checkpoint_dir: Path) -> int:
    payload = json.loads((checkpoint_dir / "epoch_summary.json").read_text())
    return int(payload["best_benchmark"]["epoch"])


def build_slice_flags(data_dict, sparse_threshold: int, recent_window: int):
    history_len = {int(user): len(items) for user, items in data_dict["user_train_dict"].items()}
    sparse_users = {user for user, length in history_len.items() if length <= sparse_threshold}

    train_df = data_dict["train_df"][["user_id", "timestamp", "department_no"]].copy()
    train_df = train_df.sort_values(["user_id", "timestamp"])
    multi_interest_users = set()
    for user_id, group in train_df.groupby("user_id"):
        recent_departments = group["department_no"].tolist()[-recent_window:]
        if not recent_departments:
            continue
        unique_departments = len(set(recent_departments))
        transitions = sum(
            1 for left, right in zip(recent_departments[:-1], recent_departments[1:]) if left != right
        )
        if unique_departments >= 3 or transitions >= 4:
            multi_interest_users.add(int(user_id))

    return {
        "sparse-history": sparse_users,
        "multi-interest": multi_interest_users,
    }


def evaluate_test_checkpoint(trainer: Trainer, checkpoint_epoch: int, sparse_threshold: int, recent_window: int):
    checkpoint_path = (
        Path(trainer.train_params["save_path"])
        / trainer.train_params["save_name"]
        / f"{checkpoint_epoch:06d}_epoch.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.model.eval()

    slice_flags = build_slice_flags(trainer.data_dict, sparse_threshold=sparse_threshold, recent_window=recent_window)
    rows = []
    top_k = int(trainer.train_params["top_k"])

    for user_batch, mask, history, history_mask in trainer.test_dl:
        user_tensor = trainer._to_device_tensor(user_batch[:, 0], torch.long)
        history_tensor = trainer._to_device_tensor(history, torch.long)
        history_mask_tensor = trainer._to_device_tensor(history_mask, torch.long)
        _ = mask  # kept for parity with trainer.test_process_batch

        predictions, _, _ = trainer.model.recommend(user_tensor, history_tensor, history_mask_tensor)
        predictions = predictions.detach()
        effective_top_k = min(top_k, predictions.shape[1])
        _, recommends = torch.topk(predictions, effective_top_k)
        recommends = recommends.detach().cpu().numpy()

        for batch_idx, user_idx in enumerate(user_tensor.detach().cpu().numpy().tolist()):
            source_user_id = trainer.data_dict["idx2user"][int(user_idx)]
            gt_items = [trainer.data_dict["item2idx"].get(i, -1) for i in trainer.data_dict["user_last_test_dict"][source_user_id]]
            recs = recommends[batch_idx].tolist()

            recall20 = hit([gt_items], [recs], batch=True)[0]
            ndcg20 = ndcg([gt_items], [recs], batch=True)[0]
            mrr20 = map_([gt_items], [recs], top_k, batch=True)[0]

            rows.append(
                {
                    "user_idx": int(user_idx),
                    "source_user_id": str(source_user_id),
                    "history_len": len(trainer.data_dict["user_train_dict"].get(int(user_idx), [])),
                    "recall@20": float(recall20),
                    "ndcg@20": float(ndcg20),
                    "mrr@20": float(mrr20),
                    "is_sparse_history": int(int(user_idx) in slice_flags["sparse-history"]),
                    "is_multi_interest": int(int(user_idx) in slice_flags["multi-interest"]),
                }
            )

    return rows


def summarize_rows(rows, predicate=None):
    selected = [row for row in rows if predicate is None or predicate(row)]
    if not selected:
        return {"num_users": 0, "recall@20": 0.0, "ndcg@20": 0.0, "mrr@20": 0.0}
    return {
        "num_users": len(selected),
        "recall@20": float(np.mean([row["recall@20"] for row in selected])),
        "ndcg@20": float(np.mean([row["ndcg@20"] for row in selected])),
        "mrr@20": float(np.mean([row["mrr@20"] for row in selected])),
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def render_overall_chart(overall_rows, output_path: Path):
    models = [row["model"] for row in overall_rows]
    recall = [row["recall@20"] for row in overall_rows]
    ndcg_vals = [row["ndcg@20"] for row in overall_rows]
    mrr_vals = [row["mrr@20"] for row in overall_rows]

    x = np.arange(len(models))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, recall, width, label="Recall@20")
    ax.bar(x, ndcg_vals, width, label="NDCG@20")
    ax.bar(x + width, mrr_vals, width, label="MRR@20")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Metric")
    ax.set_title("Closure Model Comparison")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def render_slice_chart(slice_rows, output_path: Path):
    slice_names = ["sparse-history", "multi-interest"]
    models = sorted({row["model"] for row in slice_rows})
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, slice_name in zip(axes, slice_names):
        current = [row for row in slice_rows if row["slice"] == slice_name]
        current = sorted(current, key=lambda row: models.index(row["model"]))
        recalls = [row["recall@20"] for row in current]
        ndcgs = [row["ndcg@20"] for row in current]
        ax.bar(x - width / 2, recalls, width, label="Recall@20")
        ax.bar(x + width / 2, ndcgs, width, label="NDCG@20")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(slice_name)
    axes[0].set_ylabel("Metric")
    axes[0].legend()
    fig.suptitle("Slice Comparison")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def build_readme_summary(overall_rows, slice_rows):
    overall_sorted = sorted(overall_rows, key=lambda row: row["ndcg@20"], reverse=True)
    best_overall = overall_sorted[0]

    sparse_rows = [row for row in slice_rows if row["slice"] == "sparse-history"]
    multi_rows = [row for row in slice_rows if row["slice"] == "multi-interest"]
    best_sparse = sorted(sparse_rows, key=lambda row: row["ndcg@20"], reverse=True)[0]
    best_multi = sorted(multi_rows, key=lambda row: row["ndcg@20"], reverse=True)[0]

    lines = [
        "# Sequential Recommendation with Metadata",
        "",
        "## Problem",
        "H&M purchase data contains users with short histories and potentially changing interests across categories.",
        "",
        "## Hypothesis",
        "Item metadata can improve sequential recommendation quality, especially for sparse-history or multi-interest users.",
        "",
        "## Models",
        "- SASRec",
        "- SASRec + metadata",
        "- DIF-SR",
        "- DIF-SR + metadata",
        "",
        "## Key Findings",
        f"- Best overall NDCG@20: {best_overall['model']} ({best_overall['ndcg@20']:.4f})",
        f"- Best sparse-history NDCG@20: {best_sparse['model']} ({best_sparse['ndcg@20']:.4f})",
        f"- Best multi-interest NDCG@20: {best_multi['model']} ({best_multi['ndcg@20']:.4f})",
        "- MRR@20 is computed on the single-positive next-item setting and is numerically aligned with reciprocal-rank style evaluation.",
        "",
        "## Limitations",
        "- This closure compares models on a local userhash split, not the full production-scale corpus.",
        "- Benchmark-best and test-best epochs can differ, so the main table uses the benchmark-best checkpoint policy consistently.",
    ]
    return "\n".join(lines) + "\n"


def build_results_markdown(overall_rows, slice_rows):
    lines = [
        "# Closure Results",
        "",
        "## Overall Results",
        "",
        "| Model | Checkpoint | Users | Recall@20 | NDCG@20 | MRR@20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['model']} | {row['checkpoint_epoch']} | {row['num_users']} | "
            f"{row['recall@20']:.4f} | {row['ndcg@20']:.4f} | {row['mrr@20']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Slice Results",
            "",
            "| Slice | Model | Users | Recall@20 | NDCG@20 | MRR@20 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in slice_rows:
        lines.append(
            f"| {row['slice']} | {row['model']} | {row['num_users']} | "
            f"{row['recall@20']:.4f} | {row['ndcg@20']:.4f} | {row['mrr@20']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Main table uses the benchmark-best checkpoint for each canonical model.",
            "- In this single-positive next-item setting, MRR@20 is numerically aligned with reciprocal-rank style evaluation and matches AP@20 behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data/metrics/closure"))
    parser.add_argument("--plots-dir", default=str(REPO_ROOT / "data/metrics/closure/plots"))
    parser.add_argument("--sparse-threshold", type=int, default=5)
    parser.add_argument("--recent-window", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    slice_rows = []

    for model_name, config_path in CANONICAL_MODELS:
        trainer = Trainer(str(config_path))
        checkpoint_dir = Path(trainer.train_params["save_path"]) / trainer.train_params["save_name"]
        best_epoch = load_best_benchmark_epoch(checkpoint_dir)
        user_rows = evaluate_test_checkpoint(
            trainer,
            checkpoint_epoch=best_epoch,
            sparse_threshold=args.sparse_threshold,
            recent_window=args.recent_window,
        )

        overall = summarize_rows(user_rows)
        overall_rows.append(
            {
                "model": model_name,
                "checkpoint_epoch": best_epoch,
                **overall,
            }
        )

        sparse_summary = summarize_rows(user_rows, predicate=lambda row: row["is_sparse_history"] == 1)
        multi_summary = summarize_rows(user_rows, predicate=lambda row: row["is_multi_interest"] == 1)
        slice_rows.extend(
            [
                {"model": model_name, "slice": "sparse-history", **sparse_summary},
                {"model": model_name, "slice": "multi-interest", **multi_summary},
            ]
        )

        user_path = output_dir / f"{trainer.train_params['save_name']}_user_metrics.json"
        user_path.write_text(json.dumps(user_rows, indent=2) + "\n")

    (output_dir / "overall_results.json").write_text(json.dumps(overall_rows, indent=2) + "\n")
    (output_dir / "slice_results.json").write_text(json.dumps(slice_rows, indent=2) + "\n")
    write_csv(output_dir / "overall_results.csv", overall_rows)
    write_csv(output_dir / "slice_results.csv", slice_rows)

    render_overall_chart(overall_rows, plots_dir / "model_comparison.png")
    render_slice_chart(slice_rows, plots_dir / "slice_comparison.png")
    (output_dir / "README_summary.md").write_text(build_readme_summary(overall_rows, slice_rows))
    (output_dir / "RESULTS.md").write_text(build_results_markdown(overall_rows, slice_rows))

    print(
        json.dumps(
            {
                "overall_results": str(output_dir / "overall_results.json"),
                "slice_results": str(output_dir / "slice_results.json"),
                "overall_chart": str(plots_dir / "model_comparison.png"),
                "slice_chart": str(plots_dir / "slice_comparison.png"),
                "readme_summary": str(output_dir / "README_summary.md"),
                "results_markdown": str(output_dir / "RESULTS.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
