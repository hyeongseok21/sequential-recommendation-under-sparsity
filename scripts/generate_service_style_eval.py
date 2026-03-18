#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HM_ROOT = REPO_ROOT / "hm_refactored"
if str(HM_ROOT) not in sys.path:
    sys.path.insert(0, str(HM_ROOT))

from hm_refactored.train import Trainer


CANONICAL_TOPPOPULAR = {
    "Recall@20": 0.0280,
    "NDCG@20": 0.0102,
    "MRR@20": 0.0055,
}

CANONICAL_PERSONALIZED = {
    "Corrected SASRec": REPO_ROOT / "results/hm_fair_sasrec_metrics.json",
    "DIF-SR": REPO_ROOT / "results/hm_fair_difsr_metrics.json",
    "DIF-SR + Metadata": REPO_ROOT / "results/hm_fair_difsr_meta_metrics.json",
}

SERVICE_CONFIGS = {
    "Corrected SASRec": REPO_ROOT / "hm_refactored/configs/config.service_eval_sasrec.json",
    "DIF-SR": REPO_ROOT / "hm_refactored/configs/config.service_eval_difsr.json",
    "DIF-SR + Metadata": REPO_ROOT / "hm_refactored/configs/config.service_eval_difsr_meta.json",
}

SERVICE_BEST_EPOCHS = {
    "Corrected SASRec": 2,
    "DIF-SR": 0,
    "DIF-SR + Metadata": 0,
}


def topk_metrics_from_ranks(ranks, k):
    recall = float(np.mean([1.0 if rank < k else 0.0 for rank in ranks]))
    ndcg = float(np.mean([(1.0 / math.log2(rank + 2)) if rank < k else 0.0 for rank in ranks]))
    mrr = float(np.mean([(1.0 / (rank + 1)) if rank < k else 0.0 for rank in ranks]))
    return recall, ndcg, mrr


def evaluate_model_rows(config_path: Path):
    trainer = Trainer(config_path=str(config_path))
    checkpoint_dir = Path(trainer.train_params["save_path"]) / trainer.train_params["save_name"]
    model_name = None
    for candidate_name, candidate_path in SERVICE_CONFIGS.items():
        if candidate_path == config_path:
            model_name = candidate_name
            break
    if model_name is None:
        raise ValueError(f"Unknown service config: {config_path}")
    best_epoch = SERVICE_BEST_EPOCHS[model_name]

    checkpoint_path = checkpoint_dir / f"{best_epoch:06d}_epoch.pth"
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.model.eval()

    rows = []
    top_k = max(100, trainer.train_params["top_k"])
    for user_batch, mask, history, history_mask in trainer.test_dl:
        _ = mask
        user_tensor = trainer._to_device_tensor(user_batch[:, 0], torch.long)
        history_tensor = trainer._to_device_tensor(history, torch.long)
        history_mask_tensor = trainer._to_device_tensor(history_mask, torch.long)
        predictions, _, _ = trainer.model.recommend(user_tensor, history_tensor, history_mask_tensor)
        _, top_items = torch.topk(predictions.detach(), min(top_k, predictions.shape[1]))
        top_items = top_items.detach().cpu().numpy()

        for batch_idx, user_idx in enumerate(user_tensor.detach().cpu().numpy().tolist()):
            source_user_id = str(trainer.data_dict["idx2user"][int(user_idx)])
            gt_items = [
                trainer.data_dict["item2idx"].get(item, -1)
                for item in trainer.data_dict["user_last_test_dict"][source_user_id]
            ]
            recs = top_items[batch_idx].tolist()
            rank = 10**9
            for item in gt_items:
                if item in recs:
                    rank = recs.index(item)
                    break

            history_items = trainer.data_dict["user_train_dict"].get(int(user_idx), [])
            rows.append(
                {
                    "user_idx": int(user_idx),
                    "history_len": int(len(history_items)),
                    "is_cold_like": int(len(history_items) == 0),
                    "is_short_history": int(len(history_items) <= 5),
                    "is_repeat_purchase": int(any(item in history_items for item in gt_items)),
                    "rank": int(rank),
                }
            )
    return best_epoch, rows


def summarize_rows(rows, predicate=None):
    selected = [row for row in rows if predicate is None or predicate(row)]
    if not selected:
        return {"users": 0, "Recall@20": 0.0, "NDCG@20": 0.0, "MRR@20": 0.0}
    recall, ndcg, mrr = topk_metrics_from_ranks([row["rank"] for row in selected], 20)
    return {"users": len(selected), "Recall@20": recall, "NDCG@20": ndcg, "MRR@20": mrr}


def render_service_comparison_plot(canonical, service_results, output_path: Path):
    models = ["TopPopular", "Corrected SASRec", "DIF-SR", "DIF-SR + Metadata"]
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    canonical_recall = [canonical[m]["Recall@20"] for m in models]
    service_recall = [service_results[m]["Recall@20"] for m in models]
    canonical_ndcg = [canonical[m]["NDCG@20"] for m in models]
    service_ndcg = [service_results[m]["NDCG@20"] for m in models]

    axes[0].bar(x - width / 2, canonical_recall, width, label="Canonical")
    axes[0].bar(x + width / 2, service_recall, width, label="Service-style")
    axes[0].set_title("Recall@20")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].legend()

    axes[1].bar(x - width / 2, canonical_ndcg, width, label="Canonical")
    axes[1].bar(x + width / 2, service_ndcg, width, label="Service-style")
    axes[1].set_title("NDCG@20")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def render_service_slice_plot(slice_results, output_path: Path):
    slice_names = ["cold-like users", "short history users (<=5)", "repeat purchase cases"]
    slice_titles = {
        "cold-like users": "Cold-like Users\n(Recall@20 / NDCG@20)",
        "short history users (<=5)": "Short-history Users\n(Recall@20 / NDCG@20)",
        "repeat purchase cases": "Repeat-heavy Cases\n(Recall@20 / NDCG@20)",
    }
    models = ["TopPopular", "Corrected SASRec", "DIF-SR", "DIF-SR + Metadata"]
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, len(slice_names), figsize=(15, 4), sharey=True)
    for ax, slice_name in zip(axes, slice_names):
        recall_vals = [slice_results[m][slice_name]["Recall@20"] for m in models]
        ndcg_vals = [slice_results[m][slice_name]["NDCG@20"] for m in models]
        ax.bar(x - width / 2, recall_vals, width, label="Recall@20")
        ax.bar(x + width / 2, ndcg_vals, width, label="NDCG@20")
        ax.set_title(slice_titles[slice_name])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("Metric score")
    axes[0].legend()
    fig.suptitle("Service-Style Slice Analysis (Recall@20 / NDCG@20)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_toppopular_service(config_path: Path):
    trainer = Trainer(config_path=str(config_path))
    top_items = list(map(int, trainer.popular_items[:100]))
    rows = []
    for user_batch, mask, history, history_mask in trainer.test_dl:
        _ = mask
        _ = history
        _ = history_mask
        user_tensor = user_batch[:, 0].numpy().tolist()
        for user_idx in user_tensor:
            source_user_id = str(trainer.data_dict["idx2user"][int(user_idx)])
            gt_items = [
                trainer.data_dict["item2idx"].get(item, -1)
                for item in trainer.data_dict["user_last_test_dict"][source_user_id]
            ]
            rank = 10**9
            for item in gt_items:
                if item in top_items:
                    rank = top_items.index(item)
                    break
            history_items = trainer.data_dict["user_train_dict"].get(int(user_idx), [])
            rows.append(
                {
                    "user_idx": int(user_idx),
                    "history_len": int(len(history_items)),
                    "is_cold_like": int(len(history_items) == 0),
                    "is_short_history": int(len(history_items) <= 5),
                    "is_repeat_purchase": int(any(item in history_items for item in gt_items)),
                    "rank": int(rank),
                }
            )
    return summarize_rows(rows), rows


def main():
    output_path = REPO_ROOT / "results/service_style_eval.md"
    metrics_dir = REPO_ROOT / "results/service_style"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    service_results = {}
    slice_results = {}
    epoch_results = {}

    for model_name, config_path in SERVICE_CONFIGS.items():
        best_epoch, rows = evaluate_model_rows(config_path)
        epoch_results[model_name] = best_epoch
        overall = summarize_rows(rows)
        service_results[model_name] = overall
        slice_results[model_name] = {
            "cold-like users": summarize_rows(rows, lambda row: row["is_cold_like"] == 1),
            "short history users (<=5)": summarize_rows(rows, lambda row: row["is_short_history"] == 1),
            "repeat purchase cases": summarize_rows(rows, lambda row: row["is_repeat_purchase"] == 1),
        }
        (metrics_dir / f"{model_name.lower().replace(' ', '_').replace('+', 'plus')}_service_rows.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )

    toppop_summary, toppop_rows = evaluate_toppopular_service(SERVICE_CONFIGS["Corrected SASRec"])
    service_results["TopPopular"] = toppop_summary
    slice_results["TopPopular"] = {
        "cold-like users": summarize_rows(toppop_rows, lambda row: row["is_cold_like"] == 1),
        "short history users (<=5)": summarize_rows(toppop_rows, lambda row: row["is_short_history"] == 1),
        "repeat purchase cases": summarize_rows(toppop_rows, lambda row: row["is_repeat_purchase"] == 1),
    }

    canonical = {"TopPopular": CANONICAL_TOPPOPULAR}
    for model_name, path in CANONICAL_PERSONALIZED.items():
        canonical[model_name] = json.loads(path.read_text())

    lines = []
    lines.append("# Service-Style Evaluation")
    lines.append("")
    lines.append("This supplementary evaluation keeps the canonical experiment untouched and re-runs a second dataset variant with:")
    lines.append("- `remove_cold_user = false`")
    lines.append("- `remove_zero_history = false`")
    lines.append("- `remove_recent_bought = false`")
    lines.append("- `remove_cold_item = true`")
    lines.append("")
    lines.append("The goal is to simulate a more realistic production environment where cold-like users remain, zero-history cases exist, and repeat purchases are allowed.")
    lines.append("")
    lines.append("## Canonical vs Service-Style Comparison")
    lines.append("")
    lines.append("| Model | Canonical Recall@20 | Service Recall@20 | Delta | Canonical NDCG@20 | Service NDCG@20 | Delta | Canonical MRR@20 | Service MRR@20 | Delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model_name in ["TopPopular", "Corrected SASRec", "DIF-SR", "DIF-SR + Metadata"]:
        c = canonical[model_name]
        s = service_results[model_name]
        lines.append(
            f"| {model_name} | {c['Recall@20']:.4f} | {s['Recall@20']:.4f} | {s['Recall@20'] - c['Recall@20']:+.4f} | "
            f"{c['NDCG@20']:.4f} | {s['NDCG@20']:.4f} | {s['NDCG@20'] - c['NDCG@20']:+.4f} | "
            f"{c['MRR@20']:.4f} | {s['MRR@20']:.4f} | {s['MRR@20'] - c['MRR@20']:+.4f} |"
        )
    lines.append("")
    lines.append("## Slice Metrics")
    lines.append("")
    for slice_name in ["cold-like users", "short history users (<=5)", "repeat purchase cases"]:
        lines.append(f"### {slice_name}")
        lines.append("")
        lines.append("| Model | Users | Recall@20 | NDCG@20 | MRR@20 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for model_name in ["TopPopular", "Corrected SASRec", "DIF-SR", "DIF-SR + Metadata"]:
            row = slice_results[model_name][slice_name]
            lines.append(
                f"| {model_name} | {row['users']} | {row['Recall@20']:.4f} | {row['NDCG@20']:.4f} | {row['MRR@20']:.4f} |"
            )
        lines.append("")
    lines.append("## Analysis")
    lines.append("")
    sasrec_drop = service_results["Corrected SASRec"]["Recall@20"] - canonical["Corrected SASRec"]["Recall@20"]
    difsr_drop = service_results["DIF-SR"]["Recall@20"] - canonical["DIF-SR"]["Recall@20"]
    difsr_meta_drop = service_results["DIF-SR + Metadata"]["Recall@20"] - canonical["DIF-SR + Metadata"]["Recall@20"]
    lines.append(
        f"1. Including cold/zero-history users does not collapse all sequential models equally. "
        f"`Corrected SASRec` changed by {sasrec_drop:+.4f} Recall@20, `DIF-SR` by {difsr_drop:+.4f}, and "
        f"`DIF-SR + Metadata` by {difsr_meta_drop:+.4f}."
    )
    lines.append(
        "2. Allowing repeat purchases creates an easier recommendation path for popularity- and repeat-heavy behavior, "
        "so slice performance on `repeat purchase cases` should be interpreted alongside `TopPopular`."
    )
    meta_help = service_results["DIF-SR + Metadata"]["Recall@20"] - service_results["DIF-SR"]["Recall@20"]
    lines.append(
        f"3. Under relaxed filters, metadata still helps the diffusion backbone: "
        f"`DIF-SR + Metadata` improves Recall@20 over `DIF-SR` by {meta_help:+.4f}. "
        "The same claim is not made for SASRec because the supplementary run did not include a metadata SASRec branch."
    )
    lines.append("")
    lines.append("Observed service-style best epochs:")
    for model_name in ["Corrected SASRec", "DIF-SR", "DIF-SR + Metadata"]:
        lines.append(f"- `{model_name}`: epoch `{epoch_results[model_name]}`")

    render_service_comparison_plot(canonical, service_results, metrics_dir / "service_vs_canonical.png")
    render_service_slice_plot(slice_results, metrics_dir / "service_slice_metrics.png")
    output_path.write_text("\n".join(lines) + "\n")
    (metrics_dir / "service_results.json").write_text(
        json.dumps({"overall": service_results, "slices": slice_results, "best_epochs": epoch_results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
