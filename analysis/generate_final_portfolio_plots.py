#!/usr/bin/env python3
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "plots"
HM_ROOT = REPO_ROOT / "hm_refactored"
if str(HM_ROOT) not in sys.path:
    sys.path.insert(0, str(HM_ROOT))

from train import Trainer

TOPPOPULAR = {
    "Recall@20": 0.0280,
    "NDCG@20": 0.0102,
    "MRR@20": 0.0055,
}

CANONICAL_FILES = {
    "Corrected SASRec": REPO_ROOT / "results/hm_fair_sasrec_metrics.json",
    "DIF-SR": REPO_ROOT / "results/hm_fair_difsr_metrics.json",
    "DIF-SR + Metadata": REPO_ROOT / "results/hm_fair_difsr_meta_metrics.json",
}


def load_canonical_results():
    results = {"TopPopular": TOPPOPULAR}
    for model_name, path in CANONICAL_FILES.items():
        results[model_name] = json.loads(path.read_text())
    return results


def load_service_results():
    payload = json.loads((REPO_ROOT / "results/service_style/service_results.json").read_text())
    return payload["overall"], payload["slices"]


HIGH_ENTROPY_USER_FILES = {
    "Corrected SASRec": REPO_ROOT / "data/metrics/closure/hm_sanity_sasrec_fixed_user_metrics.json",
    "DIF-SR": REPO_ROOT / "data/metrics/closure/hm_closure_difsr_user_metrics.json",
    "DIF-SR + Metadata": REPO_ROOT / "data/metrics/closure/hm_closure_difsr_meta_user_metrics.json",
}


def summarize_high_entropy_user_metrics(path: Path):
    rows = json.loads(path.read_text())
    selected = [row for row in rows if row["category_entropy_bucket"] == "high"]
    return {
        "Recall@20": float(np.mean([row["recall@20"] for row in selected])),
        "NDCG@20": float(np.mean([row["ndcg@20"] for row in selected])),
        "source_user_ids": {row["source_user_id"] for row in selected},
    }


def compute_toppopular_high_entropy(source_user_ids):
    trainer = Trainer(config_path=str(REPO_ROOT / "hm_refactored/configs/config.closure_sasrec.json"))
    popular_items = list(map(int, trainer.popular_items[:100]))
    recalls = []
    ndcgs = []

    for user_batch, mask, history, history_mask in trainer.test_dl:
        _ = mask
        _ = history
        _ = history_mask
        for user_idx in user_batch[:, 0].numpy().tolist():
            source_user_id = str(trainer.data_dict["idx2user"][int(user_idx)])
            if source_user_id not in source_user_ids:
                continue
            gt_items = [
                trainer.data_dict["item2idx"].get(item, -1)
                for item in trainer.data_dict["user_last_test_dict"][source_user_id]
            ]
            rank = 10**9
            for item in gt_items:
                if item in popular_items:
                    rank = popular_items.index(item)
                    break
            recalls.append(1.0 if rank < 20 else 0.0)
            ndcgs.append((1.0 / math.log2(rank + 2)) if rank < 20 else 0.0)

    return {
        "Recall@20": float(np.mean(recalls)),
        "NDCG@20": float(np.mean(ndcgs)),
    }


def load_high_entropy_slice_results():
    results = {}
    source_user_ids = None
    for model_name, path in HIGH_ENTROPY_USER_FILES.items():
        summary = summarize_high_entropy_user_metrics(path)
        results[model_name] = {
            "Recall@20": summary["Recall@20"],
            "NDCG@20": summary["NDCG@20"],
        }
        if source_user_ids is None:
            source_user_ids = summary["source_user_ids"]
    results["TopPopular"] = compute_toppopular_high_entropy(source_user_ids)
    return results


def plot_model_comparison(canonical_results, output_path: Path):
    models = list(canonical_results.keys())
    recall_values = [canonical_results[m]["Recall@20"] for m in models]
    ndcg_values = [canonical_results[m]["NDCG@20"] for m in models]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, recall_values, width, label="Recall@20", color="#2f6bff")
    ax.bar(x + width / 2, ndcg_values, width, label="NDCG@20", color="#ff7f32")
    ax.set_title("Canonical Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.set_ylabel("Metric")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_canonical_vs_service(canonical_results, service_results, output_path: Path):
    models = list(canonical_results.keys())
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    canonical_recall = [canonical_results[m]["Recall@20"] for m in models]
    service_recall = [service_results[m]["Recall@20"] for m in models]
    canonical_ndcg = [canonical_results[m]["NDCG@20"] for m in models]
    service_ndcg = [service_results[m]["NDCG@20"] for m in models]

    axes[0].bar(x - width / 2, canonical_recall, width, label="Canonical", color="#2f6bff")
    axes[0].bar(x + width / 2, service_recall, width, label="Service-style", color="#7db7ff")
    axes[0].set_title("Recall@20")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].legend()

    axes[1].bar(x - width / 2, canonical_ndcg, width, label="Canonical", color="#ff7f32")
    axes[1].bar(x + width / 2, service_ndcg, width, label="Service-style", color="#ffb36b")
    axes[1].set_title("NDCG@20")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha="right")

    fig.suptitle("Canonical vs Service-Style Evaluation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metadata_and_head_selection(output_path: Path):
    metadata_labels = [
        "all-features\nbaseline",
        "product_type\nx1.5",
        "product_type\nx1.7",
        "product_type\n+ dept x0.5",
        "product_type\n+ garment x1.5",
    ]
    metadata_values = [0.0137, 0.0139, 0.0129, 0.0108, 0.0122]
    metadata_colors = ["#cfd6e6", "#2f6bff", "#9bbdff", "#ffb36b", "#ffcf9f"]
    metadata_notes = ["baseline", "PASS", "FAIL", "FAIL", "FAIL"]

    head_labels = ["h1", "h2", "h4"]
    head_values = [0.0081, 0.0075, 0.0086]
    head_colors = ["#cfd6e6", "#9bbdff", "#2f6bff"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    x0 = np.arange(len(metadata_labels))
    bars0 = axes[0].bar(x0, metadata_values, color=metadata_colors, edgecolor="#33415c", linewidth=0.8)
    axes[0].set_title("Metadata Selection (Full Validation B_NDCG)")
    axes[0].set_xticks(x0)
    axes[0].set_xticklabels(metadata_labels)
    axes[0].set_ylabel("B_NDCG")
    axes[0].set_ylim(0.0, 0.016)
    for bar, note in zip(bars0, metadata_notes):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.00025,
            note,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#33415c",
        )

    x1 = np.arange(len(head_labels))
    bars1 = axes[1].bar(x1, head_values, color=head_colors, edgecolor="#33415c", linewidth=0.8)
    axes[1].set_title("Head Comparison (Fast-Scout B_NDCG)")
    axes[1].set_xticks(x1)
    axes[1].set_xticklabels(head_labels)
    axes[1].set_ylabel("B_NDCG")
    axes[1].set_ylim(0.0, 0.0105)
    for bar in bars1:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0002,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#33415c",
        )

    axes[1].text(
        1.0,
        0.001,
        "h4 looked best in fast-scout,\nbut full validation did not beat\nthe product_type x1.5 champion.",
        fontsize=8,
        color="#33415c",
        bbox={"facecolor": "#f3f6fb", "edgecolor": "#cfd6e6", "boxstyle": "round,pad=0.35"},
    )

    fig.suptitle("Metadata and Head Selection")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_service_slices(service_slices, entropy_slices, output_path: Path):
    models = ["TopPopular", "Corrected SASRec", "DIF-SR", "DIF-SR + Metadata"]
    slice_names = ["cold-like users", "short history users (<=5)", "repeat purchase cases"]
    slice_titles = {
        "cold-like users": "Cold-like Users",
        "short history users (<=5)": "Short-history Users",
        "repeat purchase cases": "Repeat-heavy Cases",
    }
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), sharey=False)
    for ax, slice_name in zip(axes, slice_names):
        recall_values = [service_slices[m][slice_name]["Recall@20"] for m in models]
        ndcg_values = [service_slices[m][slice_name]["NDCG@20"] for m in models]
        ax.bar(x - width / 2, recall_values, width, label="Recall@20", color="#2f6bff")
        ax.bar(x + width / 2, ndcg_values, width, label="NDCG@20", color="#ff7f32")
        ax.set_title(slice_titles[slice_name])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("Metric Score")
        if slice_name in {"cold-like users", "short history users (<=5)"}:
            ax.set_ylim(0.0, 0.10)

    entropy_models = ["TopPopular", "Corrected SASRec", "DIF-SR", "DIF-SR + Metadata"]
    entropy_x = np.arange(len(entropy_models))
    entropy_ax = axes[-1]
    recall_values = [entropy_slices[m]["Recall@20"] for m in entropy_models]
    ndcg_values = [entropy_slices[m]["NDCG@20"] for m in entropy_models]
    entropy_ax.bar(entropy_x - width / 2, recall_values, width, label="Recall@20", color="#2f6bff")
    entropy_ax.bar(entropy_x + width / 2, ndcg_values, width, label="NDCG@20", color="#ff7f32")
    entropy_ax.set_title("High-entropy Users")
    entropy_ax.set_xticks(entropy_x)
    entropy_ax.set_xticklabels(entropy_models, rotation=15, ha="right")
    entropy_ax.set_ylabel("Metric Score")
    axes[0].legend()
    fig.suptitle("Slice Analysis Summary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    canonical_results = load_canonical_results()
    service_results, service_slices = load_service_results()
    entropy_slices = load_high_entropy_slice_results()
    plot_model_comparison(canonical_results, PLOTS_DIR / "final_model_comparison.png")
    plot_canonical_vs_service(canonical_results, service_results, PLOTS_DIR / "canonical_vs_service_comparison.png")
    plot_metadata_and_head_selection(PLOTS_DIR / "metadata_head_selection.png")
    plot_service_slices(service_slices, entropy_slices, PLOTS_DIR / "final_slice_analysis.png")


if __name__ == "__main__":
    main()
