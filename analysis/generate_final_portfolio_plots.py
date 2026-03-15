#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "plots"

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


def plot_service_slices(service_slices, output_path: Path):
    models = ["TopPopular", "Corrected SASRec", "DIF-SR", "DIF-SR + Metadata"]
    slice_names = ["cold-like users", "short history users (<=5)", "repeat purchase cases"]
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, slice_name in zip(axes, slice_names):
        recall_values = [service_slices[m][slice_name]["Recall@20"] for m in models]
        ndcg_values = [service_slices[m][slice_name]["NDCG@20"] for m in models]
        ax.bar(x - width / 2, recall_values, width, label="Recall@20", color="#2f6bff")
        ax.bar(x + width / 2, ndcg_values, width, label="NDCG@20", color="#ff7f32")
        ax.set_title(slice_name)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
    axes[0].legend()
    fig.suptitle("Service-Style Slice Analysis")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    canonical_results = load_canonical_results()
    service_results, service_slices = load_service_results()
    plot_model_comparison(canonical_results, PLOTS_DIR / "final_model_comparison.png")
    plot_canonical_vs_service(canonical_results, service_results, PLOTS_DIR / "canonical_vs_service_comparison.png")
    plot_service_slices(service_slices, PLOTS_DIR / "final_slice_analysis.png")


if __name__ == "__main__":
    main()
