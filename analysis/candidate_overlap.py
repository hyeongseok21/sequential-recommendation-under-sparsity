from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import DATA_DIR, PLOTS_DIR, summarize_distribution, write_csv, write_json


PAIR_ORDER = [
    ("SASRec", "DIF-SR"),
    ("SASRec", "DIF-SR + metadata"),
    ("DIF-SR", "DIF-SR + metadata"),
]


def compute_pair_overlap(left: pd.DataFrame, right: pd.DataFrame, label_left: str, label_right: str):
    merged = left[["source_user_id", "category_entropy_bucket", "top100_items"]].merge(
        right[["source_user_id", "top100_items"]],
        on="source_user_id",
        suffixes=("_left", "_right"),
    )
    merged["pair"] = f"{label_left} vs {label_right}"
    merged["overlap"] = merged.apply(
        lambda row: len(set(row["top100_items_left"]).intersection(row["top100_items_right"])) / 100.0,
        axis=1,
    )
    return merged[["source_user_id", "category_entropy_bucket", "pair", "overlap"]]


def render_overlap_histograms(overlap_frame: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, len(PAIR_ORDER), figsize=(15, 4), sharey=True)
    for ax, (left, right) in zip(axes, PAIR_ORDER):
        pair_name = f"{left} vs {right}"
        values = overlap_frame.loc[overlap_frame["pair"] == pair_name, "overlap"]
        ax.hist(values, bins=20, color="#d66c5f", alpha=0.8)
        ax.set_title(pair_name)
        ax.set_xlabel("Top-100 overlap")
    axes[0].set_ylabel("User count")
    fig.suptitle("Candidate Frontier Overlap")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_candidate_overlap_analysis(records_by_model):
    overlap_frames = []
    for left_name, right_name in PAIR_ORDER:
        overlap_frames.append(
            compute_pair_overlap(records_by_model[left_name], records_by_model[right_name], left_name, right_name)
        )

    overlap_frame = pd.concat(overlap_frames, ignore_index=True)
    pair_summary = []
    for pair_name, group in overlap_frame.groupby("pair"):
        stats = summarize_distribution(group["overlap"].tolist())
        pair_summary.append({"pair": pair_name, **stats})

    slice_summary = []
    for (pair_name, entropy_bucket), group in overlap_frame.groupby(["pair", "category_entropy_bucket"]):
        stats = summarize_distribution(group["overlap"].tolist())
        slice_summary.append({"pair": pair_name, "entropy_bucket": entropy_bucket, **stats})

    write_csv(DATA_DIR / "candidate_overlap_distribution.csv", overlap_frame)
    write_json(DATA_DIR / "candidate_overlap_summary.json", pair_summary)
    write_json(DATA_DIR / "candidate_overlap_entropy_summary.json", slice_summary)
    render_overlap_histograms(overlap_frame, PLOTS_DIR / "candidate_overlap_histogram.png")

    return {
        "distribution": overlap_frame,
        "summary": pd.DataFrame(pair_summary),
        "entropy_summary": pd.DataFrame(slice_summary),
    }
