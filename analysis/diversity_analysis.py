from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import DATA_DIR, PLOTS_DIR, shannon_entropy, write_csv, write_json


def compute_diversity_frame(model_name: str, frame: pd.DataFrame, item_meta: pd.DataFrame):
    total_categories = int(item_meta["department_no"].nunique())
    per_user_rows = []
    all_categories = set()

    for row in frame.to_dict(orient="records"):
        categories = row["top20_departments"]
        all_categories.update(categories)
        unique_count = len(set(categories))
        per_user_rows.append(
            {
                "model": model_name,
                "source_user_id": row["source_user_id"],
                "history_bucket": row["history_bucket"],
                "category_entropy_bucket": row["category_entropy_bucket"],
                "unique_category_count": unique_count,
                "category_entropy": shannon_entropy(categories),
            }
        )

    per_user = pd.DataFrame(per_user_rows)
    summary = {
        "model": model_name,
        "mean_unique_category_count": float(per_user["unique_category_count"].mean()),
        "mean_category_entropy": float(per_user["category_entropy"].mean()),
        "global_category_coverage_rate": float(len(all_categories) / total_categories),
    }
    return per_user, summary


def render_diversity_chart(summary_frame: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(summary_frame["model"], summary_frame["mean_unique_category_count"], color="#4e79a7")
    axes[0].set_title("Mean Unique Categories in Top-20")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(summary_frame["model"], summary_frame["mean_category_entropy"], color="#59a14f")
    axes[1].set_title("Mean Category Entropy in Top-20")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_diversity_analysis(records_by_model, item_meta: pd.DataFrame):
    per_user_frames = []
    summary_rows = []

    for model_name, frame in records_by_model.items():
        per_user, summary = compute_diversity_frame(model_name, frame, item_meta)
        per_user_frames.append(per_user)
        summary_rows.append(summary)

    per_user_frame = pd.concat(per_user_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summary_rows)

    write_csv(DATA_DIR / "diversity_per_user.csv", per_user_frame)
    write_json(DATA_DIR / "diversity_summary.json", summary_rows)
    render_diversity_chart(summary_frame, PLOTS_DIR / "diversity_comparison.png")

    return {
        "per_user": per_user_frame,
        "summary": summary_frame,
    }
