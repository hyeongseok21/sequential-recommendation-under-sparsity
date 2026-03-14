from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import DATA_DIR, PLOTS_DIR, write_csv, write_json


def _slice_summary_row(model_name, slice_name, metric_frame, diversity_frame, popularity_frame):
    return {
        "model": model_name,
        "slice": slice_name,
        "num_users": int(len(metric_frame)),
        "recall@20": float(metric_frame["recall@20"].mean()) if len(metric_frame) else 0.0,
        "ndcg@20": float(metric_frame["ndcg@20"].mean()) if len(metric_frame) else 0.0,
        "mean_unique_category_count": float(diversity_frame["unique_category_count"].mean()) if len(diversity_frame) else 0.0,
        "mean_category_entropy": float(diversity_frame["category_entropy"].mean()) if len(diversity_frame) else 0.0,
        "tail_share": float(popularity_frame["tail_share"].mean()) if len(popularity_frame) else 0.0,
    }


def build_slice_summaries(records_by_model, diversity_per_user, popularity_per_user, overlap_distribution):
    cold_rows = []
    entropy_rows = []

    overlap_slice_summary = []
    for (pair_name, entropy_bucket), group in overlap_distribution.groupby(["pair", "category_entropy_bucket"]):
        overlap_slice_summary.append(
            {
                "pair": pair_name,
                "entropy_bucket": entropy_bucket,
                "mean_overlap": float(group["overlap"].mean()),
                "median_overlap": float(group["overlap"].median()),
            }
        )

    for model_name, frame in records_by_model.items():
        diversity_model = diversity_per_user.loc[diversity_per_user["model"] == model_name]
        popularity_model = popularity_per_user.loc[popularity_per_user["model"] == model_name]

        cold_metric = frame.loc[frame["history_bucket"] == "1-5"]
        cold_div = diversity_model.loc[diversity_model["history_bucket"] == "1-5"]
        cold_pop = popularity_model.loc[popularity_model["history_bucket"] == "1-5"]
        cold_rows.append(_slice_summary_row(model_name, "short_history", cold_metric, cold_div, cold_pop))

        for entropy_bucket in ["low", "mid", "high"]:
            metric_bucket = frame.loc[frame["category_entropy_bucket"] == entropy_bucket]
            div_bucket = diversity_model.loc[diversity_model["category_entropy_bucket"] == entropy_bucket]
            pop_bucket = popularity_model.loc[popularity_model["category_entropy_bucket"] == entropy_bucket]
            entropy_rows.append(
                _slice_summary_row(
                    model_name,
                    f"{entropy_bucket}_entropy",
                    metric_bucket,
                    div_bucket,
                    pop_bucket,
                )
            )

    return pd.DataFrame(cold_rows), pd.DataFrame(entropy_rows), pd.DataFrame(overlap_slice_summary)


def render_slice_chart(slice_frame: pd.DataFrame, output_path: Path):
    pivot = slice_frame.pivot(index="model", columns="slice", values="ndcg@20")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("NDCG@20")
    ax.set_title("Slice Analysis")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_slice_analysis(records_by_model, diversity_result, popularity_result, overlap_result):
    cold_frame, entropy_frame, overlap_entropy = build_slice_summaries(
        records_by_model,
        diversity_result["per_user"],
        popularity_result["per_user"],
        overlap_result["distribution"],
    )
    slice_frame = pd.concat([cold_frame, entropy_frame], ignore_index=True)

    write_csv(DATA_DIR / "slice_study.csv", slice_frame)
    write_json(DATA_DIR / "slice_study.json", slice_frame.to_dict(orient="records"))
    write_json(DATA_DIR / "entropy_overlap_summary.json", overlap_entropy.to_dict(orient="records"))
    render_slice_chart(slice_frame, PLOTS_DIR / "slice_analysis.png")

    return {
        "slice_summary": slice_frame,
        "entropy_overlap": overlap_entropy,
    }
