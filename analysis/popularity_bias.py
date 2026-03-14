from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import DATA_DIR, PLOTS_DIR, write_csv, write_json


BUCKET_ORDER = ["head", "mid", "tail"]


def compute_exposure_frame(model_name: str, frame: pd.DataFrame, popularity_bucket: dict):
    rows = []
    for row in frame.to_dict(orient="records"):
        top20 = row["top20_items"]
        buckets = [popularity_bucket.get(int(item_id), "tail") for item_id in top20]
        counts = pd.Series(buckets).value_counts()
        rows.append(
            {
                "model": model_name,
                "source_user_id": row["source_user_id"],
                "history_bucket": row["history_bucket"],
                "category_entropy_bucket": row["category_entropy_bucket"],
                "head_share": float(counts.get("head", 0) / len(top20)),
                "mid_share": float(counts.get("mid", 0) / len(top20)),
                "tail_share": float(counts.get("tail", 0) / len(top20)),
            }
        )
    return pd.DataFrame(rows)


def render_popularity_chart(summary_frame: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = None
    for bucket in BUCKET_ORDER:
        values = summary_frame[f"{bucket}_share"]
        ax.bar(summary_frame["model"], values, bottom=bottom, label=bucket.title())
        bottom = values if bottom is None else bottom + values
    ax.set_ylabel("Exposure Share")
    ax.set_title("Popularity Exposure in Top-20 Recommendations")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_popularity_bias_analysis(records_by_model, popularity_bucket):
    per_user_frames = []
    summary_rows = []

    for model_name, frame in records_by_model.items():
        exposure_frame = compute_exposure_frame(model_name, frame, popularity_bucket)
        per_user_frames.append(exposure_frame)
        summary_rows.append(
            {
                "model": model_name,
                "head_share": float(exposure_frame["head_share"].mean()),
                "mid_share": float(exposure_frame["mid_share"].mean()),
                "tail_share": float(exposure_frame["tail_share"].mean()),
            }
        )

    per_user_frame = pd.concat(per_user_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summary_rows)

    write_csv(DATA_DIR / "popularity_exposure_per_user.csv", per_user_frame)
    write_json(DATA_DIR / "popularity_exposure_summary.json", summary_rows)
    render_popularity_chart(summary_frame, PLOTS_DIR / "popularity_exposure.png")

    return {
        "per_user": per_user_frame,
        "summary": summary_frame,
    }
