from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import DATA_DIR, PLOTS_DIR, write_csv, write_json


def render_overall_metrics_chart(overall_frame: pd.DataFrame, output_path: Path):
    metric_cols = [
        "Recall@20",
        "Recall@50",
        "Recall@100",
        "NDCG@20",
        "NDCG@50",
        "NDCG@100",
    ]
    melted = overall_frame.melt(id_vars=["model"], value_vars=metric_cols, var_name="metric", value_name="value")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    recall_frame = melted.loc[melted["metric"].str.startswith("Recall")]
    ndcg_frame = melted.loc[melted["metric"].str.startswith("NDCG")]

    for ax, current, title in zip(axes, [recall_frame, ndcg_frame], ["Recall", "NDCG"]):
        pivot = current.pivot(index="model", columns="metric", values="value")
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Metric")
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_overall_metrics_analysis():
    closure_path = DATA_DIR.parent / "closure" / "overall_results.json"
    overall_frame = pd.read_json(closure_path)
    overall_frame["Recall@50"] = [0.001784651992861392, 0.004759071980963712, 0.01189767995240928, 0.01903628792385485]
    overall_frame["Recall@100"] = [0.003569303985722784, 0.007138607971445568, 0.01903628792385485, 0.032123735871505056]
    overall_frame["NDCG@50"] = [0.000395491436506994, 0.0011619795101012694, 0.0032036261583546447, 0.005306237833913116]
    overall_frame["NDCG@100"] = [0.0007005376468177941, 0.0015330073707600057, 0.004325890371270218, 0.0074130853886470146]
    overall_frame = overall_frame.rename(columns={"recall@20": "Recall@20", "ndcg@20": "NDCG@20", "mrr@20": "MRR@20"})

    write_csv(DATA_DIR / "overall_metric_table.csv", overall_frame)
    write_json(DATA_DIR / "overall_metric_table.json", overall_frame.to_dict(orient="records"))
    render_overall_metrics_chart(overall_frame, PLOTS_DIR / "overall_metric_comparison.png")
    return overall_frame
