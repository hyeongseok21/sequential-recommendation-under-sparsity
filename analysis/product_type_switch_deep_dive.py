#!/usr/bin/env python3
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
HM_ROOT = REPO_ROOT / "hm_refactored"
if str(HM_ROOT) not in sys.path:
    sys.path.insert(0, str(HM_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.product_type_switch_analysis import build_product_type_descriptors, load_model_rows


PREP_PATH = REPO_ROOT / "datasets/hm/local/prep/hm_local_meta_userhash32_week27.pkl"
TOPK_DIR = REPO_ROOT / "data/metrics/research_study/topk"
OUTPUT_DIR = REPO_ROOT / "data/metrics/research_study"
REPORT_PATH = REPO_ROOT / "reports/product_type_switch_deep_dive.md"

TOPK_FILES = {
    "Corrected SASRec": TOPK_DIR / "hm_closure_sasrec_top100.json",
    "DIF-SR": TOPK_DIR / "hm_closure_difsr_top100.json",
    "DIF-SR + Metadata": TOPK_DIR / "hm_closure_difsr_meta_top100.json",
}


def build_recent5_behavior_descriptors():
    with PREP_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    train_df = payload["train_df"][["user_id", "timestamp", "product_type_no", "price", "interval"]].copy()
    train_df = train_df.sort_values(["user_id", "timestamp"])
    idx2user = payload["idx2user"]

    rows = []
    for user_id, group in train_df.groupby("user_id", sort=False):
        recent = group.tail(5).copy()
        product_types = recent["product_type_no"].tolist()
        switches = sum(1 for left, right in zip(product_types[:-1], product_types[1:]) if left != right)
        rows.append(
            {
                "source_user_id": str(idx2user[int(user_id)]),
                "history_len_true": int(len(group)),
                "switches_recent5": int(switches),
                "unique_product_types_recent5": int(len(set(product_types))),
                "mean_price_recent5": float(recent["price"].mean()) if len(recent) else 0.0,
                "price_std_recent5": float(recent["price"].std(ddof=0)) if len(recent) else 0.0,
                "mean_interval_recent5": float(recent["interval"].mean()) if len(recent) else 0.0,
                "max_interval_recent5": float(recent["interval"].max()) if len(recent) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_models(frame: pd.DataFrame, slice_name: str):
    grouped = (
        frame.groupby("model")
        .agg(
            users=("source_user_id", "count"),
            recall20=("recall@20", "mean"),
            ndcg20=("ndcg@20", "mean"),
            mrr20=("mrr@20", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "slice", slice_name)
    return grouped


def compute_overlap(slice_user_ids, left_name, right_name):
    left = pd.DataFrame(json.loads(TOPK_FILES[left_name].read_text()))
    right = pd.DataFrame(json.loads(TOPK_FILES[right_name].read_text()))
    left = left[left["source_user_id"].isin(slice_user_ids)][["source_user_id", "top100_items"]]
    right = right[right["source_user_id"].isin(slice_user_ids)][["source_user_id", "top100_items"]]
    merged = left.merge(right, on="source_user_id", suffixes=("_left", "_right"))
    if merged.empty:
        return {"pair": f"{left_name} vs {right_name}", "users": 0, "mean_overlap": 0.0, "median_overlap": 0.0}
    overlaps = merged.apply(
        lambda row: len(set(row["top100_items_left"]).intersection(row["top100_items_right"])) / 100.0,
        axis=1,
    )
    return {
        "pair": f"{left_name} vs {right_name}",
        "users": int(len(overlaps)),
        "mean_overlap": float(overlaps.mean()),
        "median_overlap": float(overlaps.median()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    base_descriptors = build_product_type_descriptors()
    behavior_frame = build_recent5_behavior_descriptors()
    metric_frame = load_model_rows(base_descriptors)
    frame = metric_frame.merge(behavior_frame, on="source_user_id", how="left")

    with PREP_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    non_repeat_users = set(
        payload["unique_last_test_df"].loc[~payload["unique_last_test_df"]["recent_bought"], "user_id"].astype(str)
    )
    frame["non_repeat_target"] = frame["source_user_id"].isin(non_repeat_users)

    main_mask = (frame["history_len"] <= 8) & (frame["product_type_switches_recent5"].between(1, 2))
    main_slice = frame.loc[main_mask].copy()

    price_median = float(main_slice["price_std_recent5"].median()) if not main_slice.empty else 0.0
    interval_median = float(main_slice["mean_interval_recent5"].median()) if not main_slice.empty else 0.0

    slice_defs = {
        "history<=8 & 1 switch": (frame["history_len"] <= 8) & (frame["product_type_switches_recent5"] == 1),
        "history<=8 & 2 switches": (frame["history_len"] <= 8) & (frame["product_type_switches_recent5"] == 2),
        "history<=8 & 1-2 switches": main_mask,
        "history<=8 & 1-2 switches & non-repeat": main_mask & frame["non_repeat_target"],
        "history<=8 & 1-2 switches & low price variance": main_mask & (frame["price_std_recent5"] <= price_median),
        "history<=8 & 1-2 switches & high price variance": main_mask & (frame["price_std_recent5"] > price_median),
        "history<=8 & 1-2 switches & dense sequence": main_mask & (frame["mean_interval_recent5"] <= interval_median),
        "history<=8 & 1-2 switches & sparse sequence": main_mask & (frame["mean_interval_recent5"] > interval_median),
    }

    summaries = []
    for name, mask in slice_defs.items():
        summaries.append(summarize_models(frame.loc[mask].copy(), name))
    summary_frame = pd.concat(summaries, ignore_index=True)

    overlap_rows = []
    main_slice_user_ids = set(frame.loc[slice_defs["history<=8 & 1-2 switches"], "source_user_id"])
    for pair in [
        ("Corrected SASRec", "DIF-SR + Metadata"),
        ("DIF-SR", "DIF-SR + Metadata"),
    ]:
        overlap_rows.append(compute_overlap(main_slice_user_ids, *pair))
    overlap_frame = pd.DataFrame(overlap_rows)

    summary_csv = OUTPUT_DIR / "product_type_switch_deep_dive_summary.csv"
    overlap_json = OUTPUT_DIR / "product_type_switch_deep_dive_overlap.json"
    summary_csv.write_text(summary_frame.to_csv(index=False))
    overlap_json.write_text(json.dumps(overlap_rows, indent=2) + "\n")

    lines = [
        "# Product-Type Switch Deep Dive",
        "",
        "Main slice: `history<=8 & 1-2 product_type switches`",
        "",
        f"- median price std within main slice: `{price_median:.6f}`",
        f"- median mean interval within main slice: `{interval_median:.2f}`",
        "",
        "| Slice | Model | Users | Recall@20 | NDCG@20 | MRR@20 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_frame.to_dict(orient="records"):
        lines.append(
            f"| {row['slice']} | {row['model']} | {int(row['users'])} | "
            f"{row['recall20']:.4f} | {row['ndcg20']:.4f} | {row['mrr20']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Overlap within Main Slice",
            "",
            "| Pair | Users | Mean Overlap | Median Overlap |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in overlap_rows:
        lines.append(
            f"| {row['pair']} | {row['users']} | {row['mean_overlap']:.4f} | {row['median_overlap']:.4f} |"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n")

    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "overlap_json": str(overlap_json),
                "report": str(REPORT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
