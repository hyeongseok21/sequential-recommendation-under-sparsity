#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
HM_ROOT = REPO_ROOT / "hm_refactored"
if str(HM_ROOT) not in sys.path:
    sys.path.insert(0, str(HM_ROOT))

from train import Trainer


PREP_PATH = REPO_ROOT / "datasets/hm/local/prep/hm_local_meta_userhash32_week27.pkl"
OUTPUT_DIR = REPO_ROOT / "data/metrics/research_study"
REPORT_PATH = REPO_ROOT / "reports/product_type_switch_analysis.md"

USER_METRIC_FILES = {
    "Corrected SASRec": REPO_ROOT / "data/metrics/closure/hm_sanity_sasrec_fixed_user_metrics.json",
    "DIF-SR": REPO_ROOT / "data/metrics/closure/hm_closure_difsr_user_metrics.json",
    "DIF-SR + Metadata": REPO_ROOT / "data/metrics/closure/hm_closure_difsr_meta_user_metrics.json",
}


def switch_bucket_recent5(switches: int) -> str:
    if switches == 0:
        return "0 switches"
    if switches <= 2:
        return "1-2 switches"
    return "3+ switches"


def build_product_type_descriptors():
    with PREP_PATH.open("rb") as handle:
        payload = pd.read_pickle(handle)
    train_df = payload["train_df"][["user_id", "timestamp", "product_type_no"]].copy()
    train_df = train_df.sort_values(["user_id", "timestamp"])
    idx2user = payload["idx2user"]

    descriptors = {}
    for user_id, group in train_df.groupby("user_id", sort=False):
        seq = group["product_type_no"].tolist()
        recent5 = seq[-5:]
        switches = sum(1 for left, right in zip(recent5[:-1], recent5[1:]) if left != right)
        source_user_id = str(idx2user[int(user_id)])
        descriptors[source_user_id] = {
            "product_type_recent5_len": len(recent5),
            "product_type_switches_recent5": int(switches),
            "product_type_switch_bucket_recent5": switch_bucket_recent5(int(switches)),
            "product_type_unique_recent5": int(len(set(recent5))),
            "history_len_true": int(len(seq)),
        }
    return descriptors


def compute_toppopular_rows(descriptors):
    trainer = Trainer(config_path=str(REPO_ROOT / "hm_refactored/configs/config.closure_sasrec.json"))
    popular_items = list(map(int, trainer.popular_items[:100]))
    rows = []
    for user_batch, mask, history, history_mask in trainer.test_dl:
        _ = mask
        _ = history
        _ = history_mask
        for user_idx in user_batch[:, 0].numpy().tolist():
            source_user_id = str(trainer.data_dict["idx2user"][int(user_idx)])
            descriptor = descriptors.get(source_user_id)
            if descriptor is None:
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
            rows.append(
                {
                    "user_idx": int(user_idx),
                    "source_user_id": source_user_id,
                    "recall@20": float(rank < 20),
                    "ndcg@20": float((1.0 / math.log2(rank + 2)) if rank < 20 else 0.0),
                    "mrr@20": float((1.0 / (rank + 1)) if rank < 20 else 0.0),
                    "history_len": descriptor["history_len_true"],
                    **descriptor,
                    "model": "TopPopular",
                }
            )
    return pd.DataFrame(rows)


def load_model_rows(descriptors):
    frames = []
    for model_name, path in USER_METRIC_FILES.items():
        frame = pd.DataFrame(json.loads(path.read_text()))
        for col in [
            "product_type_recent5_len",
            "product_type_switches_recent5",
            "product_type_switch_bucket_recent5",
            "product_type_unique_recent5",
            "history_len_true",
        ]:
            frame[col] = frame["source_user_id"].astype(str).map(lambda uid, c=col: descriptors[uid][c])
        frame["model"] = model_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_bucket(frame, bucket_name):
    selected = frame.loc[frame["product_type_switch_bucket_recent5"] == bucket_name].copy()
    grouped = (
        selected.groupby("model")
        .agg(
            users=("source_user_id", "count"),
            recall20=("recall@20", "mean"),
            ndcg20=("ndcg@20", "mean"),
            mrr20=("mrr@20", "mean"),
            history_len=("history_len_true", "mean"),
            unique_product_types=("product_type_unique_recent5", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "switch_bucket", bucket_name)
    return grouped


def build_report(summary_frame):
    lines = [
        "# Product-Type Switch Analysis",
        "",
        "최근 5회 구매에서 `product_type` 전환 횟수를 기준으로 `0`, `1-2`, `3+` switch bucket을 구성했습니다.",
        "이 분석의 목적은 `DIF-SR + Metadata`가 과도하게 넓은 mixed-intent slice가 아니라, 보다 국소적인 intent switching 구간에서 추가 이득을 보이는지 확인하는 것입니다.",
        "",
        "| Switch Bucket | Model | Users | Recall@20 | NDCG@20 | MRR@20 | Mean History | Mean Unique Product Types |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_frame.to_dict(orient="records"):
        lines.append(
            f"| {row['switch_bucket']} | {row['model']} | {int(row['users'])} | "
            f"{row['recall20']:.4f} | {row['ndcg20']:.4f} | {row['mrr20']:.4f} | "
            f"{row['history_len']:.2f} | {row['unique_product_types']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `0 switches`는 최근 구매 맥락이 매우 안정적인 경우입니다.",
            "- `1-2 switches`는 locally coherent한 intent switching을 나타내는 후보 구간입니다.",
            "- `3+ switches`는 최근 구매 맥락이 더 복잡하거나 noisy한 경우에 가깝습니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    descriptors = build_product_type_descriptors()
    model_rows = load_model_rows(descriptors)
    toppop_rows = compute_toppopular_rows(descriptors)
    all_rows = pd.concat([model_rows, toppop_rows], ignore_index=True)

    buckets = ["0 switches", "1-2 switches", "3+ switches"]
    summary_frame = pd.concat([summarize_bucket(all_rows, bucket) for bucket in buckets], ignore_index=True)

    csv_path = OUTPUT_DIR / "product_type_switch_summary.csv"
    json_path = OUTPUT_DIR / "product_type_switch_summary.json"
    summary_frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary_frame.to_dict(orient="records"), indent=2) + "\n")
    REPORT_PATH.write_text(build_report(summary_frame))

    print(
        json.dumps(
            {
                "summary_csv": str(csv_path),
                "summary_json": str(json_path),
                "report": str(REPORT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
