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


USER_METRIC_FILES = {
    "Corrected SASRec": REPO_ROOT / "data/metrics/closure/hm_sanity_sasrec_fixed_user_metrics.json",
    "DIF-SR": REPO_ROOT / "data/metrics/closure/hm_closure_difsr_user_metrics.json",
    "DIF-SR + Metadata": REPO_ROOT / "data/metrics/closure/hm_closure_difsr_meta_user_metrics.json",
}

OUTPUT_DIR = REPO_ROOT / "data/metrics/research_study"
REPORT_PATH = REPO_ROOT / "reports/intent_switching_analysis.md"


def _history_bucket(history_len: int) -> str:
    if history_len <= 5:
        return "short<=5"
    if history_len <= 15:
        return "mid6-15"
    return "long>15"


def _switch_bucket(switch_rate: float) -> str:
    if switch_rate < 0.2:
        return "low-switch"
    if switch_rate < 0.5:
        return "mid-switch"
    return "high-switch"


def _entropy_bucket(entropy: float) -> str:
    if entropy < 0.5:
        return "low"
    if entropy < 1.0:
        return "mid"
    return "high"


def build_user_descriptors():
    trainer = Trainer(config_path=str(REPO_ROOT / "hm_refactored/configs/config.closure_sasrec.json"))
    train_df = trainer.data_dict["train_df"][["user_id", "timestamp", "department_no"]].copy()
    train_df = train_df.sort_values(["user_id", "timestamp"])

    descriptors = {}
    for user_id, group in train_df.groupby("user_id"):
        departments = group["department_no"].tolist()
        recent = departments[-10:]
        unique_departments = len(set(recent))
        transitions = sum(1 for left, right in zip(recent[:-1], recent[1:]) if left != right)
        if recent:
            probs = pd.Series(recent).value_counts(normalize=True)
            entropy = float(-(probs * np.log(probs)).sum())
        else:
            entropy = 0.0
        switch_rate = float(transitions / max(len(recent) - 1, 1)) if recent else 0.0
        descriptors[int(user_id)] = {
            "history_len_true": len(departments),
            "history_bucket_true": _history_bucket(len(departments)),
            "recent_len": len(recent),
            "transition_count_recent10": transitions,
            "switch_rate_recent10": switch_rate,
            "switch_bucket_recent10": _switch_bucket(switch_rate),
            "entropy_recent10": entropy,
            "category_entropy_bucket_recent10": _entropy_bucket(entropy),
            "is_multi_interest_recent10": int(unique_departments >= 3 or transitions >= 4),
        }

    # TopPopular on the same user universe as the corrected baseline.
    popular_items = list(map(int, trainer.popular_items[:100]))
    toppop_rows = []
    for user_batch, mask, history, history_mask in trainer.test_dl:
        _ = mask
        _ = history
        _ = history_mask
        for user_idx in user_batch[:, 0].numpy().tolist():
            source_user_id = str(trainer.data_dict["idx2user"][int(user_idx)])
            descriptor = descriptors[int(user_idx)]
            gt_items = [
                trainer.data_dict["item2idx"].get(item, -1)
                for item in trainer.data_dict["user_last_test_dict"][source_user_id]
            ]
            rank = 10**9
            for item in gt_items:
                if item in popular_items:
                    rank = popular_items.index(item)
                    break
            recall20 = 1.0 if rank < 20 else 0.0
            ndcg20 = (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0
            mrr20 = (1.0 / (rank + 1)) if rank < 20 else 0.0
            row = {
                "user_idx": int(user_idx),
                "source_user_id": source_user_id,
                "recall@20": float(recall20),
                "ndcg@20": float(ndcg20),
                "mrr@20": float(mrr20),
                "history_len": descriptor["history_len_true"],
                "history_bucket": descriptor["history_bucket_true"],
                "category_entropy_bucket": descriptor["category_entropy_bucket_recent10"],
                "is_multi_interest": descriptor["is_multi_interest_recent10"],
            }
            row.update(descriptor)
            toppop_rows.append(row)

    return descriptors, pd.DataFrame(toppop_rows)


def load_frames(descriptors):
    frames = []
    for model_name, path in USER_METRIC_FILES.items():
        frame = pd.DataFrame(json.loads(path.read_text()))
        for col in [
            "history_len_true",
            "history_bucket_true",
            "recent_len",
            "transition_count_recent10",
            "switch_rate_recent10",
            "switch_bucket_recent10",
            "entropy_recent10",
            "category_entropy_bucket_recent10",
            "is_multi_interest_recent10",
        ]:
            frame[col] = frame["user_idx"].map(lambda uid, c=col: descriptors[int(uid)][c])
        frame["model"] = model_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_slice(frame, slice_name, predicate):
    selected = frame.loc[predicate(frame)].copy()
    if selected.empty:
        return pd.DataFrame()
    grouped = (
        selected.groupby("model")
        .agg(
            users=("source_user_id", "count"),
            recall20=("recall@20", "mean"),
            ndcg20=("ndcg@20", "mean"),
            mrr20=("mrr@20", "mean"),
            history_len=("history_len_true", "mean"),
            transitions=("transition_count_recent10", "mean"),
            switch_rate=("switch_rate_recent10", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "slice", slice_name)
    return grouped


def build_report(summary_frame, descriptors_frame):
    low = descriptors_frame.loc[descriptors_frame["category_entropy_bucket"] == "low"]
    mid = descriptors_frame.loc[descriptors_frame["category_entropy_bucket"] == "mid"]
    high = descriptors_frame.loc[descriptors_frame["category_entropy_bucket"] == "high"]

    lines = [
        "# Intent Switching Analysis",
        "",
        "## Key Checks",
        "",
        "- `high_entropy`가 실제 intent-switching proxy로도 읽히는지 확인",
        "- `high_entropy`와 `history length`가 얼마나 강하게 결합되어 있는지 확인",
        "- `DIF-SR + Metadata`가 switch-heavy user에서 정말 더 강한지 확인",
        "",
        "## Descriptor Sanity Check",
        "",
        f"- low entropy: users `{len(low)}`, mean history `{low['history_len'].mean():.2f}`, mean switch rate `{low['switch_rate_recent10'].mean():.3f}`",
        f"- mid entropy: users `{len(mid)}`, mean history `{mid['history_len'].mean():.2f}`, mean switch rate `{mid['switch_rate_recent10'].mean():.3f}`",
        f"- high entropy: users `{len(high)}`, mean history `{high['history_len'].mean():.2f}`, mean switch rate `{high['switch_rate_recent10'].mean():.3f}`",
        "",
        "## Slice Summary",
        "",
        "| Slice | Model | Users | Recall@20 | NDCG@20 | MRR@20 | Mean History | Mean Transitions | Mean Switch Rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_frame.to_dict(orient="records"):
        lines.append(
            f"| {row['slice']} | {row['model']} | {int(row['users'])} | "
            f"{row['recall20']:.4f} | {row['ndcg20']:.4f} | {row['mrr20']:.4f} | "
            f"{row['history_len']:.2f} | {row['transitions']:.2f} | {row['switch_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        "- `high_entropy`는 mixed-intent proxy로는 유효하지만, 동시에 long-history user가 많이 섞여 있어 pure switching slice로 보기 어렵습니다.",
        "- `switch-heavy`로 다시 봐도 `Corrected SASRec`가 가장 강하면, 현재 레짐에서는 intent switching보다 sequence signal과 popularity alignment가 더 크게 작동한다고 볼 수 있습니다.",
        "- `DIF-SR + Metadata`가 switch-heavy slice에서 약하면, metadata가 switching signal을 넓게 포착하기보다 candidate space를 더 좁히는 방향으로 작동했을 가능성이 큽니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    descriptors, toppop_frame = build_user_descriptors()
    base_frames = load_frames(descriptors)
    all_frames = pd.concat([base_frames, toppop_frame.assign(model="TopPopular")], ignore_index=True)

    slices = [
        summarize_slice(all_frames, "high-entropy", lambda df: df["category_entropy_bucket"] == "high"),
        summarize_slice(all_frames, "switch-heavy", lambda df: df["transition_count_recent10"] >= 4),
        summarize_slice(
            all_frames,
            "high-entropy & short<=5",
            lambda df: (df["category_entropy_bucket"] == "high") & (df["history_len"] <= 5),
        ),
        summarize_slice(
            all_frames,
            "high-entropy & mid6-15",
            lambda df: (df["category_entropy_bucket"] == "high") & (df["history_len"].between(6, 15)),
        ),
        summarize_slice(
            all_frames,
            "high-entropy & long>15",
            lambda df: (df["category_entropy_bucket"] == "high") & (df["history_len"] > 15),
        ),
    ]
    summary_frame = pd.concat([frame for frame in slices if not frame.empty], ignore_index=True)

    summary_csv = OUTPUT_DIR / "intent_switching_summary.csv"
    summary_json = OUTPUT_DIR / "intent_switching_summary.json"
    summary_frame.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_frame.to_dict(orient="records"), indent=2) + "\n")

    sanity_frame = pd.DataFrame(json.loads(USER_METRIC_FILES["Corrected SASRec"].read_text()))
    for col in [
        "transition_count_recent10",
        "switch_rate_recent10",
    ]:
        sanity_frame[col] = sanity_frame["user_idx"].map(lambda uid, c=col: descriptors[int(uid)][c])
    REPORT_PATH.write_text(build_report(summary_frame, sanity_frame))

    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json),
                "report": str(REPORT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
