import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
HM_ROOT = REPO_ROOT / "hm_refactored"
if str(HM_ROOT) not in sys.path:
    sys.path.insert(0, str(HM_ROOT))

from train import Trainer  # noqa: E402


DATA_DIR = REPO_ROOT / "data/metrics/research_study"
PLOTS_DIR = REPO_ROOT / "plots"
TOPK_CACHE_DIR = DATA_DIR / "topk"
PAIRWISE_OVERLAP_CACHE = DATA_DIR / "candidate_overlap_raw.csv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    uses_metadata: bool
    config_path: Path
    checkpoint_name: str
    user_metrics_path: Path


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TOPK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def canonical_model_specs():
    return [
        ModelSpec(
            name="SASRec",
            family="sasrec",
            uses_metadata=False,
            config_path=REPO_ROOT / "hm_refactored/configs/config.closure_sasrec.json",
            checkpoint_name="hm_closure_sasrec",
            user_metrics_path=REPO_ROOT / "data/metrics/closure/hm_closure_sasrec_user_metrics.json",
        ),
        ModelSpec(
            name="SASRec + metadata",
            family="sasrec",
            uses_metadata=True,
            config_path=REPO_ROOT / "hm_refactored/configs/config.closure_sasrec_meta.json",
            checkpoint_name="hm_closure_sasrec_meta",
            user_metrics_path=REPO_ROOT / "data/metrics/closure/hm_closure_sasrec_meta_user_metrics.json",
        ),
        ModelSpec(
            name="DIF-SR",
            family="difsr",
            uses_metadata=False,
            config_path=REPO_ROOT / "hm_refactored/configs/config.closure_difsr.json",
            checkpoint_name="hm_closure_difsr",
            user_metrics_path=REPO_ROOT / "data/metrics/closure/hm_closure_difsr_user_metrics.json",
        ),
        ModelSpec(
            name="DIF-SR + metadata",
            family="difsr",
            uses_metadata=True,
            config_path=REPO_ROOT / "hm_refactored/configs/config.closure_difsr_meta.json",
            checkpoint_name="hm_closure_difsr_meta",
            user_metrics_path=REPO_ROOT / "data/metrics/closure/hm_closure_difsr_meta_user_metrics.json",
        ),
    ]


def load_best_epoch(checkpoint_dir: Path) -> int:
    payload = json.loads((checkpoint_dir / "epoch_summary.json").read_text())
    return int(payload["best_benchmark"]["epoch"])


def load_user_metrics(spec):
    rows = json.loads(spec.user_metrics_path.read_text())
    return pd.DataFrame(rows)


def build_item_metadata(trainer: Trainer) -> pd.DataFrame:
    columns = [
        "item_id",
        "department_no",
        "product_type_no",
        "garment_group_no",
    ]
    item_meta = (
        trainer.data_dict["train_df"][columns]
        .drop_duplicates("item_id")
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    return item_meta


def build_item_popularity(trainer: Trainer):
    counts = trainer.data_dict["train_df"]["item_id"].value_counts().sort_values(ascending=False)
    items = counts.index.tolist()
    total_items = len(items)
    head_cut = int(total_items * 0.2)
    mid_cut = int(total_items * 0.5)

    bucket_map = {}
    for rank, item_id in enumerate(items):
        if rank < head_cut:
            bucket_map[int(item_id)] = "head"
        elif rank < mid_cut:
            bucket_map[int(item_id)] = "mid"
        else:
            bucket_map[int(item_id)] = "tail"
    return bucket_map, counts.to_dict()


def load_or_build_topk_records(spec, top_k=100, force=False):
    ensure_dirs()
    cache_path = TOPK_CACHE_DIR / f"{spec.checkpoint_name}_top{top_k}.json"
    if cache_path.exists() and not force:
        return pd.DataFrame(json.loads(cache_path.read_text()))

    trainer = Trainer(str(spec.config_path))
    checkpoint_dir = Path(trainer.train_params["save_path"]) / trainer.train_params["save_name"]
    best_epoch = load_best_epoch(checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"{best_epoch:06d}_epoch.pth"
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.model.eval()

    user_metrics = load_user_metrics(spec).set_index("source_user_id")
    item_meta = build_item_metadata(trainer).set_index("item_id")

    records = []
    for user_batch, mask, history, history_mask in trainer.test_dl:
        _ = mask
        user_tensor = trainer._to_device_tensor(user_batch[:, 0], torch.long)
        history_tensor = trainer._to_device_tensor(history, torch.long)
        history_mask_tensor = trainer._to_device_tensor(history_mask, torch.long)

        predictions, _, _ = trainer.model.recommend(user_tensor, history_tensor, history_mask_tensor)
        effective_top_k = min(top_k, predictions.shape[1])
        _, recommends = torch.topk(predictions.detach(), effective_top_k)
        recommends = recommends.detach().cpu().numpy()

        for batch_idx, user_idx in enumerate(user_tensor.detach().cpu().numpy().tolist()):
            source_user_id = str(trainer.data_dict["idx2user"][int(user_idx)])
            user_row = user_metrics.loc[source_user_id].to_dict()
            top_items = [int(item) for item in recommends[batch_idx].tolist()]
            top20_items = top_items[:20]
            departments = [
                int(item_meta.loc[item_id, "department_no"])
                for item_id in top20_items
                if item_id in item_meta.index
            ]

            records.append(
                {
                    "model": spec.name,
                    "checkpoint_epoch": best_epoch,
                    "user_idx": int(user_idx),
                    "source_user_id": source_user_id,
                    "history_len": int(user_row["history_len"]),
                    "history_bucket": user_row["history_bucket"],
                    "category_entropy": float(user_row["category_entropy"]),
                    "category_entropy_bucket": user_row["category_entropy_bucket"],
                    "recall@20": float(user_row["recall@20"]),
                    "ndcg@20": float(user_row["ndcg@20"]),
                    "mrr@20": float(user_row["mrr@20"]),
                    "is_sparse_history": int(user_row["is_sparse_history"]),
                    "is_multi_interest": int(user_row["is_multi_interest"]),
                    "top20_items": top20_items,
                    "top100_items": top_items,
                    "top20_departments": departments,
                }
            )

    cache_path.write_text(json.dumps(records, indent=2) + "\n")
    return pd.DataFrame(records)


def load_all_model_records(top_k=100, force=False):
    records = {}
    item_meta = None
    popularity_bucket = None
    popularity_count = None

    for spec in canonical_model_specs():
        frame = load_or_build_topk_records(spec, top_k=top_k, force=force)
        records[spec.name] = frame
        if item_meta is None or popularity_bucket is None:
            trainer = Trainer(str(spec.config_path))
            item_meta = build_item_metadata(trainer)
            popularity_bucket, popularity_count = build_item_popularity(trainer)

    return {
        "records": records,
        "item_meta": item_meta,
        "popularity_bucket": popularity_bucket,
        "popularity_count": popularity_count,
    }


def shannon_entropy(values):
    if not values:
        return 0.0
    counts = pd.Series(values).value_counts(normalize=True)
    return float(-(counts * np.log(counts)).sum())


def summarize_distribution(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "std": float(np.std(arr)) if len(arr) else 0.0,
    }


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
