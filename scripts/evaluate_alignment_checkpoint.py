#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_closure_report import evaluate_test_checkpoint
from hm_refactored.train import Trainer


def compute_at_k(rows, k):
    recall = sum(1.0 if row["rank"] < k else 0.0 for row in rows) / len(rows)
    ndcg = sum((1.0 / math.log2(row["rank"] + 2)) if row["rank"] < k else 0.0 for row in rows) / len(rows)
    return recall, ndcg


def attach_ranks(trainer, checkpoint_epoch, sparse_threshold=5, recent_window=10):
    checkpoint_path = (
        Path(trainer.train_params["save_path"])
        / trainer.train_params["save_name"]
        / f"{checkpoint_epoch:06d}_epoch.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.model.eval()

    descriptor_rows = evaluate_test_checkpoint(
        trainer,
        checkpoint_epoch=checkpoint_epoch,
        sparse_threshold=sparse_threshold,
        recent_window=recent_window,
    )
    per_user = {row["source_user_id"]: row for row in descriptor_rows}

    rows = []
    for user_batch, mask, history, history_mask in trainer.test_dl:
        _ = mask
        user_tensor = trainer._to_device_tensor(user_batch[:, 0], torch.long)
        history_tensor = trainer._to_device_tensor(history, torch.long)
        history_mask_tensor = trainer._to_device_tensor(history_mask, torch.long)

        predictions, _, _ = trainer.model.recommend(user_tensor, history_tensor, history_mask_tensor)
        _, top100 = torch.topk(predictions.detach(), min(100, predictions.shape[1]))
        top100 = top100.detach().cpu().numpy()

        for batch_idx, user_idx in enumerate(user_tensor.detach().cpu().numpy().tolist()):
            source_user_id = str(trainer.data_dict["idx2user"][int(user_idx)])
            gt_items = [trainer.data_dict["item2idx"].get(i, -1) for i in trainer.data_dict["user_last_test_dict"][source_user_id]]
            recs = top100[batch_idx].tolist()
            rank = 10**9
            for item in gt_items:
                if item in recs:
                    rank = recs.index(item)
                    break
            row = dict(per_user[source_user_id])
            row["rank"] = int(rank)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-epoch", required=True, type=int)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    trainer = Trainer(config_path=args.config_path)
    rows = attach_ranks(trainer, checkpoint_epoch=args.checkpoint_epoch)

    summary = {
        "config_path": str(Path(args.config_path).resolve()),
        "checkpoint_epoch": args.checkpoint_epoch,
        "num_users": len(rows),
        "Recall@20": compute_at_k(rows, 20)[0],
        "Recall@50": compute_at_k(rows, 50)[0],
        "Recall@100": compute_at_k(rows, 100)[0],
        "NDCG@20": compute_at_k(rows, 20)[1],
        "NDCG@50": compute_at_k(rows, 50)[1],
        "NDCG@100": compute_at_k(rows, 100)[1],
        "MRR@20": sum((1.0 / (row["rank"] + 1)) if row["rank"] < 20 else 0.0 for row in rows) / len(rows),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
