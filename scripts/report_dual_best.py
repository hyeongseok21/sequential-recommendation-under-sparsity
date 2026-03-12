#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


B_RE = re.compile(
    r"\[(?P<epoch>\d+) epoch\] B_USER: .* B_HR: (?P<b_hr>[0-9.]+), B_NDCG: (?P<b_ndcg>[0-9.]+)"
)
T_RE = re.compile(
    r"\[(?P<epoch>\d+) epoch\] T_USER: .* T_HR: (?P<t_hr>[0-9.]+), T_MAP: (?P<t_map>[0-9.]+)"
)


def parse_loss(loss_path: Path):
    rows = {}
    for line in loss_path.read_text().splitlines():
        b_match = B_RE.search(line)
        if b_match:
            epoch = int(b_match.group("epoch"))
            rows.setdefault(epoch, {"epoch": epoch})
            rows[epoch]["b_hr"] = float(b_match.group("b_hr"))
            rows[epoch]["b_ndcg"] = float(b_match.group("b_ndcg"))
        t_match = T_RE.search(line)
        if t_match:
            epoch = int(t_match.group("epoch"))
            rows.setdefault(epoch, {"epoch": epoch})
            rows[epoch]["t_hr"] = float(t_match.group("t_hr"))
            rows[epoch]["t_map"] = float(t_match.group("t_map"))
    return sorted(rows.values(), key=lambda row: row["epoch"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    loss_path = checkpoint_dir / "loss.txt"
    rows = parse_loss(loss_path)
    benchmark_rows = [row for row in rows if "b_ndcg" in row]
    test_rows = [row for row in rows if "t_map" in row]
    best_benchmark = max(benchmark_rows, key=lambda row: (row["b_ndcg"], row.get("b_hr", 0.0)))
    best_test = max(test_rows, key=lambda row: (row["t_map"], row.get("t_hr", 0.0)))

    payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "loss_path": str(loss_path),
        "best_benchmark": best_benchmark,
        "best_test": best_test,
        "different_epoch": best_benchmark["epoch"] != best_test["epoch"],
        "epoch_gap": best_test["epoch"] - best_benchmark["epoch"],
    }

    for epoch in {best_benchmark["epoch"], best_test["epoch"]}:
        eval_path = checkpoint_dir / f"eval_checkpoint_{epoch:06d}.json"
        if eval_path.exists():
            payload.setdefault("evaluated_checkpoints", {})[str(epoch)] = json.loads(eval_path.read_text())

    rendered = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
