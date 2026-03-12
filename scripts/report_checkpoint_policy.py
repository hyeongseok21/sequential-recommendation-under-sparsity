#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


LINE_RE = re.compile(
    r"\[(?P<epoch>\d+) epoch\] B_USER: .* B_HR: (?P<b_hr>[0-9.]+), B_NDCG: (?P<b_ndcg>[0-9.]+)"
)


def parse_loss(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        match = LINE_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "b_hr": float(match.group("b_hr")),
                "b_ndcg": float(match.group("b_ndcg")),
            }
        )
    if not rows:
        raise ValueError(f"no benchmark rows found in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-path", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = parse_loss(Path(args.loss_path))
    best_hr = max(rows, key=lambda row: (row["b_hr"], row["b_ndcg"]))
    best_ndcg = max(rows, key=lambda row: (row["b_ndcg"], row["b_hr"]))

    payload = {
        "loss_path": args.loss_path,
        "best_by_hr": best_hr,
        "best_by_ndcg": best_ndcg,
        "delta_b_hr": round(best_ndcg["b_hr"] - best_hr["b_hr"], 6),
        "delta_b_ndcg": round(best_ndcg["b_ndcg"] - best_hr["b_ndcg"], 6),
        "different_epoch": best_hr["epoch"] != best_ndcg["epoch"],
    }

    rendered = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
