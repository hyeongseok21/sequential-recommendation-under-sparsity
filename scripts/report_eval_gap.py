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


def parse_loss(path: Path):
    rows = {}
    for line in path.read_text().splitlines():
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
    parser.add_argument("--loss-path", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = parse_loss(Path(args.loss_path))
    benchmark_rows = [row for row in rows if "b_ndcg" in row]
    test_rows = [row for row in rows if "t_map" in row]
    if not benchmark_rows or not test_rows:
        raise ValueError("loss file missing benchmark or test rows")

    best_benchmark = max(benchmark_rows, key=lambda row: (row["b_ndcg"], row.get("b_hr", 0.0)))
    best_test = max(test_rows, key=lambda row: (row["t_map"], row.get("t_hr", 0.0)))

    payload = {
        "loss_path": args.loss_path,
        "best_benchmark": best_benchmark,
        "best_test": best_test,
        "different_epoch": best_benchmark["epoch"] != best_test["epoch"],
        "epoch_gap": best_test["epoch"] - best_benchmark["epoch"],
        "benchmark_view_at_test_epoch": next((row for row in rows if row["epoch"] == best_test["epoch"]), None),
        "test_view_at_benchmark_epoch": next((row for row in rows if row["epoch"] == best_benchmark["epoch"]), None),
    }

    rendered = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
