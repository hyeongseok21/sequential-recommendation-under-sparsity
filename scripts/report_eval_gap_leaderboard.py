#!/usr/bin/env python3
import argparse
import csv
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


def summarize_run(loss_path: Path):
    rows = parse_loss(loss_path)
    benchmark_rows = [row for row in rows if "b_ndcg" in row]
    test_rows = [row for row in rows if "t_map" in row]
    if not benchmark_rows or not test_rows:
        return None

    best_benchmark = max(benchmark_rows, key=lambda row: (row["b_ndcg"], row.get("b_hr", 0.0)))
    best_test = max(test_rows, key=lambda row: (row["t_map"], row.get("t_hr", 0.0)))
    return {
        "run_name": loss_path.parent.name,
        "loss_path": str(loss_path),
        "benchmark_epoch": best_benchmark["epoch"],
        "benchmark_b_ndcg": best_benchmark["b_ndcg"],
        "benchmark_t_map": best_benchmark.get("t_map", 0.0),
        "test_epoch": best_test["epoch"],
        "test_b_ndcg": best_test.get("b_ndcg", 0.0),
        "test_t_map": best_test["t_map"],
        "epoch_gap": best_test["epoch"] - best_benchmark["epoch"],
        "different_epoch": best_test["epoch"] != best_benchmark["epoch"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    summaries = []
    for loss_path in root.rglob("loss.txt"):
        if args.pattern not in str(loss_path.parent):
            continue
        summary = summarize_run(loss_path)
        if summary:
            summaries.append(summary)

    summaries.sort(key=lambda row: row["benchmark_b_ndcg"], reverse=True)

    Path(args.output_json).write_text(json.dumps(summaries, indent=2) + "\n")
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()) if summaries else [])
        if summaries:
            writer.writeheader()
            writer.writerows(summaries)

    print(json.dumps({"num_runs": len(summaries), "top_runs": summaries[:5]}, indent=2))


if __name__ == "__main__":
    main()
