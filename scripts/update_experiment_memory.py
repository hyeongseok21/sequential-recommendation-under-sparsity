#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
from pathlib import Path


HEADER = [
    "timestamp",
    "phase",
    "axis",
    "hypothesis",
    "baseline_config",
    "treatment_config",
    "primary_metric",
    "baseline_value",
    "treatment_value",
    "delta",
    "verdict",
    "notes",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--axis", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--treatment-config", required=True)
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument("--baseline-value", type=float, required=True)
    parser.add_argument("--treatment-value", type=float, required=True)
    parser.add_argument("--verdict", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    delta = args.treatment_value - args.baseline_value

    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            args.phase,
            args.axis,
            args.hypothesis,
            args.baseline_config,
            args.treatment_config,
            args.primary_metric,
            args.baseline_value,
            args.treatment_value,
            delta,
            args.verdict,
            args.notes,
        ])


if __name__ == "__main__":
    main()
