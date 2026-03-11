#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument("--baseline", type=float, required=True)
    parser.add_argument("--treatment", type=float, required=True)
    parser.add_argument("--secondary-baseline", type=float, default=None)
    parser.add_argument("--secondary-treatment", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    delta = args.treatment - args.baseline
    verdict = "PASS"
    reason = "treatment improved primary metric"

    if delta < 0:
        verdict = "FAIL"
        reason = "treatment regressed primary metric"
    elif delta == 0 and args.secondary_baseline is not None and args.secondary_treatment is not None:
        if args.secondary_treatment < args.secondary_baseline:
            verdict = "FAIL"
            reason = "primary metric tied and secondary metric regressed"

    payload = {
        "phase": args.phase,
        "primary_metric": args.primary_metric,
        "baseline": args.baseline,
        "treatment": args.treatment,
        "delta": delta,
        "verdict": verdict,
        "reason": reason,
        "notes": args.notes,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
