#!/usr/bin/env python3
from pathlib import Path
import sys


REQUIRED = [
    "AGENT.md",
    "RUNBOOK.md",
    "SELF_EVOLUTION_LOOP.md",
    "protocol.md",
    "agents/operator.md",
    "agents/analyst.md",
    "agents/research.md",
    "agents/governor.md",
    "agents/handoff.md",
    "agents/orchestration.md",
    "templates/update_log_template.md",
    "data/metrics/experiment_memory.csv",
]


def main():
    root = Path.cwd()
    missing = [path for path in REQUIRED if not (root / path).exists()]
    if missing:
        print("Missing required experiment framework files:")
        for path in missing:
            print(path)
        sys.exit(1)
    print("Experiment framework docs look consistent.")


if __name__ == "__main__":
    main()
