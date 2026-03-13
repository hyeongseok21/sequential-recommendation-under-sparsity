#!/usr/bin/env python3
from pathlib import Path
import sys


REQUIRED = [
    "AGENT.md",
    "EXPERIMENT_PHASES.md",
    "RUNBOOK.md",
    "SELF_EVOLUTION_LOOP.md",
    "protocol.md",
    "agents/operator.md",
    "agents/analyst.md",
    "agents/research.md",
    "agents/governor.md",
    "agents/handoff.md",
    "agents/orchestration.md",
    "agents/automation.md",
    "automations/README.md",
    "automations/analyst.md",
    "automations/operator.md",
    "automations/governor.md",
    "templates/agent_handoff_template.md",
    "templates/agent_prompt_template.md",
    "templates/automation_prompt_bundle_template.md",
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
