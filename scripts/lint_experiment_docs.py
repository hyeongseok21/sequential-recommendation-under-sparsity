#!/usr/bin/env python3
from pathlib import Path
import sys


REQUIRED = [
    "docs/framework/AGENT.md",
    "docs/CLOSURE_MODELS.md",
    "docs/CLOSURE_PREFLIGHT.md",
    "docs/EXPERIMENT_PHASES.md",
    "docs/EXPERIMENT_PLAN.md",
    "SLICE_EVALUATION.md",
    "docs/framework/RUNBOOK.md",
    "docs/framework/SELF_EVOLUTION_LOOP.md",
    "docs/framework/protocol.md",
    "docs/agents/operator.md",
    "docs/agents/analyst.md",
    "docs/agents/research.md",
    "docs/agents/governor.md",
    "docs/agents/handoff.md",
    "docs/agents/orchestration.md",
    "docs/agents/automation.md",
    "docs/automations/README.md",
    "docs/automations/analyst.md",
    "docs/automations/operator.md",
    "docs/automations/governor.md",
    "docs/templates/agent_handoff_template.md",
    "docs/templates/agent_prompt_template.md",
    "docs/templates/automation_prompt_bundle_template.md",
    "docs/templates/update_log_template.md",
    "scripts/check_closure_readiness.py",
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
