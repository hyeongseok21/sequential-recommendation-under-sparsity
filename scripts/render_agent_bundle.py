#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


ROLE_DOCS = {
    "operator": "agents/operator.md",
    "analyst": "agents/analyst.md",
    "research": "agents/research.md",
    "governor": "agents/governor.md",
}


def build_bundle(args: argparse.Namespace) -> str:
    role_doc = ROLE_DOCS[args.role]
    treatment_line = args.treatment_config or "(to be proposed)"
    hypothesis_line = args.hypothesis or "(to be proposed)"
    update_log_line = args.update_log or "(to be created)"
    gate_result = args.gate_result or "data/metrics/gate_result.json"
    memory_path = args.memory_path or "data/metrics/experiment_memory.csv"

    return f"""# {args.role.capitalize()} Automation Prompt

You are operating inside the recommendation experiment workspace.

## Read First

- AGENT.md
- RUNBOOK.md
- SELF_EVOLUTION_LOOP.md
- protocol.md
- {role_doc}
- agents/handoff.md
- agents/orchestration.md
- templates/agent_handoff_template.md
- templates/agent_prompt_template.md

## Current State

- phase: {args.phase}
- current champion: {args.champion_config}
- primary metric: {args.primary_metric}
- baseline config: {args.baseline_config}
- treatment config: {treatment_line}
- hypothesis: {hypothesis_line}

## Required Outputs

- update log: {update_log_line}
- gate result: {gate_result}
- experiment memory: {memory_path}

## Hard Rules

- one hypothesis per loop
- keep baseline fixed to the current champion unless explicitly told otherwise
- use {args.primary_metric} as the primary metric
- record file paths and numeric metrics
- distinguish concept fail from current implementation fail
- update artifacts, not just prose
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a role-specific automation prompt bundle.")
    parser.add_argument("--role", required=True, choices=sorted(ROLE_DOCS))
    parser.add_argument("--phase", required=True)
    parser.add_argument("--primary-metric", default="B_NDCG")
    parser.add_argument("--champion-config", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--treatment-config")
    parser.add_argument("--hypothesis")
    parser.add_argument("--update-log")
    parser.add_argument("--gate-result")
    parser.add_argument("--memory-path")
    parser.add_argument("--output")
    args = parser.parse_args()

    bundle = build_bundle(args)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(bundle, encoding="utf-8")
    else:
        print(bundle)


if __name__ == "__main__":
    main()
