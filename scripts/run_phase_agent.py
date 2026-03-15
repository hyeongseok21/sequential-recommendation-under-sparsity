#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE_DOC = REPO_ROOT / "docs/EXPERIMENT_PHASES.md"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"


@dataclass
class StepResult:
    label: str
    command: list[str]
    returncode: int
    stdout_tail: str
    stderr_tail: str


PHASE_UPDATE_DIR = {
    "P0": "0.local-setup",
    "P1": "1.sasrec-vs-meta-embedding",
    "P2": "2.sasrec-vs-meta-embedding-vs-difsr",
    "P3": "3.architecture-validation",
    "P4": "4.inference-policy",
    "P5": "5.slice-serving-finalization",
    "P6": "6.portfolio-closure",
}


PHASE_HYPOTHESIS = {
    "P6": "closure phase에서는 새 탐색보다 결과 패키징과 문서 일관성 검증이 더 중요하다.",
}


def detect_active_phase() -> str:
    text = PHASE_DOC.read_text(encoding="utf-8")
    match = re.search(r"- 현재 active phase:\s*\n\s*-\s*`(P\d+)`", text)
    if not match:
        raise RuntimeError(f"Could not detect active phase from {PHASE_DOC}")
    return match.group(1)


def phase_commands(phase: str) -> list[tuple[str, list[str]]]:
    python_cmd = str(PYTHON_BIN if PYTHON_BIN.exists() else Path(sys.executable))
    if phase == "P6":
        return [
            ("lint framework docs", [python_cmd, "scripts/lint_experiment_docs.py"]),
            ("run research analysis", [python_cmd, "experiments/run_evaluation.py"]),
            ("run service-style evaluation", [python_cmd, "scripts/generate_service_style_eval.py"]),
            ("package portfolio artifact", [python_cmd, "experiments/package_portfolio_artifact.py"]),
        ]
    raise RuntimeError(f"Active phase {phase} is not automated yet. Supported phases: P6")


def run_step(label: str, command: list[str]) -> StepResult:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    stdout_tail = "\n".join(completed.stdout.strip().splitlines()[-12:])
    stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-12:])
    return StepResult(
        label=label,
        command=command,
        returncode=completed.returncode,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def write_update_log(phase: str, results: list[StepResult], dry_run: bool) -> Path:
    timestamp = datetime.now()
    update_dir = REPO_ROOT / "updates" / PHASE_UPDATE_DIR.get(phase, "misc")
    update_dir.mkdir(parents=True, exist_ok=True)
    update_path = update_dir / f"{timestamp:%Y-%m-%d-%H%M}-phase-agent-run.md"

    hypothesis = PHASE_HYPOTHESIS.get(phase, "현재 active phase 기준으로 자동 실행 루프를 수행한다.")
    overall_verdict = "PASS" if all(result.returncode == 0 for result in results) else "FAIL"

    lines = [
        "# Phase Agent Run",
        "",
        "## Goal",
        "",
        f"- active phase를 문서에서 자동 감지하고 phase 맞춤 명령을 실행한다.",
        f"- 감지된 phase: `{phase}`",
        f"- 가설: {hypothesis}",
        f"- dry-run: `{str(dry_run).lower()}`",
        "",
        "## Commands",
        "",
    ]

    for result in results:
        lines.extend(
            [
                f"### {result.label}",
                "",
                "```bash",
                " ".join(result.command),
                "```",
                "",
                f"- return code: `{result.returncode}`",
            ]
        )
        if result.stdout_tail:
            lines.extend(["- stdout tail:", "", "```text", result.stdout_tail, "```"])
        if result.stderr_tail:
            lines.extend(["- stderr tail:", "", "```text", result.stderr_tail, "```"])
        lines.append("")

    lines.extend(
        [
            "## Current Verdict",
            "",
            f"- overall verdict: `{overall_verdict}`",
            f"- update log path: `{update_path.relative_to(REPO_ROOT)}`",
            "",
            "## Notes",
            "",
            "- 현재 구현은 active phase가 `P6`일 때 closure automation을 수행한다.",
            "- 다른 phase는 추후 축별 baseline/treatment 루프로 확장할 수 있다.",
            "",
        ]
    )

    update_path.write_text("\n".join(lines), encoding="utf-8")
    return update_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phase-aware experiment agent.")
    parser.add_argument("--phase", help="Override detected phase.")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute commands; only log planned commands.")
    args = parser.parse_args()

    phase = args.phase or detect_active_phase()
    commands = phase_commands(phase)

    if args.dry_run:
        results = [StepResult(label=label, command=command, returncode=0, stdout_tail="", stderr_tail="") for label, command in commands]
    else:
        results = []
        for label, command in commands:
            result = run_step(label, command)
            results.append(result)
            if result.returncode != 0:
                break

    update_path = write_update_log(phase, results, dry_run=args.dry_run)
    print(f"phase={phase}")
    print(f"update_log={update_path}")
    if not all(result.returncode == 0 for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
