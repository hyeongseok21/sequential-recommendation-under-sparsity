#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.generate_final_portfolio_plots import load_canonical_results, load_service_results, main as generate_plots


REPORTS_DIR = REPO_ROOT / "reports"


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def build_canonical_report(canonical_results):
    lines = [
        "# Canonical Evaluation",
        "",
        "Primary artifact using the clean research-style dataset.",
        "",
        "| Model | Recall@20 | NDCG@20 | MRR@20 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model_name in ["TopPopular", "Corrected SASRec", "DIF-SR", "DIF-SR + Metadata"]:
        row = canonical_results[model_name]
        lines.append(
            f"| {model_name} | {row['Recall@20']:.4f} | {row['NDCG@20']:.4f} | {row['MRR@20']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- `TopPopular` remains the strongest model, which confirms strong popularity dominance.",
            "- `DIF-SR + Metadata` is the strongest personalized model under aligned conditions.",
            "- `Corrected SASRec` is the verified transformer baseline after causal masking and training fixes.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_service_report(canonical_results, service_results, service_slices):
    lines = [
        "# Service-Style Supplementary Evaluation",
        "",
        "Supplementary robustness analysis under relaxed production-like filtering:",
        "- cold-like users allowed",
        "- zero-history users allowed",
        "- repeat purchases allowed",
        "",
        "| Model | Canonical Recall@20 | Service Recall@20 | Canonical NDCG@20 | Service NDCG@20 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name in ["TopPopular", "Corrected SASRec", "DIF-SR", "DIF-SR + Metadata"]:
        c = canonical_results[model_name]
        s = service_results[model_name]
        lines.append(
            f"| {model_name} | {c['Recall@20']:.4f} | {s['Recall@20']:.4f} | {c['NDCG@20']:.4f} | {s['NDCG@20']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Key Slices",
            "",
            "| Slice | Best Model | Recall@20 | NDCG@20 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for slice_name in ["cold-like users", "short history users (<=5)", "repeat purchase cases"]:
        best_model = max(
            ["TopPopular", "Corrected SASRec", "DIF-SR", "DIF-SR + Metadata"],
            key=lambda m: service_slices[m][slice_name]["NDCG@20"],
        )
        row = service_slices[best_model][slice_name]
        lines.append(f"| {slice_name} | {best_model} | {row['Recall@20']:.4f} | {row['NDCG@20']:.4f} |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "- `DIF-SR + Metadata` is best on cold-like and short-history users.",
            "- `Corrected SASRec` is best on repeat purchase cases, which is consistent with memorization-heavy behavior.",
            "- Service-style metrics rise overall because repeat purchase cases make the task easier.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_summary_report():
    return "\n".join(
        [
            "# Research Portfolio Summary",
            "",
            "This repository studies sequential recommendation under sparse interaction regimes using H&M purchase sequences.",
            "",
            "Main conclusions:",
            "- baseline verification was necessary because the original SASRec baseline was invalid",
            "- the dataset is sparse and strongly popularity-dominated",
            "- `DIF-SR + Metadata` is the strongest personalized model in the canonical aligned comparison",
            "- metadata is most useful when behavioral signal is weak, especially in cold-like and short-history service slices",
            "",
            "Primary entry points:",
            f"- canonical summary: [{(REPORTS_DIR / 'canonical_evaluation.md').name}]({(REPORTS_DIR / 'canonical_evaluation.md').as_posix()})",
            f"- service-style summary: [{(REPORTS_DIR / 'service_style_evaluation.md').name}]({(REPORTS_DIR / 'service_style_evaluation.md').as_posix()})",
            f"- baseline sanity check: [{(REPO_ROOT / 'SASREC_SANITY_FIX.md').name}]({(REPO_ROOT / 'SASREC_SANITY_FIX.md').as_posix()})",
        ]
    ) + "\n"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_plots()
    canonical_results = load_canonical_results()
    service_results, service_slices = load_service_results()

    write_text(REPORTS_DIR / "canonical_evaluation.md", build_canonical_report(canonical_results))
    write_text(REPORTS_DIR / "service_style_evaluation.md", build_service_report(canonical_results, service_results, service_slices))
    write_text(REPORTS_DIR / "research_summary.md", build_summary_report())

    manifest = {
        "canonical_report": str((REPORTS_DIR / "canonical_evaluation.md").resolve()),
        "service_report": str((REPORTS_DIR / "service_style_evaluation.md").resolve()),
        "portfolio_summary": str((REPORTS_DIR / "research_summary.md").resolve()),
        "plot_model_comparison": str((REPO_ROOT / "plots/final_model_comparison.png").resolve()),
        "plot_canonical_vs_service": str((REPO_ROOT / "plots/canonical_vs_service_comparison.png").resolve()),
        "plot_slice_analysis": str((REPO_ROOT / "plots/final_slice_analysis.png").resolve()),
    }
    write_text(REPORTS_DIR / "manifest.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
