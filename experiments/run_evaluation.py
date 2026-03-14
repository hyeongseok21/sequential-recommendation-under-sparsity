#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.candidate_overlap import run_candidate_overlap_analysis
from analysis.common import DATA_DIR, PLOTS_DIR, ensure_dirs, load_all_model_records, write_json
from analysis.diversity_analysis import run_diversity_analysis
from analysis.overall_metrics import run_overall_metrics_analysis
from analysis.popularity_bias import run_popularity_bias_analysis
from analysis.slice_analysis import run_slice_analysis


def build_research_summary(overall_frame, overlap_result, diversity_result, popularity_result, slice_result):
    best_overall = overall_frame.sort_values("NDCG@20", ascending=False).iloc[0]

    cold_rows = slice_result["slice_summary"]
    cold_best = cold_rows.loc[cold_rows["slice"] == "short_history"].sort_values("ndcg@20", ascending=False).iloc[0]
    high_entropy = cold_rows.loc[cold_rows["slice"] == "high_entropy"].sort_values("ndcg@20", ascending=False).iloc[0]

    overlap_summary = overlap_result["summary"].set_index("pair")
    diversity_summary = diversity_result["summary"].set_index("model")
    popularity_summary = popularity_result["summary"].set_index("model")
    lowest_overlap_pair = overlap_summary["mean"].idxmin()
    difsr_tail = popularity_summary.loc["DIF-SR", "tail_share"]
    difsr_meta_tail = popularity_summary.loc["DIF-SR + metadata", "tail_share"]

    lines = [
        "# Sequential Recommendation with Metadata: Research Summary",
        "",
        "## Problem",
        "We study whether item metadata improves sequential recommendation on sparse H&M purchase sequences, and which user segments benefit most.",
        "",
        "## Dataset",
        "- Users: 22,258",
        "- Items: 29,785",
        "- Interactions: 135,412",
        "- Average sequence length: 9.35",
        "- Sparsity: 99.98%",
        "- Metadata: product_type, department, garment_group",
        "",
        "## Models",
        "- SASRec",
        "- SASRec + Metadata",
        "- DIF-SR",
        "- DIF-SR + Metadata",
        "",
        "## Key Results",
        f"- Best overall model: {best_overall['model']} (NDCG@20 {best_overall['NDCG@20']:.4f}, Recall@20 {best_overall['Recall@20']:.4f})",
        f"- Best cold-start slice (`history<=5`): {cold_best['model']} (NDCG@20 {cold_best['ndcg@20']:.4f})",
        f"- Best high-entropy slice: {high_entropy['model']} (NDCG@20 {high_entropy['ndcg@20']:.4f})",
        "",
        "## Additional Analyses",
        f"- Frontier overlap is lowest for `{lowest_overlap_pair}` (mean overlap {overlap_summary.loc[lowest_overlap_pair, 'mean']:.4f}), showing that DIF-SR family models retrieve substantially different candidates than SASRec.",
        f"- `DIF-SR + metadata` improves accuracy but sharply reduces recommendation diversity (mean category entropy {diversity_summary.loc['DIF-SR + metadata', 'mean_category_entropy']:.4f} vs {diversity_summary.loc['DIF-SR', 'mean_category_entropy']:.4f} for DIF-SR).",
        f"- `DIF-SR + metadata` remains head-heavy; tail exposure drops from {difsr_tail:.4f} in DIF-SR to {difsr_meta_tail:.4f} after metadata injection.",
        f"- In high-entropy users, `DIF-SR` retains slightly better NDCG@20 than `DIF-SR + metadata`, suggesting multi-interest robustness mainly comes from the backbone.",
        "",
        "## Interpretation",
        "- Metadata consistently improves both backbones overall, but the gain is substantially larger on the DIF-SR backbone.",
        "- Metadata is most helpful for short-history users, where sequence-only signal is weakest.",
        "- DIF-SR changes the top-100 candidate frontier rather than merely reranking SASRec outputs, which supports the claim that multi-interest modeling changes candidate retrieval behavior.",
        "- The accuracy gain from metadata comes with lower diversity and stronger concentration on popular categories, so metadata is acting more like a relevance sharpener than a diversity enhancer in this setting.",
        "",
        "## Reproducibility",
        "Run the full research analysis pipeline with:",
        "",
        "```bash",
        "python experiments/run_evaluation.py",
        "```",
        "",
        "Artifacts are written to `data/metrics/research_study/` and plots are written to `plots/`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ensure_dirs()
    overall_frame = run_overall_metrics_analysis()
    records_bundle = load_all_model_records(top_k=100, force=False)
    overlap_result = run_candidate_overlap_analysis(records_bundle["records"])
    diversity_result = run_diversity_analysis(records_bundle["records"], records_bundle["item_meta"])
    popularity_result = run_popularity_bias_analysis(records_bundle["records"], records_bundle["popularity_bucket"])
    slice_result = run_slice_analysis(
        records_bundle["records"],
        diversity_result,
        popularity_result,
        overlap_result,
    )

    summary = build_research_summary(
        overall_frame,
        overlap_result,
        diversity_result,
        popularity_result,
        slice_result,
    )
    summary_path = Path(DATA_DIR) / "RESEARCH_SUMMARY.md"
    summary_path.write_text(summary)

    manifest = {
        "research_summary": str(summary_path),
        "candidate_overlap_summary": str(Path(DATA_DIR) / "candidate_overlap_summary.json"),
        "overall_metric_table": str(Path(DATA_DIR) / "overall_metric_table.json"),
        "diversity_summary": str(Path(DATA_DIR) / "diversity_summary.json"),
        "popularity_exposure_summary": str(Path(DATA_DIR) / "popularity_exposure_summary.json"),
        "slice_study": str(Path(DATA_DIR) / "slice_study.json"),
        "plots_dir": str(PLOTS_DIR),
    }
    write_json(Path(DATA_DIR) / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
