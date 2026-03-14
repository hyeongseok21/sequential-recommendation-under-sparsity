from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    uses_metadata: bool
    config_path: Path
    checkpoint_name: str
    user_metrics_path: Path


SASREC_MODELS = [
    ModelSpec(
        name="SASRec",
        family="sasrec",
        uses_metadata=False,
        config_path=REPO_ROOT / "hm_refactored/configs/config.closure_sasrec.json",
        checkpoint_name="hm_closure_sasrec",
        user_metrics_path=REPO_ROOT / "data/metrics/closure/hm_closure_sasrec_user_metrics.json",
    ),
    ModelSpec(
        name="SASRec + metadata",
        family="sasrec",
        uses_metadata=True,
        config_path=REPO_ROOT / "hm_refactored/configs/config.closure_sasrec_meta.json",
        checkpoint_name="hm_closure_sasrec_meta",
        user_metrics_path=REPO_ROOT / "data/metrics/closure/hm_closure_sasrec_meta_user_metrics.json",
    ),
]
