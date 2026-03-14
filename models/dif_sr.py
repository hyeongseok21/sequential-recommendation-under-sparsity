from pathlib import Path

from .sasrec import ModelSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


DIFSR_MODELS = [
    ModelSpec(
        name="DIF-SR",
        family="difsr",
        uses_metadata=False,
        config_path=REPO_ROOT / "hm_refactored/configs/config.closure_difsr.json",
        checkpoint_name="hm_closure_difsr",
        user_metrics_path=REPO_ROOT / "data/metrics/closure/hm_closure_difsr_user_metrics.json",
    ),
    ModelSpec(
        name="DIF-SR + metadata",
        family="difsr",
        uses_metadata=True,
        config_path=REPO_ROOT / "hm_refactored/configs/config.closure_difsr_meta.json",
        checkpoint_name="hm_closure_difsr_meta",
        user_metrics_path=REPO_ROOT / "data/metrics/closure/hm_closure_difsr_meta_user_metrics.json",
    ),
]
