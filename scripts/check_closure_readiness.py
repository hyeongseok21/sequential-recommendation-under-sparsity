#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "SASRec": ROOT / "hm_refactored/configs/config.closure_sasrec.json",
    "SASRec + metadata": ROOT / "hm_refactored/configs/config.closure_sasrec_meta.json",
    "DIF-SR": ROOT / "hm_refactored/configs/config.closure_difsr.json",
    "DIF-SR + metadata": ROOT / "hm_refactored/configs/config.closure_difsr_meta.json",
}

SPLIT_KEYS = [
    "target_week",
    "train_data_name",
    "test_data_name",
    "count_high",
    "count_low",
    "recent_two_weeks",
    "remove_recent_bought",
    "remove_train_recent_bought",
]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []

    loaded: dict[str, dict] = {}
    for label, path in CONFIGS.items():
        if not path.exists():
            errors.append(f"missing config: {path}")
            continue
        try:
            loaded[label] = load_config(path)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid json: {path} ({exc})")

    if errors:
        print("Closure readiness: FAIL")
        for item in errors:
            print(f"- {item}")
        sys.exit(1)

    save_names = set()
    split_signature = None

    for label, cfg in loaded.items():
        dataset = cfg["dataset_params"]
        train = cfg["train_params"]

        if train.get("seed") != 42:
            errors.append(f"{label}: seed is not 42")
        if train.get("top_k") != 20:
            errors.append(f"{label}: top_k is not 20")

        save_name = train.get("save_name")
        if save_name in save_names:
            errors.append(f"{label}: duplicate save_name {save_name}")
        save_names.add(save_name)

        orig_path = ROOT / dataset["orig_path"]
        prep_path = ROOT / dataset["dataset_path"]
        if not orig_path.exists():
            errors.append(f"{label}: missing orig_path {orig_path}")
        if not prep_path.exists():
            warnings.append(f"{label}: prep_path does not exist yet {prep_path}")

        train_file = orig_path / dataset["train_data_name"]
        test_file = orig_path / dataset["test_data_name"]
        if not train_file.exists():
            errors.append(f"{label}: missing train file {train_file}")
        if not test_file.exists():
            errors.append(f"{label}: missing test file {test_file}")

        signature = tuple(dataset.get(key) for key in SPLIT_KEYS)
        if split_signature is None:
            split_signature = signature
        elif signature != split_signature:
            errors.append(f"{label}: split signature mismatch")

        if cfg["eval_params"].get("test") is not True or cfg["eval_params"].get("benchmark") is not True:
            errors.append(f"{label}: test/benchmark flags must both be true")

    sasrec_dataset = loaded["SASRec"]["dataset_params"]["save_name"]
    sasrec_meta_dataset = loaded["SASRec + metadata"]["dataset_params"]["save_name"]
    if sasrec_dataset == sasrec_meta_dataset:
        warnings.append("SASRec and SASRec + metadata share the same save_name; verify preprocessing intent.")
    else:
        warnings.append("SASRec and SASRec + metadata use different preprocessed save_name values; raw split parity will be used for fairness.")

    difsr_features = loaded["DIF-SR"]["model_params"].get("metadata_features", None)
    if difsr_features != []:
        errors.append("DIF-SR baseline must set metadata_features to []")

    if errors:
        print("Closure readiness: FAIL")
        for item in errors:
            print(f"- {item}")
        for item in warnings:
            print(f"! {item}")
        sys.exit(1)

    print("Closure readiness: PASS")
    print("Checked configs:")
    for label, path in CONFIGS.items():
        print(f"- {label}: {path}")
    for item in warnings:
        print(f"! {item}")


if __name__ == "__main__":
    main()
