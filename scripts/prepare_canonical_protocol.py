from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from formal_semisup.data.protocol import (
    apply_normalization,
    compute_normalization_stats,
    create_canonical_split,
    create_label_subset,
    create_pairwise_constraints,
    load_npy_inputs,
    save_canonical_protocol,
    stack_canonical_samples,
    validate_raw_samples,
)
from formal_semisup.utils.config import load_config
from formal_semisup.utils.io import ensure_dir, save_json


def prepare_canonical_protocol(config_path: str | Path, output_dir: str | Path, x_path: str | Path, y_path: str | Path) -> dict:
    config = load_config(config_path)
    data_cfg = config["data"]
    protocol_cfg = config["protocol"]
    constraints_cfg = config["constraints"]
    x_data, y_data = load_npy_inputs(x_path, y_path)
    valid_x, valid_y, invalid_samples = validate_raw_samples(x_data, y_data, data_cfg["spectral_dim"])
    stacked = stack_canonical_samples(valid_x, seq_len=data_cfg["seq_len"], spectral_dim=data_cfg["spectral_dim"])
    split_manifest = create_canonical_split(
        len(valid_y),
        protocol_cfg["train_ratio"],
        protocol_cfg["val_ratio"],
        protocol_cfg["split_seed"],
    )
    normalization_stats = compute_normalization_stats(stacked["x_spec"], stacked["mask"], split_manifest["train"])
    normalized_spec = apply_normalization(
        stacked["x_spec"],
        stacked["mask"],
        normalization_stats,
        protocol_cfg["clip_min"],
        protocol_cfg["clip_max"],
    )
    stacked["x_spec"] = normalized_spec
    stacked["x_seq"] = stacked["x_seq"].copy()
    stacked["x_seq"][:, :, : data_cfg["spectral_dim"]] = normalized_spec
    label_subset = create_label_subset(split_manifest["train"], valid_y, protocol_cfg["label_fraction"], protocol_cfg["subset_seed"])
    pairwise_constraints = create_pairwise_constraints(
        label_subset["selected_indices"],
        valid_y,
        protocol_cfg["constraint_seed"],
        must_link_multiplier=constraints_cfg["must_link_multiplier"],
        cannot_link_multiplier=constraints_cfg["cannot_link_multiplier"],
    )
    canonical_dir = ensure_dir(output_dir)
    artifacts = save_canonical_protocol(
        canonical_dir=canonical_dir,
        split_manifest=split_manifest,
        normalization_stats=normalization_stats,
        label_subset=label_subset,
        pairwise_constraints=pairwise_constraints,
        invalid_samples=invalid_samples,
    )
    npz_path = canonical_dir / "canonical_data.npz"
    np.savez_compressed(
        npz_path,
        x_spec=stacked["x_spec"],
        x_doy=stacked["x_doy"],
        x_seq=stacked["x_seq"],
        mask=stacked["mask"],
        y=valid_y,
    )
    save_json(
        canonical_dir / "canonical_manifest.json",
        {
            "canonical_data_npz": str(npz_path),
            "artifacts": {
                "split_manifest": str(artifacts.split_manifest_path),
                "normalization_stats": str(artifacts.normalization_stats_path),
                "label_subset_csv": str(artifacts.label_subset_csv_path),
                "label_subset_json": str(artifacts.label_subset_json_path),
                "pairwise_constraints": str(artifacts.pairwise_constraints_path),
                "invalid_samples": str(artifacts.invalid_samples_path),
            },
        },
    )
    return {
        "canonical_dir": str(canonical_dir),
        "canonical_data_npz": str(npz_path),
        "split_manifest": split_manifest,
        "normalization_stats": normalization_stats,
        "label_subset": label_subset,
        "pairwise_constraints": pairwise_constraints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "formal_semisup_pack.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-x", required=True)
    parser.add_argument("--data-y", required=True)
    args = parser.parse_args()
    prepare_canonical_protocol(args.config, args.output_dir, args.data_x, args.data_y)


if __name__ == "__main__":
    main()
