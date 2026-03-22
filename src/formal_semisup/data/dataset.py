from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from formal_semisup.utils.io import load_json


@dataclass
class CanonicalDataset:
    x_spec: np.ndarray
    x_doy: np.ndarray
    x_seq: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    split_manifest: dict[str, list[int]]
    normalization_stats: dict[str, Any]
    label_subset: dict[str, Any]
    pairwise_constraints: dict[str, Any]

    @property
    def flattened_x(self) -> np.ndarray:
        return self.x_seq.reshape(self.x_seq.shape[0], -1).astype(np.float32)

    def split_arrays(self, split: str) -> dict[str, np.ndarray]:
        indices = np.asarray(self.split_manifest[split], dtype=np.int64)
        return {
            "indices": indices,
            "x_spec": self.x_spec[indices],
            "x_doy": self.x_doy[indices],
            "x_seq": self.x_seq[indices],
            "mask": self.mask[indices],
            "y": self.y[indices],
            "x_flat": self.flattened_x[indices],
        }

    def labeled_train_indices(self) -> list[int]:
        return [int(row["index"]) for row in self.label_subset["selected_rows"]]


def load_canonical_dataset(canonical_dir: str | Path) -> CanonicalDataset:
    canonical_path = Path(canonical_dir)
    manifest = load_json(canonical_path / "canonical_manifest.json")
    npz = np.load(manifest["canonical_data_npz"], allow_pickle=True)
    split_manifest = load_json(canonical_path / "split_manifest.json")
    normalization_stats = load_json(canonical_path / "normalization_stats.json")
    label_subset = load_json(canonical_path / "label_subset.json")
    pairwise_constraints = load_json(canonical_path / "pairwise_constraints.json")
    return CanonicalDataset(
        x_spec=np.asarray(npz["x_spec"], dtype=np.float32),
        x_doy=np.asarray(npz["x_doy"], dtype=np.float32),
        x_seq=np.asarray(npz["x_seq"], dtype=np.float32),
        mask=np.asarray(npz["mask"], dtype=bool),
        y=np.asarray(npz["y"], dtype=np.int64),
        split_manifest=split_manifest,
        normalization_stats=normalization_stats,
        label_subset=label_subset,
        pairwise_constraints=pairwise_constraints,
    )
