from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from formal_semisup.data.protocol import (
    apply_normalization,
    canonicalize_sample,
    compute_normalization_stats,
    create_label_subset,
    create_pairwise_constraints,
)


class ProtocolTests(unittest.TestCase):
    def test_canonicalize_sample_with_cloud_and_doy(self):
        sample = np.zeros((5, 11), dtype=np.float32)
        sample[:, :9] = np.arange(45, dtype=np.float32).reshape(5, 9)
        sample[:, 9] = [0.0, 1.0, 0.0, 0.0, 0.0]
        sample[:, 10] = [10, 20, 30, 40, 50]
        item = canonicalize_sample(sample, seq_len=8, spectral_dim=9)
        self.assertEqual(item["x_spec"].shape, (8, 9))
        self.assertEqual(item["x_doy"].shape, (8, 1))
        self.assertEqual(item["mask"].shape, (8,))
        self.assertTrue(item["mask"][1])
        self.assertTrue(item["mask"][5])
        self.assertAlmostEqual(float(item["x_doy"][0, 0]), 10 / 365.0, places=6)

    def test_train_only_normalization_masks_invalid(self):
        x_spec = np.array(
            [
                [[1.0] * 9, [2.0] * 9],
                [[10.0] * 9, [20.0] * 9],
            ],
            dtype=np.float32,
        )
        mask = np.array([[False, True], [False, False]], dtype=bool)
        stats = compute_normalization_stats(x_spec, mask, [0])
        normalized = apply_normalization(x_spec, mask, stats, -3.0, 3.0)
        self.assertTrue(np.all(normalized[0, 1] == 0.0))
        self.assertTrue(np.allclose(normalized[0, 0], 0.0))

    def test_label_subset_exact_allocation_and_constraints(self):
        y = np.array([0] * 50 + [1] * 30 + [2] * 20, dtype=np.int64)
        train_indices = list(range(len(y)))
        subset = create_label_subset(train_indices, y, 0.10, 42)
        self.assertEqual(subset["subset_size_actual"], 10)
        sampled_classes = {row["label"] for row in subset["selected_rows"]}
        self.assertTrue({0, 1, 2}.issubset(sampled_classes))
        constraints = create_pairwise_constraints(
            subset["selected_indices"],
            y,
            42,
            must_link_multiplier=4,
            cannot_link_multiplier=4,
        )
        self.assertIn("must_link", constraints)
        self.assertIn("cannot_link", constraints)
        self.assertEqual(constraints["conflict_count"], 0)


if __name__ == "__main__":
    unittest.main()
