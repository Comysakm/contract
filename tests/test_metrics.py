from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from formal_semisup.evaluation.metrics import classification_metrics, clustering_with_semantic_mapping


class MetricsTests(unittest.TestCase):
    def test_macro_only_uses_present_classes(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        y_pred = np.array([0, 1, 1, 1], dtype=np.int64)
        metrics = classification_metrics(y_true, y_pred, num_classes=16)
        self.assertEqual(metrics["present_classes"], [0, 1])
        self.assertEqual(len(metrics["per_class"]), 2)

    def test_hungarian_mapping_returns_semantic_metrics(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        y_cluster = np.array([1, 1, 0, 0], dtype=np.int64)
        summary = clustering_with_semantic_mapping(y_true, y_cluster, num_classes=16, features=np.eye(4, dtype=np.float32))
        self.assertAlmostEqual(summary["mapped_semantic_metrics"]["OA"], 1.0, places=6)
        self.assertIn("NMI", summary["clustering_metrics"])


if __name__ == "__main__":
    unittest.main()
