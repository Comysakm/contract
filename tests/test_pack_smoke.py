from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class PackSmokeTests(unittest.TestCase):
    def test_pack_runner_with_clustering_variants(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="formal_semisup_smoke_"))
        try:
            x_path = temp_dir / "trainx9.npy"
            y_path = temp_dir / "trainy9.npy"
            rng = np.random.default_rng(42)
            samples = []
            labels = []
            centers = [0.0, 5.0, 10.0]
            for class_id, center in enumerate(centers):
                for _ in range(320):
                    length = int(rng.integers(20, 39))
                    sample = np.zeros((length, 11), dtype=np.float32)
                    sample[:, :9] = rng.normal(loc=center, scale=0.5, size=(length, 9)).astype(np.float32)
                    sample[:, 9] = 0.0
                    sample[:, 10] = np.linspace(1, 365, num=length, dtype=np.float32)
                    samples.append(sample)
                    labels.append(class_id)
            np.save(x_path, np.array(samples, dtype=object), allow_pickle=True)
            np.save(y_path, np.asarray(labels, dtype=np.int64))
            pack_id = "smoke_pack"
            cmd = [
                sys.executable,
                "run_formal_semisup_pack.py",
                "--pack-id",
                pack_id,
                "--variants",
                "cop_kmeans,semi_supervised_spectral",
                "--data-x",
                str(x_path),
                "--data-y",
                str(y_path),
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)
            subprocess.run(cmd + ["--reuse-existing"], cwd=ROOT, check=True)
            server_root = ROOT / "runs" / "formal_semisup_pack" / pack_id
            server_dirs = [path for path in server_root.iterdir() if path.is_dir()]
            self.assertTrue(server_dirs)
            pack_root = server_dirs[0]
            self.assertTrue((pack_root / "summary.csv").exists())
            self.assertTrue((pack_root / "report.md").exists())
            summary = pd.read_csv(pack_root / "summary.csv")
            self.assertIn("variant", summary.columns)
            self.assertTrue({"cop_kmeans", "semi_supervised_spectral"}.issubset(set(summary["variant"].unique())))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
