from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class SDECTests(unittest.TestCase):
    def test_stacked_autoencoder_initializes(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch unavailable")
        try:
            from formal_semisup.methods.sdec import StackedAutoencoder
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"torch/sdec import unavailable: {exc}")
        model = StackedAutoencoder(input_dim=380, hidden_dims=[256, 128], latent_dim=32)
        self.assertEqual(model.model.cluster_centers.shape[-1], 32)


if __name__ == "__main__":
    unittest.main()
