from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from formal_semisup.data.dataset import load_canonical_dataset
from formal_semisup.methods.semi_supervised_spectral import run_semi_supervised_spectral
from formal_semisup.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "formal_semisup_pack.yaml"))
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--exp-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = load_canonical_dataset(args.canonical_dir)
    run_semi_supervised_spectral(dataset=dataset, config=config, exp_dir=args.exp_dir)


if __name__ == "__main__":
    main()
