from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from formal_semisup.reporting.aggregate import build_pack_report
from formal_semisup.utils.io import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    pack_root = Path(args.pack_root)
    manifest = load_json(args.manifest)
    experiment_dirs = [path for path in pack_root.iterdir() if path.is_dir() and path.name != "canonical"]
    build_pack_report(pack_root, manifest, experiment_dirs)


if __name__ == "__main__":
    main()
