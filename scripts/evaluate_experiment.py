from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from formal_semisup.evaluation.export import export_eval_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export_eval_summary(args.eval_summary, args.output_dir)


if __name__ == "__main__":
    main()
