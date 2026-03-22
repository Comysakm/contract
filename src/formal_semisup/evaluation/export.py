from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from formal_semisup.utils.io import load_json, save_text


def export_eval_summary(eval_summary_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    payload = load_json(eval_summary_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    md_lines = [f"# Evaluation Report: {payload['variant']}", ""]
    for split, summary in payload["splits"].items():
        md_lines.append(f"## {split}")
        md_lines.append("")
        rows.append({"split": split, "evaluation_type": summary.get("evaluation_type")})
        if "classification_metrics" in summary:
            for key, value in summary["classification_metrics"].items():
                if isinstance(value, (int, float)) or value is None:
                    rows[-1][key] = value
            md_lines.append("- Classification metrics exported from classifier-head evaluation.")
        if "clustering_metrics" in summary:
            for key, value in summary["clustering_metrics"].items():
                rows[-1][f"cluster_{key}"] = value
            for key, value in summary["mapped_semantic_metrics"].items():
                if isinstance(value, (int, float)) or value is None:
                    rows[-1][f"mapped_{key}"] = value
            md_lines.append("- Clustering metrics and mapped semantic metrics exported.")
        md_lines.append("")
    df = pd.DataFrame(rows)
    csv_path = output_path / "eval_summary_flat.csv"
    md_path = output_path / "eval_report.md"
    df.to_csv(csv_path, index=False)
    save_text(md_path, "\n".join(md_lines))
    return {"csv": str(csv_path), "markdown": str(md_path)}
