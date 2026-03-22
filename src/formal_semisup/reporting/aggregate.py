from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from formal_semisup.utils.io import ensure_dir, save_json, save_text


TABLE1_VARIANTS = {
    "supervised_lstm",
    "supervised_rnn",
    "supervised_gru",
    "supervised_transformer",
    "cae_pretrain_classifier",
}

TABLE2_VARIANTS = {"cop_kmeans", "semi_supervised_spectral", "sdec"}


def _flatten_summary_row(method: str, split: str, eval_summary: dict[str, Any]) -> dict[str, Any]:
    row = {"variant": method, "split": split, "evaluation_type": eval_summary.get("evaluation_type")}
    if "classification_metrics" in eval_summary:
        for key, value in eval_summary["classification_metrics"].items():
            if isinstance(value, (int, float)) or value is None:
                row[key] = value
    if "clustering_metrics" in eval_summary:
        for key, value in eval_summary["clustering_metrics"].items():
            row[f"cluster_{key}"] = value
    if "mapped_semantic_metrics" in eval_summary:
        for key, value in eval_summary["mapped_semantic_metrics"].items():
            if isinstance(value, (int, float)) or value is None:
                row[f"mapped_{key}"] = value
    return row


def build_pack_report(pack_root: str | Path, manifest: dict[str, Any], experiment_dirs: list[Path]) -> dict[str, Any]:
    pack_path = ensure_dir(pack_root)
    rows: list[dict[str, Any]] = []
    report_lines = [
        "# Formal Semi-Supervised Pack Report",
        "",
        "## Protocol",
        "",
        "- full-data / unfiltered",
        "- clean_cloud_threshold = 1.0",
        "- min_valid_steps = 0",
        "- split_seed = 42",
        "- label_fraction = 0.01",
        "- strict_label_training_only = 1",
        "",
        "## Notes",
        "",
        "- Cop-KMeans and semi-supervised spectral clustering are implemented according to classic public definitions.",
        "- SDEC follows `paper/ssdec.pdf` with the minimum adaptation of flattening fixed-length `[38, 10]` time series to 380-dimensional vectors before model input.",
        "- `cae_pretrain_classifier` is reported as an unsupervised pretraining plus low-label fine-tuning supplementary supervised baseline.",
        "",
        "## Experiments",
        "",
    ]
    completed_variants = []
    for exp_dir in sorted(experiment_dirs):
        eval_summary_path = exp_dir / "eval_summary.json"
        if not eval_summary_path.exists():
            continue
        with eval_summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        method = payload["variant"]
        completed_variants.append(method)
        report_lines.append(f"- `{method}`: completed")
        for split, split_summary in payload["splits"].items():
            rows.append(_flatten_summary_row(method, split, split_summary))
    summary_df = pd.DataFrame(rows)
    summary_path = pack_path / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    if summary_df.empty:
        table1_df = pd.DataFrame()
        table2_df = pd.DataFrame()
    else:
        table1_df = summary_df[(summary_df["variant"].isin(TABLE1_VARIANTS)) & (summary_df["split"] == "test")].copy()
        table2_df = summary_df[(summary_df["variant"].isin(TABLE2_VARIANTS)) & (summary_df["split"] == "test")].copy()
    table1_cols = [col for col in ["variant", "OA", "MA", "Macro-F1", "mIoU"] if col in table1_df.columns]
    table1_path = pack_path / "summary_table1_supervised.csv"
    table1_df[table1_cols].to_csv(table1_path, index=False)
    table2_cols = [
        col
        for col in [
            "variant",
            "cluster_purity",
            "cluster_NMI",
            "cluster_ARI",
            "cluster_silhouette",
            "cluster_within_cluster_variance",
            "mapped_OA",
            "mapped_MA",
            "mapped_Macro-F1",
            "mapped_mIoU",
        ]
        if col in table2_df.columns
    ]
    table2_path = pack_path / "summary_table2_semisup_clustering.csv"
    table2_df[table2_cols].to_csv(table2_path, index=False)
    report_lines.extend(["", f"- Completed variants: {', '.join(sorted(completed_variants))}", ""])
    save_text(pack_path / "report.md", "\n".join(report_lines))
    save_json(pack_path / "manifest.json", manifest)
    return {
        "summary_csv": str(summary_path),
        "summary_table1": str(table1_path),
        "summary_table2": str(table2_path),
        "report_md": str(pack_path / "report.md"),
    }
