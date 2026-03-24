from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from formal_semisup.evaluation.export import export_eval_summary
from formal_semisup.utils.io import ensure_dir, save_csv_rows


def _plot_confusion_matrix(path: Path, matrix: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _collect_histories(node: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    if isinstance(node, dict):
        history = node.get("history")
        if isinstance(history, list) and history and all(isinstance(item, dict) for item in history):
            yield ("__".join(prefix) or "train"), history
        for key, value in node.items():
            if key == "history":
                continue
            yield from _collect_histories(value, prefix + (str(key),))


def _plot_history(path: Path, df: pd.DataFrame, title: str) -> None:
    numeric_cols = [col for col in df.columns if col != "epoch" and pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        return
    x = df["epoch"] if "epoch" in df.columns else np.arange(len(df))
    plt.figure(figsize=(9, 5))
    for col in numeric_cols:
        plt.plot(x, df[col], label=col)
    plt.title(title)
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def export_paper_artifacts(exp_dir: str | Path, train_summary: dict[str, Any], eval_summary: dict[str, Any]) -> dict[str, Any]:
    exp_path = Path(exp_dir)
    tables_dir = ensure_dir(exp_path / "tables")
    figures_dir = ensure_dir(exp_path / "figures")
    eval_paths = export_eval_summary(exp_path / "eval_summary.json", tables_dir)
    metric_rows: list[dict[str, Any]] = []
    for split, summary in eval_summary["splits"].items():
        row = {"split": split, "evaluation_type": summary.get("evaluation_type")}
        if "classification_metrics" in summary:
            metrics = summary["classification_metrics"]
            for key in ["OA", "ACC", "MA", "Macro-F1", "mIoU", "precision_macro", "recall_macro"]:
                row[key] = metrics.get(key)
            save_csv_rows(tables_dir / f"per_class_{split}.csv", metrics.get("per_class", []))
            cm = np.asarray(metrics.get("confusion_matrix", []), dtype=np.int64)
            pd.DataFrame(cm).to_csv(tables_dir / f"confusion_matrix_{split}.csv", index=False)
            _plot_confusion_matrix(figures_dir / f"confusion_matrix_{split}.png", cm, f"{split} confusion matrix")
        if "clustering_metrics" in summary:
            clustering = summary["clustering_metrics"]
            for key, value in clustering.items():
                row[f"cluster_{key}"] = value
            mapped = summary["mapped_semantic_metrics"]
            for key in ["OA", "ACC", "MA", "Macro-F1", "mIoU", "precision_macro", "recall_macro"]:
                row[f"mapped_{key}"] = mapped.get(key)
            save_csv_rows(tables_dir / f"mapped_per_class_{split}.csv", mapped.get("per_class", []))
            cm = np.asarray(mapped.get("confusion_matrix", []), dtype=np.int64)
            pd.DataFrame(cm).to_csv(tables_dir / f"mapped_confusion_matrix_{split}.csv", index=False)
            _plot_confusion_matrix(figures_dir / f"mapped_confusion_matrix_{split}.png", cm, f"{split} mapped confusion matrix")
            mapping_rows = [{"cluster_id": int(key), "class_id": int(value)} for key, value in summary.get("mapping", {}).items()]
            save_csv_rows(tables_dir / f"mapping_{split}.csv", mapping_rows)
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(tables_dir / "split_metrics.csv", index=False)

    history_exports = []
    for stage_name, history in _collect_histories(train_summary):
        df = pd.DataFrame(history)
        csv_path = tables_dir / f"train_history_{stage_name}.csv"
        fig_path = figures_dir / f"train_curve_{stage_name}.png"
        df.to_csv(csv_path, index=False)
        _plot_history(fig_path, df, f"{stage_name} history")
        history_exports.append({"stage": stage_name, "csv": str(csv_path), "figure": str(fig_path)})
    return {
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "eval_exports": eval_paths,
        "history_exports": history_exports,
    }
