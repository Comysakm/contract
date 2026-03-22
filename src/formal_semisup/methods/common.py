from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from formal_semisup.utils.io import ensure_dir, save_json


def copy_canonical_artifacts(canonical_dir: str | Path, exp_dir: str | Path) -> None:
    exp_path = Path(exp_dir)
    canonical_path = Path(canonical_dir)
    for name in [
        "split_manifest.json",
        "normalization_stats.json",
        "label_subset.csv",
        "label_subset.json",
        "pairwise_constraints.json",
    ]:
        source = canonical_path / name
        if source.exists():
            shutil.copy2(source, exp_path / name)


def metric_value(summary: dict[str, Any], key: str) -> float:
    if key in summary:
        return float(summary[key])
    raise KeyError(f"metric '{key}' not found in summary")


def select_best_candidate(candidates: list[dict[str, Any]], primary_key: str, tie_break_keys: list[str]) -> dict[str, Any]:
    def score(item: dict[str, Any]) -> tuple[float, ...]:
        values = [metric_value(item, primary_key)]
        values.extend(metric_value(item, key) for key in tie_break_keys)
        return tuple(values)

    return sorted(candidates, key=score, reverse=True)[0]


def write_experiment_payload(
    exp_dir: str | Path,
    *,
    resolved_config: dict[str, Any],
    train_summary: dict[str, Any],
    eval_summary: dict[str, Any],
) -> None:
    exp_path = ensure_dir(exp_dir)
    logs_path = ensure_dir(exp_path / "logs")
    save_json(exp_path / "resolved_config.json", resolved_config)
    save_json(exp_path / "train_summary.json", train_summary)
    save_json(exp_path / "eval_summary.json", eval_summary)
    save_json(logs_path / "payload_snapshot.json", {"resolved_config": resolved_config, "train_summary": train_summary, "eval_summary": eval_summary})


def save_numpy_predictions(path: str | Path, values: np.ndarray) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_obj, values)
