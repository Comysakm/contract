from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as sk_metrics

from formal_semisup.evaluation.semantic_mapping import hungarian_cluster_mapping


def _present_labels(y_true: np.ndarray) -> list[int]:
    return [int(label) for label in sorted(np.unique(y_true))]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    present = _present_labels(y_true)
    labels_full = list(range(num_classes))
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=labels_full)
    precision, recall, f1, support = sk_metrics.precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=present,
        zero_division=0,
    )
    oa = float(sk_metrics.accuracy_score(y_true, y_pred))
    mean_accuracy = float(np.mean(recall)) if len(recall) else 0.0
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    per_class_iou = []
    per_class_rows = []
    for idx, label in enumerate(present):
        tp = cm[label, label]
        fp = cm[:, label].sum() - tp
        fn = cm[label, :].sum() - tp
        denom = tp + fp + fn
        iou = float(tp / denom) if denom > 0 else 0.0
        per_class_iou.append(iou)
        per_class_rows.append(
            {
                "class_id": int(label),
                "support": int(support[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "iou": float(iou),
            }
        )
    return {
        "OA": oa,
        "ACC": oa,
        "MA": mean_accuracy,
        "Macro-F1": macro_f1,
        "mIoU": float(np.mean(per_class_iou)) if per_class_iou else 0.0,
        "precision_macro": float(np.mean(precision)) if len(precision) else 0.0,
        "recall_macro": float(np.mean(recall)) if len(recall) else 0.0,
        "present_classes": present,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_rows,
    }


def purity_score(y_true: np.ndarray, y_pred_cluster: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred_cluster = np.asarray(y_pred_cluster)
    total = len(y_true)
    if total == 0:
        return 0.0
    purity_hits = 0
    for cluster_id in np.unique(y_pred_cluster):
        members = y_true[y_pred_cluster == cluster_id]
        if len(members) == 0:
            continue
        _, counts = np.unique(members, return_counts=True)
        purity_hits += int(counts.max())
    return float(purity_hits / total)


def within_cluster_variance(features: np.ndarray, assignments: np.ndarray) -> float | None:
    features = np.asarray(features, dtype=np.float64)
    assignments = np.asarray(assignments, dtype=np.int64)
    if len(np.unique(assignments)) <= 1:
        return None
    total = 0.0
    count = 0
    for cluster_id in np.unique(assignments):
        cluster_features = features[assignments == cluster_id]
        if len(cluster_features) == 0:
            continue
        center = cluster_features.mean(axis=0, keepdims=True)
        distances = ((cluster_features - center) ** 2).sum(axis=1)
        total += float(distances.sum())
        count += int(len(cluster_features))
    return float(total / max(count, 1))


def clustering_metrics(y_true: np.ndarray, y_pred_cluster: np.ndarray, *, features: np.ndarray | None = None) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_cluster = np.asarray(y_pred_cluster, dtype=np.int64)
    metrics = {
        "purity": purity_score(y_true, y_pred_cluster),
        "NMI": float(sk_metrics.normalized_mutual_info_score(y_true, y_pred_cluster)),
        "ARI": float(sk_metrics.adjusted_rand_score(y_true, y_pred_cluster)),
        "silhouette": None,
        "within_cluster_variance": None,
    }
    if features is not None and len(np.unique(y_pred_cluster)) > 1 and len(y_pred_cluster) > len(np.unique(y_pred_cluster)):
        try:
            metrics["silhouette"] = float(sk_metrics.silhouette_score(features, y_pred_cluster))
        except Exception:
            metrics["silhouette"] = None
        metrics["within_cluster_variance"] = within_cluster_variance(features, y_pred_cluster)
    return metrics


def clustering_with_semantic_mapping(
    y_true: np.ndarray,
    y_pred_cluster: np.ndarray,
    *,
    num_classes: int,
    features: np.ndarray | None = None,
) -> dict[str, Any]:
    mapping_info = hungarian_cluster_mapping(y_true, y_pred_cluster)
    mapped = mapping_info["mapped_predictions"]
    return {
        "clustering_metrics": clustering_metrics(y_true, y_pred_cluster, features=features),
        "mapped_semantic_metrics": classification_metrics(y_true, mapped, num_classes=num_classes),
        "mapping": mapping_info["mapping"],
    }
