from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_cluster_mapping(y_true: np.ndarray, y_pred_cluster: np.ndarray) -> dict[str, Any]:
    true_labels = np.asarray(sorted(np.unique(y_true)), dtype=np.int64)
    cluster_labels = np.asarray(sorted(np.unique(y_pred_cluster)), dtype=np.int64)
    cost = np.zeros((len(cluster_labels), len(true_labels)), dtype=np.int64)
    for row, cluster_id in enumerate(cluster_labels):
        members = y_true[y_pred_cluster == cluster_id]
        for col, true_id in enumerate(true_labels):
            cost[row, col] = -int(np.sum(members == true_id))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {int(cluster_labels[row]): int(true_labels[col]) for row, col in zip(row_ind, col_ind)}
    mapped = np.asarray([mapping.get(int(cluster), int(true_labels[0])) for cluster in y_pred_cluster], dtype=np.int64)
    return {
        "mapping": mapping,
        "mapped_predictions": mapped,
        "cluster_labels": cluster_labels.tolist(),
        "true_labels": true_labels.tolist(),
    }
