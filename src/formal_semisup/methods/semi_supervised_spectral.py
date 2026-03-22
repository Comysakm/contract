from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import NearestNeighbors

from formal_semisup.data.dataset import CanonicalDataset
from formal_semisup.evaluation.metrics import clustering_with_semantic_mapping
from formal_semisup.methods.common import copy_canonical_artifacts, select_best_candidate, write_experiment_payload
from formal_semisup.utils.io import ensure_dir, save_json
from formal_semisup.utils.repro import set_global_seed


def _build_knn_affinity(x_train: np.ndarray, k_neighbors: int) -> tuple[sparse.csr_matrix, float, NearestNeighbors]:
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(x_train)), metric="euclidean")
    nn.fit(x_train)
    distances, neighbors = nn.kneighbors(x_train)
    sigma = float(np.median(distances[:, 1:])) if distances.shape[1] > 1 else 1.0
    sigma = sigma if sigma > 1e-8 else 1.0
    rows = []
    cols = []
    values = []
    for i in range(len(x_train)):
        for dist, j in zip(distances[i, 1:], neighbors[i, 1:]):
            weight = float(np.exp(-((dist ** 2) / (2.0 * sigma ** 2))))
            rows.append(i)
            cols.append(int(j))
            values.append(weight)
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(len(x_train), len(x_train)))
    matrix = matrix.maximum(matrix.transpose()).tocsr()
    return matrix, sigma, nn


def _inject_constraints(
    affinity: sparse.csr_matrix,
    train_indices: np.ndarray,
    pairwise_constraints: dict[str, Any],
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    idx_to_local = {int(idx): int(pos) for pos, idx in enumerate(train_indices)}
    work = affinity.tolil(copy=True)
    must_added = 0
    cannot_zeroed = 0
    for pair in pairwise_constraints["must_link"]:
        i = idx_to_local.get(int(pair["i"]))
        j = idx_to_local.get(int(pair["j"]))
        if i is None or j is None:
            continue
        work[i, j] = 1.0
        work[j, i] = 1.0
        must_added += 1
    for pair in pairwise_constraints["cannot_link"]:
        i = idx_to_local.get(int(pair["i"]))
        j = idx_to_local.get(int(pair["j"]))
        if i is None or j is None:
            continue
        work[i, j] = 0.0
        work[j, i] = 0.0
        cannot_zeroed += 1
    work.setdiag(1.0)
    return work.tocsr(), {"must_link_injected": must_added, "cannot_link_zeroed": cannot_zeroed}


def _extrapolate_embedding(x_new: np.ndarray, x_train: np.ndarray, train_embedding: np.ndarray, nn: NearestNeighbors, sigma: float) -> np.ndarray:
    distances, neighbors = nn.kneighbors(x_new)
    weights = np.exp(-((distances ** 2) / (2.0 * sigma ** 2)))
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_sum = np.where(weights_sum <= 1e-8, 1.0, weights_sum)
    normalized = weights / weights_sum
    output = np.zeros((len(x_new), train_embedding.shape[1]), dtype=np.float32)
    for i in range(len(x_new)):
        output[i] = (normalized[i][:, None] * train_embedding[neighbors[i]]).sum(axis=0)
    return output


def run_semi_supervised_spectral(
    *,
    dataset: CanonicalDataset,
    config: dict[str, Any],
    exp_dir: str | Path,
) -> dict[str, Any]:
    set_global_seed(config["protocol"]["split_seed"])
    exp_path = Path(exp_dir)
    ensure_dir(exp_path / "checkpoints")
    ensure_dir(exp_path / "logs")
    copy_canonical_artifacts(exp_path.parent / "canonical", exp_path)
    train = dataset.split_arrays("train")
    val = dataset.split_arrays("val")
    test = dataset.split_arrays("test")
    cfg = config["semi_supervised_spectral"]
    affinity, sigma, nn = _build_knn_affinity(train["x_flat"], cfg["k_neighbors"])
    affinity, inject_summary = _inject_constraints(affinity, train["indices"], dataset.pairwise_constraints)
    graph_components, component_labels = connected_components(affinity)
    train_embedding = spectral_embedding(
        affinity,
        n_components=cfg["n_clusters"],
        random_state=cfg["random_seed"],
        drop_first=False,
        norm_laplacian=True,
    ).astype(np.float32)
    candidates = []
    for init_id in range(cfg["kmeans_n_init"]):
        kmeans = KMeans(n_clusters=cfg["n_clusters"], n_init=1, random_state=cfg["random_seed"] + init_id)
        train_assignments = kmeans.fit_predict(train_embedding)
        val_embedding = _extrapolate_embedding(val["x_flat"], train["x_flat"], train_embedding, nn, sigma)
        val_assignments = kmeans.predict(val_embedding)
        val_eval = clustering_with_semantic_mapping(val["y"], val_assignments, num_classes=config["data"]["num_classes"], features=val_embedding)
        candidates.append(
            {
                "init_id": init_id,
                "kmeans": kmeans,
                "train_assignments": train_assignments,
                "val_embedding": val_embedding,
                "val_assignments": val_assignments,
                "val_mapped_MA": float(val_eval["mapped_semantic_metrics"]["MA"]),
                "val_NMI": float(val_eval["clustering_metrics"]["NMI"]),
                "val_ARI": float(val_eval["clustering_metrics"]["ARI"]),
            }
        )
    best = select_best_candidate(candidates, "val_mapped_MA", ["val_NMI", "val_ARI"])
    kmeans = best["kmeans"]
    train_assignments = best["train_assignments"]
    val_embedding = best["val_embedding"]
    val_assignments = best["val_assignments"]
    test_embedding = _extrapolate_embedding(test["x_flat"], train["x_flat"], train_embedding, nn, sigma)
    test_assignments = kmeans.predict(test_embedding)
    train_eval = clustering_with_semantic_mapping(train["y"], train_assignments, num_classes=config["data"]["num_classes"], features=train_embedding)
    val_eval = clustering_with_semantic_mapping(val["y"], val_assignments, num_classes=config["data"]["num_classes"], features=val_embedding)
    test_eval = clustering_with_semantic_mapping(test["y"], test_assignments, num_classes=config["data"]["num_classes"], features=test_embedding)
    np.save(exp_path / "checkpoints" / "train_embedding.npy", train_embedding)
    np.save(exp_path / "checkpoints" / "cluster_centers.npy", kmeans.cluster_centers_)
    save_json(exp_path / "graph_summary.json", {"nodes": int(affinity.shape[0]), "edges": int(affinity.nnz), "connected_components": int(graph_components), "sigma": sigma, **inject_summary})
    save_json(exp_path / "mapping.json", {"val": val_eval["mapping"], "test": test_eval["mapping"]})
    eval_summary = {
        "variant": "semi_supervised_spectral",
        "evaluation_type": "clustering",
        "splits": {
            "train": {"evaluation_type": "clustering", **train_eval},
            "val": {"evaluation_type": "clustering", **val_eval},
            "test": {"evaluation_type": "clustering", **test_eval},
        },
    }
    train_summary = {
        "selection_rule": {"primary": "val_mapped_MA", "tie_break": ["val_NMI", "val_ARI"]},
        "graph_summary": {"nodes": int(affinity.shape[0]), "edges": int(affinity.nnz), "connected_components": int(graph_components)},
        "constraint_injection": inject_summary,
        "best_init_id": int(best["init_id"]),
    }
    write_experiment_payload(
        exp_path,
        resolved_config={"variant": "semi_supervised_spectral", "semi_supervised_spectral": cfg, "protocol": config["protocol"], "data": config["data"]},
        train_summary=train_summary,
        eval_summary=eval_summary,
    )
    return eval_summary
