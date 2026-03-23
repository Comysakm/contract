from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding

from formal_semisup.data.dataset import CanonicalDataset
from formal_semisup.evaluation.metrics import clustering_with_semantic_mapping
from formal_semisup.methods.common import copy_canonical_artifacts, select_best_candidate, write_experiment_payload
from formal_semisup.utils.faiss_utils import NeighborSearchIndex, build_neighbor_index
from formal_semisup.utils.io import ensure_dir, save_json
from formal_semisup.utils.repro import set_global_seed


def _torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for GPU spectral backend") from exc
    return torch


def _build_knn_affinity(x_train: np.ndarray, k_neighbors: int, performance_cfg: dict[str, Any]) -> tuple[sparse.csr_matrix, float, NeighborSearchIndex, dict[str, Any]]:
    n_query_neighbors = min(k_neighbors + 1, len(x_train))
    neighbor_index = build_neighbor_index(x_train, n_query_neighbors, performance_cfg)
    distances, neighbors = neighbor_index.kneighbors(x_train)
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
    return matrix, sigma, neighbor_index, neighbor_index.summary()


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


def _resolve_backend(device: str, performance_cfg: dict[str, Any]) -> str:
    mode = str(performance_cfg.get("spectral_backend", "auto")).lower()
    if mode in {"cpu", "gpu"}:
        return mode
    try:
        torch = _torch()
        return "gpu" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _cpu_spectral_embedding(affinity: sparse.csr_matrix, n_components: int, random_seed: int) -> np.ndarray:
    return spectral_embedding(
        affinity,
        n_components=n_components,
        random_state=random_seed,
        drop_first=False,
        norm_laplacian=True,
    ).astype(np.float32)


def _gpu_sparse_operator(affinity: sparse.csr_matrix, device: str):
    torch = _torch()
    coo = affinity.tocoo()
    degree = np.asarray(affinity.sum(axis=1)).reshape(-1).astype(np.float32)
    inv_sqrt_degree = 1.0 / np.sqrt(np.clip(degree, 1e-8, None))
    norm_values = coo.data.astype(np.float32) * inv_sqrt_degree[coo.row] * inv_sqrt_degree[coo.col]
    indices = np.vstack([coo.row, coo.col]).astype(np.int64)
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
    value_tensor = torch.as_tensor(norm_values, dtype=torch.float32, device=device)
    sparse_operator = torch.sparse_coo_tensor(index_tensor, value_tensor, size=coo.shape, device=device).coalesce()
    return sparse_operator


def _gpu_spectral_embedding(affinity: sparse.csr_matrix, n_components: int, random_seed: int, power_iters: int, device: str) -> np.ndarray:
    torch = _torch()
    sparse_operator = _gpu_sparse_operator(affinity, device)
    n_nodes = affinity.shape[0]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_seed))
    q = torch.randn((n_nodes, n_components), generator=generator, device=device, dtype=torch.float32)
    q = torch.linalg.qr(q, mode="reduced").Q
    for _ in range(int(power_iters)):
        z = torch.sparse.mm(sparse_operator, q)
        q = torch.linalg.qr(z, mode="reduced").Q
    rayleigh = q.T @ torch.sparse.mm(sparse_operator, q)
    eigvals, eigvecs = torch.linalg.eigh(rayleigh)
    order = torch.argsort(eigvals, descending=True)
    embedding = q @ eigvecs[:, order[:n_components]]
    return embedding.detach().cpu().numpy().astype(np.float32)


def _torch_kmeans(train_embedding: np.ndarray, n_clusters: int, n_init: int, max_iter: int, random_seed: int, device: str) -> list[dict[str, Any]]:
    torch = _torch()
    x = torch.as_tensor(train_embedding, dtype=torch.float32, device=device)
    x_norm = (x**2).sum(dim=1, keepdim=True)
    n_samples = x.shape[0]
    candidates: list[dict[str, Any]] = []
    for init_id in range(n_init):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(random_seed + init_id))
        initial = torch.randperm(n_samples, generator=generator, device=device)[:n_clusters]
        centers = x.index_select(0, initial).clone()
        previous_assignments = None
        for _ in range(int(max_iter)):
            center_norm = (centers**2).sum(dim=1).unsqueeze(0)
            distances = x_norm - 2.0 * (x @ centers.T) + center_norm
            assignments = torch.argmin(distances, dim=1)
            if previous_assignments is not None and torch.equal(assignments, previous_assignments):
                break
            previous_assignments = assignments
            new_centers = []
            for cluster_id in range(n_clusters):
                members = x[assignments == cluster_id]
                if members.shape[0] == 0:
                    fallback_idx = int(torch.randint(0, n_samples, (1,), generator=generator, device=device).item())
                    new_centers.append(x[fallback_idx])
                else:
                    new_centers.append(members.mean(dim=0))
            centers = torch.stack(new_centers, dim=0)
        center_norm = (centers**2).sum(dim=1).unsqueeze(0)
        distances = x_norm - 2.0 * (x @ centers.T) + center_norm
        assignments = torch.argmin(distances, dim=1)
        inertia = float(distances.gather(1, assignments.unsqueeze(1)).sum().item())
        candidates.append(
            {
                "init_id": init_id,
                "assignments": assignments.detach().cpu().numpy().astype(np.int64),
                "centers": centers.detach().cpu().numpy().astype(np.float32),
                "inertia": inertia,
            }
        )
    return candidates


def _predict_kmeans(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x_norm = (x**2).sum(axis=1, keepdims=True)
    center_norm = (centers**2).sum(axis=1, keepdims=True).T
    distances = x_norm - 2.0 * (x @ centers.T) + center_norm
    return np.argmin(distances, axis=1).astype(np.int64)


def _extrapolate_embedding(x_new: np.ndarray, train_embedding: np.ndarray, neighbor_index: NeighborSearchIndex, sigma: float) -> np.ndarray:
    distances, neighbors = neighbor_index.kneighbors(x_new)
    weights = np.exp(-((distances ** 2) / (2.0 * sigma ** 2)))
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_sum = np.where(weights_sum <= 1e-8, 1.0, weights_sum)
    normalized = weights / weights_sum
    gathered = train_embedding[neighbors]
    return np.einsum("nk,nkd->nd", normalized, gathered, optimize=True).astype(np.float32)


def run_semi_supervised_spectral(
    *,
    dataset: CanonicalDataset,
    config: dict[str, Any],
    exp_dir: str | Path,
    device: str,
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
    performance_cfg = config.get("performance", {})
    affinity, sigma, neighbor_index, neighbor_summary = _build_knn_affinity(train["x_flat"], cfg["k_neighbors"], performance_cfg)
    affinity, inject_summary = _inject_constraints(affinity, train["indices"], dataset.pairwise_constraints)
    graph_components, _ = connected_components(affinity)
    spectral_backend = _resolve_backend(device, performance_cfg)
    if spectral_backend == "gpu":
        train_embedding = _gpu_spectral_embedding(
            affinity,
            n_components=cfg["n_clusters"],
            random_seed=cfg["random_seed"],
            power_iters=performance_cfg.get("spectral_power_iters", 24),
            device=device,
        )
        kmeans_candidates = _torch_kmeans(
            train_embedding,
            n_clusters=cfg["n_clusters"],
            n_init=cfg["kmeans_n_init"],
            max_iter=performance_cfg.get("spectral_kmeans_max_iter", 100),
            random_seed=cfg["random_seed"],
            device=device,
        )
        clustering_backend = "torch-gpu"
    else:
        train_embedding = _cpu_spectral_embedding(affinity, n_components=cfg["n_clusters"], random_seed=cfg["random_seed"])
        kmeans_candidates = []
        for init_id in range(cfg["kmeans_n_init"]):
            kmeans = KMeans(n_clusters=cfg["n_clusters"], n_init=1, random_state=cfg["random_seed"] + init_id)
            assignments = kmeans.fit_predict(train_embedding)
            kmeans_candidates.append(
                {
                    "init_id": init_id,
                    "assignments": assignments.astype(np.int64),
                    "centers": kmeans.cluster_centers_.astype(np.float32),
                    "kmeans": kmeans,
                }
            )
        clustering_backend = "sklearn-cpu"

    candidates = []
    for candidate in kmeans_candidates:
        val_embedding = _extrapolate_embedding(val["x_flat"], train_embedding, neighbor_index, sigma)
        val_assignments = _predict_kmeans(val_embedding, candidate["centers"])
        val_eval = clustering_with_semantic_mapping(val["y"], val_assignments, num_classes=config["data"]["num_classes"], features=val_embedding)
        candidates.append(
            {
                "init_id": candidate["init_id"],
                "centers": candidate["centers"],
                "train_assignments": candidate["assignments"],
                "val_embedding": val_embedding,
                "val_assignments": val_assignments,
                "val_mapped_MA": float(val_eval["mapped_semantic_metrics"]["MA"]),
                "val_NMI": float(val_eval["clustering_metrics"]["NMI"]),
                "val_ARI": float(val_eval["clustering_metrics"]["ARI"]),
            }
        )
    best = select_best_candidate(candidates, "val_mapped_MA", ["val_NMI", "val_ARI"])
    centers = best["centers"]
    train_assignments = best["train_assignments"]
    val_embedding = best["val_embedding"]
    val_assignments = best["val_assignments"]
    test_embedding = _extrapolate_embedding(test["x_flat"], train_embedding, neighbor_index, sigma)
    test_assignments = _predict_kmeans(test_embedding, centers)
    train_eval = clustering_with_semantic_mapping(train["y"], train_assignments, num_classes=config["data"]["num_classes"], features=train_embedding)
    val_eval = clustering_with_semantic_mapping(val["y"], val_assignments, num_classes=config["data"]["num_classes"], features=val_embedding)
    test_eval = clustering_with_semantic_mapping(test["y"], test_assignments, num_classes=config["data"]["num_classes"], features=test_embedding)
    np.save(exp_path / "checkpoints" / "train_embedding.npy", train_embedding)
    np.save(exp_path / "checkpoints" / "cluster_centers.npy", centers)
    save_json(
        exp_path / "graph_summary.json",
        {
            "nodes": int(affinity.shape[0]),
            "edges": int(affinity.nnz),
            "connected_components": int(graph_components),
            "sigma": sigma,
            "neighbor_backend": neighbor_summary,
            "spectral_backend": spectral_backend,
            "clustering_backend": clustering_backend,
            **inject_summary,
        },
    )
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
        "graph_summary": {
            "nodes": int(affinity.shape[0]),
            "edges": int(affinity.nnz),
            "connected_components": int(graph_components),
            "neighbor_backend": neighbor_summary,
            "spectral_backend": spectral_backend,
            "clustering_backend": clustering_backend,
        },
        "constraint_injection": inject_summary,
        "best_init_id": int(best["init_id"]),
    }
    write_experiment_payload(
        exp_path,
        resolved_config={
            "variant": "semi_supervised_spectral",
            "semi_supervised_spectral": cfg,
            "protocol": config["protocol"],
            "data": config["data"],
            "performance": performance_cfg,
            "device": device,
        },
        train_summary=train_summary,
        eval_summary=eval_summary,
    )
    return eval_summary
