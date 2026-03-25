from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from formal_semisup.data.dataset import CanonicalDataset
from formal_semisup.evaluation.metrics import clustering_with_semantic_mapping
from formal_semisup.methods.common import copy_canonical_artifacts, select_best_candidate, write_experiment_payload
from formal_semisup.utils.experiment_logger import get_experiment_logger
from formal_semisup.utils.faiss_utils import build_neighbor_index
from formal_semisup.utils.io import ensure_dir, save_json
from formal_semisup.utils.repro import set_global_seed


def _build_constraint_maps(pairwise_constraints: dict[str, Any]) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    must = defaultdict(set)
    cannot = defaultdict(set)
    for pair in pairwise_constraints["must_link"]:
        i, j = int(pair["i"]), int(pair["j"])
        must[i].add(j)
        must[j].add(i)
    for pair in pairwise_constraints["cannot_link"]:
        i, j = int(pair["i"]), int(pair["j"])
        cannot[i].add(j)
        cannot[j].add(i)
    return must, cannot


def _build_constraint_components(
    train_indices: np.ndarray,
    must: dict[int, set[int]],
    cannot: dict[int, set[int]],
) -> tuple[list[list[int]], dict[int, set[int]], set[int]]:
    train_set = {int(idx) for idx in train_indices.tolist()}
    constrained_nodes = set()
    for node, neighbors in must.items():
        if node in train_set:
            constrained_nodes.add(int(node))
        constrained_nodes.update(int(neighbor) for neighbor in neighbors if int(neighbor) in train_set)
    for node, neighbors in cannot.items():
        if node in train_set:
            constrained_nodes.add(int(node))
        constrained_nodes.update(int(neighbor) for neighbor in neighbors if int(neighbor) in train_set)
    components: list[list[int]] = []
    component_by_node: dict[int, int] = {}
    seen: set[int] = set()
    for node in sorted(constrained_nodes):
        if node in seen:
            continue
        stack = [node]
        component: list[int] = []
        seen.add(node)
        while stack:
            current = stack.pop()
            component.append(int(current))
            for neighbor in must.get(current, set()):
                neighbor = int(neighbor)
                if neighbor in constrained_nodes and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_id = len(components)
        for member in component:
            component_by_node[member] = component_id
        components.append(sorted(component))
    component_cannot: dict[int, set[int]] = {comp_id: set() for comp_id in range(len(components))}
    for node in constrained_nodes:
        left = component_by_node[int(node)]
        for neighbor in cannot.get(int(node), set()):
            neighbor = int(neighbor)
            if neighbor not in constrained_nodes:
                continue
            right = component_by_node[neighbor]
            if left == right:
                continue
            component_cannot[left].add(right)
            component_cannot[right].add(left)
    return components, component_cannot, constrained_nodes


def _init_centers(x_train: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    seeds = rng.choice(len(x_train), size=n_clusters, replace=False)
    return x_train[seeds].copy()


def _feasible(cluster_id: int, sample_idx: int, assignments: dict[int, int], must: dict[int, set[int]], cannot: dict[int, set[int]]) -> bool:
    for neighbor in must.get(sample_idx, set()):
        if neighbor in assignments and assignments[neighbor] != cluster_id:
            return False
    for neighbor in cannot.get(sample_idx, set()):
        if neighbor in assignments and assignments[neighbor] == cluster_id:
            return False
    return True


def _nearest_assign_batched(
    x: np.ndarray,
    centers: np.ndarray,
    performance_cfg: dict[str, Any],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(x) == 0:
        return np.empty((0,), dtype=np.int64), {"backend": "empty", "batch_size": int(batch_size)}
    neighbor_index = build_neighbor_index(centers, 1, performance_cfg, device=device)
    if neighbor_index.backend.startswith("faiss"):
        _, neighbors = neighbor_index.kneighbors(x)
        return neighbors[:, 0].astype(np.int64), neighbor_index.summary()
    centers32 = centers.astype(np.float32, copy=False)
    center_norm = (centers32**2).sum(axis=1, keepdims=True).T
    assignments = np.empty(len(x), dtype=np.int64)
    step = max(1, int(batch_size))
    for start in range(0, len(x), step):
        block = x[start : start + step].astype(np.float32, copy=False)
        block_norm = (block**2).sum(axis=1, keepdims=True)
        distances = block_norm - 2.0 * (block @ centers32.T) + center_norm
        assignments[start : start + len(block)] = np.argmin(distances, axis=1).astype(np.int64)
    return assignments, {"backend": "numpy-batched", "batch_size": int(step), "n_centers": int(len(centers))}


def _assign_points(
    x_train: np.ndarray,
    train_indices: np.ndarray,
    unconstrained_locals: np.ndarray,
    centers: np.ndarray,
    must: dict[int, set[int]],
    cannot: dict[int, set[int]],
    components: list[list[int]],
    component_cannot: dict[int, set[int]],
    performance_cfg: dict[str, Any],
    device: str,
    assignment_chunk: int,
) -> tuple[np.ndarray | None, bool, dict[str, Any] | None]:
    index_to_local = {int(idx): int(pos) for pos, idx in enumerate(train_indices)}
    assignments: dict[int, int] = {}
    cluster_assignments = np.full(len(x_train), fill_value=-1, dtype=np.int64)
    weighted_components = [
        (
            -len(component_cannot.get(comp_id, set())),
            -len(members),
            comp_id,
        )
        for comp_id, members in enumerate(components)
    ]
    for _, _, comp_id in sorted(weighted_components):
        members = components[comp_id]
        local_members = [index_to_local[int(member)] for member in members]
        centroid = x_train[np.asarray(local_members, dtype=np.int64)].mean(axis=0, keepdims=True)
        distances = ((centers - centroid) ** 2).sum(axis=1)
        blocked = {
            int(assignments[int(neighbor)])
            for member in members
            for neighbor in cannot.get(int(member), set())
            if int(neighbor) in assignments
        }
        candidate_clusters = [int(cluster_id) for cluster_id in np.argsort(distances) if int(cluster_id) not in blocked]
        if not candidate_clusters:
            return None, False, None
        chosen_cluster = int(candidate_clusters[0])
        for member in members:
            assignments[int(member)] = chosen_cluster
            cluster_assignments[index_to_local[int(member)]] = chosen_cluster

    assignment_backend = None
    if len(unconstrained_locals) > 0:
        unconstrained_assignments, assignment_backend = _nearest_assign_batched(
            x_train[unconstrained_locals],
            centers,
            performance_cfg,
            device,
            assignment_chunk,
        )
        cluster_assignments[unconstrained_locals] = unconstrained_assignments
    if np.any(cluster_assignments < 0):
        return None, False, assignment_backend
    return cluster_assignments, True, assignment_backend


def _recompute_centers(x_train: np.ndarray, assignments: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    centers = []
    for cluster_id in range(n_clusters):
        members = x_train[assignments == cluster_id]
        if len(members) == 0:
            centers.append(x_train[rng.integers(0, len(x_train))].copy())
        else:
            centers.append(members.mean(axis=0))
    return np.stack(centers, axis=0).astype(np.float32)


def _constraint_satisfaction(train_indices: np.ndarray, assignments: np.ndarray, pairwise_constraints: dict[str, Any]) -> dict[str, float]:
    idx_to_cluster = {int(idx): int(assignments[pos]) for pos, idx in enumerate(train_indices)}
    must_total = len(pairwise_constraints["must_link"])
    cannot_total = len(pairwise_constraints["cannot_link"])
    must_hit = sum(1 for pair in pairwise_constraints["must_link"] if idx_to_cluster[int(pair["i"])] == idx_to_cluster[int(pair["j"])])
    cannot_hit = sum(1 for pair in pairwise_constraints["cannot_link"] if idx_to_cluster[int(pair["i"])] != idx_to_cluster[int(pair["j"])])
    return {
        "must_link_satisfaction": float(must_hit / must_total) if must_total else 1.0,
        "cannot_link_satisfaction": float(cannot_hit / cannot_total) if cannot_total else 1.0,
    }


def run_cop_kmeans(
    *,
    dataset: CanonicalDataset,
    config: dict[str, Any],
    exp_dir: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    set_global_seed(config["protocol"]["split_seed"])
    exp_path = Path(exp_dir)
    ensure_dir(exp_path / "checkpoints")
    ensure_dir(exp_path / "logs")
    logger = get_experiment_logger(exp_path, "cop_kmeans")
    copy_canonical_artifacts(exp_path.parent / "canonical", exp_path)
    train = dataset.split_arrays("train")
    val = dataset.split_arrays("val")
    test = dataset.split_arrays("test")
    x_train = train["x_flat"]
    x_val = val["x_flat"]
    x_test = test["x_flat"]
    cfg = config["cop_kmeans"]
    performance_cfg = config.get("performance", {})
    assignment_chunk = int(cfg.get("assignment_chunk", performance_cfg.get("faiss_query_chunk", 8192) or 8192))
    must, cannot = _build_constraint_maps(dataset.pairwise_constraints)
    components, component_cannot, constrained_nodes = _build_constraint_components(train["indices"], must, cannot)
    constrained_mask = np.asarray([int(idx) in constrained_nodes for idx in train["indices"]], dtype=bool)
    unconstrained_locals = np.flatnonzero(~constrained_mask).astype(np.int64, copy=False)
    logger.log(f"backend=constraint_kmeans device={device}")
    logger.log_metrics(
        "cop_kmeans_constraints",
        constrained_nodes=len(constrained_nodes),
        must_components=len(components),
    )
    candidates = []
    rng_master = np.random.default_rng(config["protocol"]["split_seed"])
    assignment_backend = None
    for init_id in range(cfg["n_init"]):
        rng = np.random.default_rng(int(rng_master.integers(0, 1_000_000)))
        centers = _init_centers(x_train, cfg["n_clusters"], rng)
        feasible = True
        assignments = None
        for _ in range(cfg["max_iter"]):
            assignments_new, feasible, current_backend = _assign_points(
                x_train,
                train["indices"],
                unconstrained_locals,
                centers,
                must,
                cannot,
                components,
                component_cannot,
                performance_cfg,
                device,
                assignment_chunk,
            )
            if current_backend is not None:
                assignment_backend = current_backend
            if not feasible or assignments_new is None:
                break
            new_centers = _recompute_centers(x_train, assignments_new, cfg["n_clusters"], rng)
            if assignments is not None and np.array_equal(assignments, assignments_new):
                assignments = assignments_new
                centers = new_centers
                break
            assignments = assignments_new
            centers = new_centers
        if not feasible or assignments is None:
            candidates.append({"init_id": init_id, "feasible": False, "val_mapped_MA": -1.0, "val_NMI": -1.0, "val_ARI": -1.0})
            logger.log_metrics("cop_kmeans_init", init_id=init_id, feasible=False)
            continue
        val_assignments, _ = _nearest_assign_batched(x_val, centers, performance_cfg, device, assignment_chunk)
        val_eval = clustering_with_semantic_mapping(val["y"], val_assignments, num_classes=config["data"]["num_classes"], features=x_val)
        candidates.append(
            {
                "init_id": init_id,
                "feasible": True,
                "val_mapped_MA": float(val_eval["mapped_semantic_metrics"]["MA"]),
                "val_NMI": float(val_eval["clustering_metrics"]["NMI"]),
                "val_ARI": float(val_eval["clustering_metrics"]["ARI"]),
                "centers": centers,
                "train_assignments": assignments,
                "val_eval": val_eval,
            }
        )
        logger.log_metrics(
            "cop_kmeans_init",
            init_id=init_id,
            feasible=True,
            val_mapped_MA=float(val_eval["mapped_semantic_metrics"]["MA"]),
            val_NMI=float(val_eval["clustering_metrics"]["NMI"]),
            val_ARI=float(val_eval["clustering_metrics"]["ARI"]),
        )
    feasible_candidates = [candidate for candidate in candidates if candidate.get("feasible")]
    if not feasible_candidates:
        raise RuntimeError("all Cop-KMeans initializations were infeasible")
    best = select_best_candidate(feasible_candidates, "val_mapped_MA", ["val_NMI", "val_ARI"])
    centers = best["centers"]
    train_assignments = best["train_assignments"]
    val_assignments, val_backend = _nearest_assign_batched(x_val, centers, performance_cfg, device, assignment_chunk)
    test_assignments, test_backend = _nearest_assign_batched(x_test, centers, performance_cfg, device, assignment_chunk)
    backend_summary = assignment_backend or val_backend or test_backend
    logger.log_metrics(
        "cop_kmeans_backend",
        backend=str((backend_summary or {}).get("backend", "unknown")),
        device=device,
        assignment_chunk=int(assignment_chunk),
    )
    train_eval = clustering_with_semantic_mapping(train["y"], train_assignments, num_classes=config["data"]["num_classes"], features=x_train)
    val_eval = clustering_with_semantic_mapping(val["y"], val_assignments, num_classes=config["data"]["num_classes"], features=x_val)
    test_eval = clustering_with_semantic_mapping(test["y"], test_assignments, num_classes=config["data"]["num_classes"], features=x_test)
    np.save(exp_path / "checkpoints" / "best_centers.npy", centers)
    save_json(exp_path / "mapping.json", {"val": val_eval["mapping"], "test": test_eval["mapping"]})
    eval_summary = {
        "variant": "cop_kmeans",
        "evaluation_type": "clustering",
        "splits": {
            "train": {"evaluation_type": "clustering", **train_eval},
            "val": {"evaluation_type": "clustering", **val_eval},
            "test": {"evaluation_type": "clustering", **test_eval},
        },
    }
    train_summary = {
        "candidate_count": len(candidates),
        "feasible_candidates": len(feasible_candidates),
        "best_init_id": int(best["init_id"]),
        "selection_rule": {
            "primary": "val_mapped_MA",
            "tie_break": ["val_NMI", "val_ARI"],
        },
        "device": device,
        "assignment_backend": backend_summary,
        "constraint_satisfaction": _constraint_satisfaction(train["indices"], train_assignments, dataset.pairwise_constraints),
        "infeasible_restarts": int(len([candidate for candidate in candidates if not candidate.get("feasible")])),
        "constraint_components": int(len(components)),
        "constrained_nodes": int(len(constrained_nodes)),
    }
    write_experiment_payload(
        exp_path,
        resolved_config={
            "variant": "cop_kmeans",
            "cop_kmeans": cfg,
            "protocol": config["protocol"],
            "data": config["data"],
            "performance": performance_cfg,
            "device": device,
        },
        train_summary=train_summary,
        eval_summary=eval_summary,
    )
    logger.log(
        f"completed test_mapped_OA={test_eval['mapped_semantic_metrics']['OA']:.6f} "
        f"test_mapped_MA={test_eval['mapped_semantic_metrics']['MA']:.6f} "
        f"test_NMI={test_eval['clustering_metrics']['NMI']:.6f}"
    )
    return eval_summary
