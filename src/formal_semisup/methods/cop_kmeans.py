from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from formal_semisup.data.dataset import CanonicalDataset
from formal_semisup.evaluation.metrics import clustering_with_semantic_mapping
from formal_semisup.methods.common import copy_canonical_artifacts, select_best_candidate, write_experiment_payload
from formal_semisup.utils.experiment_logger import get_experiment_logger
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


def _assign_points(
    x_train: np.ndarray,
    train_indices: np.ndarray,
    centers: np.ndarray,
    must: dict[int, set[int]],
    cannot: dict[int, set[int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, bool]:
    order = np.arange(len(x_train))
    rng.shuffle(order)
    assignments: dict[int, int] = {}
    cluster_assignments = np.full(len(x_train), fill_value=-1, dtype=np.int64)
    for local_idx in order:
        global_idx = int(train_indices[local_idx])
        distances = ((centers - x_train[local_idx : local_idx + 1]) ** 2).sum(axis=1)
        candidate_clusters = np.argsort(distances)
        assigned = False
        for cluster_id in candidate_clusters:
            if _feasible(int(cluster_id), global_idx, assignments, must, cannot):
                assignments[global_idx] = int(cluster_id)
                cluster_assignments[local_idx] = int(cluster_id)
                assigned = True
                break
        if not assigned:
            return None, False
    return cluster_assignments, True


def _recompute_centers(x_train: np.ndarray, assignments: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    centers = []
    for cluster_id in range(n_clusters):
        members = x_train[assignments == cluster_id]
        if len(members) == 0:
            centers.append(x_train[rng.integers(0, len(x_train))].copy())
        else:
            centers.append(members.mean(axis=0))
    return np.stack(centers, axis=0).astype(np.float32)


def _nearest_assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1).astype(np.int64)


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
    must, cannot = _build_constraint_maps(dataset.pairwise_constraints)
    logger.log("backend=cpu algorithm=constraint_kmeans")
    candidates = []
    rng_master = np.random.default_rng(config["protocol"]["split_seed"])
    for init_id in range(cfg["n_init"]):
        rng = np.random.default_rng(int(rng_master.integers(0, 1_000_000)))
        centers = _init_centers(x_train, cfg["n_clusters"], rng)
        feasible = True
        assignments = None
        for _ in range(cfg["max_iter"]):
            assignments_new, feasible = _assign_points(x_train, train["indices"], centers, must, cannot, rng)
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
        val_assignments = _nearest_assign(x_val, centers)
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
    val_assignments = _nearest_assign(x_val, centers)
    test_assignments = _nearest_assign(x_test, centers)
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
        "constraint_satisfaction": _constraint_satisfaction(train["indices"], train_assignments, dataset.pairwise_constraints),
        "infeasible_restarts": int(len([candidate for candidate in candidates if not candidate.get("feasible")])),
    }
    write_experiment_payload(
        exp_path,
        resolved_config={"variant": "cop_kmeans", "cop_kmeans": cfg, "protocol": config["protocol"], "data": config["data"]},
        train_summary=train_summary,
        eval_summary=eval_summary,
    )
    logger.log(
        f"completed test_mapped_OA={test_eval['mapped_semantic_metrics']['OA']:.6f} "
        f"test_mapped_MA={test_eval['mapped_semantic_metrics']['MA']:.6f} "
        f"test_NMI={test_eval['clustering_metrics']['NMI']:.6f}"
    )
    return eval_summary
