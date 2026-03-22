from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _try_import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:
        return None


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _want_faiss_gpu(performance_cfg: dict[str, Any]) -> bool:
    mode = str(performance_cfg.get("faiss_use_gpu", "auto")).lower()
    if mode == "true" or mode == "always":
        return True
    if mode == "false" or mode == "never":
        return False
    return _torch_cuda_available()


@dataclass
class NeighborSearchIndex:
    backend: str
    index: Any
    n_neighbors: int
    use_gpu: bool
    train_size: int
    dim: int
    query_chunk: int

    def kneighbors(self, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_query = np.ascontiguousarray(x_query.astype(np.float32))
        if self.backend.startswith("faiss"):
            if self.query_chunk and x_query.shape[0] > self.query_chunk:
                all_dist = []
                all_idx = []
                for start in range(0, x_query.shape[0], self.query_chunk):
                    chunk = x_query[start : start + self.query_chunk]
                    distances_sq, neighbors = self.index.search(chunk, self.n_neighbors)
                    distances = np.sqrt(np.maximum(distances_sq, 0.0)).astype(np.float32)
                    all_dist.append(distances)
                    all_idx.append(neighbors.astype(np.int64))
                return np.vstack(all_dist), np.vstack(all_idx)
            distances_sq, neighbors = self.index.search(x_query, self.n_neighbors)
            distances = np.sqrt(np.maximum(distances_sq, 0.0)).astype(np.float32)
            return distances.astype(np.float32), neighbors.astype(np.int64)
        distances, neighbors = self.index.kneighbors(x_query)
        return distances.astype(np.float32), neighbors.astype(np.int64)

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "use_gpu": bool(self.use_gpu),
            "train_size": int(self.train_size),
            "dim": int(self.dim),
            "n_neighbors": int(self.n_neighbors),
        }


def build_neighbor_index(x_train: np.ndarray, n_neighbors: int, performance_cfg: dict[str, Any]) -> NeighborSearchIndex:
    x_train = np.ascontiguousarray(x_train.astype(np.float32))
    prefer_faiss = bool(performance_cfg.get("prefer_faiss", True))
    faiss = _try_import_faiss() if prefer_faiss else None
    query_chunk = int(performance_cfg.get("faiss_query_chunk", 0) or 0)
    if faiss is not None:
        use_gpu = _want_faiss_gpu(performance_cfg) and hasattr(faiss, "StandardGpuResources")
        try:
            if use_gpu:
                res = faiss.StandardGpuResources()
                cpu_index = faiss.IndexFlatL2(x_train.shape[1])
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                gpu_index.add(x_train)
                return NeighborSearchIndex(
                    backend="faiss-gpu",
                    index=gpu_index,
                    n_neighbors=n_neighbors,
                    use_gpu=True,
                    train_size=len(x_train),
                    dim=x_train.shape[1],
                    query_chunk=query_chunk,
                )
            cpu_index = faiss.IndexFlatL2(x_train.shape[1])
            cpu_index.add(x_train)
            return NeighborSearchIndex(
                backend="faiss-cpu",
                index=cpu_index,
                n_neighbors=n_neighbors,
                use_gpu=False,
                train_size=len(x_train),
                dim=x_train.shape[1],
                query_chunk=query_chunk,
            )
        except Exception:
            pass
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(x_train)), metric="euclidean")
    nn.fit(x_train)
    return NeighborSearchIndex(
        backend="sklearn",
        index=nn,
        n_neighbors=min(n_neighbors, len(x_train)),
        use_gpu=False,
        train_size=len(x_train),
        dim=x_train.shape[1],
        query_chunk=0,
    )
