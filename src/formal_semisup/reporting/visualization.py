from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def _scatter(path: Path, embedding_2d: np.ndarray, labels: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, s=12, cmap="tab20")
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_embedding_plots(output_dir: str | Path, embeddings: np.ndarray, labels: np.ndarray) -> dict[str, str]:
    output_path = Path(output_dir)
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) < 2:
        raise ValueError("embeddings must be [N, D] with N >= 2")
    try:
        import umap
    except Exception as exc:
        raise RuntimeError("umap-learn is required for UMAP plot generation") from exc
    umap_proj = umap.UMAP(random_state=42).fit_transform(embeddings)
    tsne_proj = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(embeddings)
    umap_path = output_path / "umap.png"
    tsne_path = output_path / "tsne.png"
    _scatter(umap_path, umap_proj, labels, "UMAP")
    _scatter(tsne_path, tsne_proj, labels, "t-SNE")
    return {"umap": str(umap_path), "tsne": str(tsne_path)}

