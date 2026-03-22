from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from formal_semisup.data.dataset import CanonicalDataset
from formal_semisup.evaluation.metrics import clustering_with_semantic_mapping
from formal_semisup.methods.common import copy_canonical_artifacts, write_experiment_payload
from formal_semisup.reporting.visualization import save_embedding_plots
from formal_semisup.utils.io import ensure_dir, save_json
from formal_semisup.utils.performance import make_flat_tensor_batch_stream, resolve_amp_dtype, setup_torch_performance
from formal_semisup.utils.repro import set_global_seed


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("torch is required for SDEC") from exc
    return torch, nn, F


class StackedAutoencoder:
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        torch, nn, _, _, _ = _torch()
        encoder_layers = []
        prev = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, dim), nn.ReLU()])
            prev = dim
        encoder_layers.append(nn.Linear(prev, latent_dim))
        decoder_layers = []
        prev = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, dim), nn.ReLU()])
            prev = dim
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.model = nn.Module()
        self.model.encoder = nn.Sequential(*encoder_layers)
        self.model.decoder = nn.Sequential(*decoder_layers)
        self.model.cluster_centers = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=True)

        def encode(x):
            return self.model.encoder(x)

        def reconstruct(x):
            return self.model.decoder(self.model.encoder(x))

        def soft_assign(z):
            centers = self.model.cluster_centers
            dist = torch.cdist(z, centers, p=2.0) ** 2
            numerator = (1.0 + dist) ** -1
            return numerator / numerator.sum(dim=1, keepdim=True)

        self.model.encode = encode
        self.model.reconstruct = reconstruct
        self.model.soft_assign = soft_assign


def _target_distribution(q):
    weight = (q ** 2) / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _pairwise_loss(z, ml_pairs: np.ndarray, cl_pairs: np.ndarray):
    torch, _, _ = _torch()
    total = torch.tensor(0.0, device=z.device)
    if len(ml_pairs):
        ml_i = torch.as_tensor(ml_pairs[:, 0], dtype=torch.long, device=z.device)
        ml_j = torch.as_tensor(ml_pairs[:, 1], dtype=torch.long, device=z.device)
        total = total + ((z[ml_i] - z[ml_j]) ** 2).sum(dim=1).mean()
    if len(cl_pairs):
        cl_i = torch.as_tensor(cl_pairs[:, 0], dtype=torch.long, device=z.device)
        cl_j = torch.as_tensor(cl_pairs[:, 1], dtype=torch.long, device=z.device)
        total = total - ((z[cl_i] - z[cl_j]) ** 2).sum(dim=1).mean()
    return total


def _evaluate_sdec(model, x: np.ndarray, y: np.ndarray, device: str, num_classes: int, performance_cfg: dict[str, Any]):
    torch, _, _ = _torch()
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        with _autocast_context(device, performance_cfg):
            z = model.encode(x_tensor)
            q = model.soft_assign(z)
        assignments = torch.argmax(q, dim=1).cpu().numpy().astype(np.int64)
        embeddings = z.cpu().numpy().astype(np.float32)
    return clustering_with_semantic_mapping(y, assignments, num_classes=num_classes, features=embeddings), embeddings, assignments


def _autocast_context(device: str, performance_cfg: dict[str, Any]):
    torch, _, _ = _torch()
    enabled = bool(performance_cfg.get("mixed_precision", True)) and str(device).startswith("cuda")
    dtype = resolve_amp_dtype(performance_cfg.get("amp_dtype", "bfloat16"))
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def _make_scaler(device: str, performance_cfg: dict[str, Any]):
    torch, _, _ = _torch()
    enabled = bool(performance_cfg.get("mixed_precision", True)) and str(device).startswith("cuda")
    try:
        return torch.cuda.amp.GradScaler(enabled=enabled)
    except Exception:
        return None


def _maybe_compile(model, device: str, performance_cfg: dict[str, Any]):
    torch, _, _ = _torch()
    if not (str(device).startswith("cuda") and bool(performance_cfg.get("torch_compile", False))):
        return model
    try:
        return torch.compile(model, mode=performance_cfg.get("compile_mode", "max-autotune"))
    except Exception:
        return model


def run_sdec(
    *,
    dataset: CanonicalDataset,
    config: dict[str, Any],
    exp_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    torch, nn, F = _torch()
    set_global_seed(config["protocol"]["split_seed"])
    performance_cfg = config.get("performance", {})
    setup_torch_performance(device, performance_cfg)
    exp_path = Path(exp_dir)
    ensure_dir(exp_path / "checkpoints")
    ensure_dir(exp_path / "logs")
    copy_canonical_artifacts(exp_path.parent / "canonical", exp_path)
    train = dataset.split_arrays("train")
    val = dataset.split_arrays("val")
    test = dataset.split_arrays("test")
    cfg = config["sdec"]
    model_wrapper = StackedAutoencoder(cfg["input_dim"], cfg["hidden_dims"], cfg["latent_dim"])
    model = _maybe_compile(model_wrapper.model.to(device), device, performance_cfg)
    x_train = train["x_flat"].astype(np.float32)
    x_val = val["x_flat"].astype(np.float32)
    x_test = test["x_flat"].astype(np.float32)
    loader, train_stream_info = make_flat_tensor_batch_stream(
        x=x_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        device=device,
        performance_cfg=performance_cfg,
    )
    val_tensor = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    pretrain_scaler = _make_scaler(device, performance_cfg)
    best_pretrain = float("inf")
    best_pretrain_epoch = -1
    pretrain_history = []
    best_pretrain_path = exp_path / "checkpoints" / "best_pretrain.ckpt"
    last_pretrain_path = exp_path / "checkpoints" / "last_pretrain.ckpt"
    pretrain_start_epoch = 0
    if last_pretrain_path.exists():
        state = torch.load(last_pretrain_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)
        pretrain_optimizer.load_state_dict(state["optimizer_state"])
        best_pretrain = float(state.get("best_pretrain", best_pretrain))
        best_pretrain_epoch = int(state.get("best_pretrain_epoch", best_pretrain_epoch))
        pretrain_history = state.get("history", pretrain_history)
        pretrain_start_epoch = int(state.get("epoch", -1)) + 1
    elif best_pretrain_path.exists():
        state = torch.load(best_pretrain_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)
        best_pretrain = float(state.get("best_pretrain", best_pretrain))
        best_pretrain_epoch = int(state.get("epoch", best_pretrain_epoch))
        pretrain_start_epoch = cfg["pretrain_epochs"]
    for epoch in range(pretrain_start_epoch, cfg["pretrain_epochs"]):
        model.train()
        losses = []
        for batch in loader:
            batch_x = batch["x"]
            if str(batch_x.device) != device:
                batch_x = batch_x.to(device=device, non_blocking=True)
            pretrain_optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, performance_cfg):
                recon = model.reconstruct(batch_x)
                loss = F.mse_loss(recon, batch_x)
            if pretrain_scaler is not None:
                pretrain_scaler.scale(loss).backward()
                pretrain_scaler.step(pretrain_optimizer)
                pretrain_scaler.update()
            else:
                loss.backward()
                pretrain_optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            with _autocast_context(device, performance_cfg):
                val_recon = float(F.mse_loss(model.reconstruct(val_tensor), val_tensor).item())
        pretrain_history.append({"epoch": epoch, "train_reconstruction_loss": float(np.mean(losses)), "val_reconstruction_loss": val_recon})
        if val_recon < best_pretrain:
            best_pretrain = val_recon
            best_pretrain_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "best_pretrain": best_pretrain}, best_pretrain_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": pretrain_optimizer.state_dict(),
                "best_pretrain": best_pretrain,
                "best_pretrain_epoch": best_pretrain_epoch,
                "history": pretrain_history,
            },
            last_pretrain_path,
        )
    if best_pretrain_path.exists():
        state = torch.load(best_pretrain_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)
    with torch.no_grad():
        z_train = model.encode(torch.as_tensor(x_train, dtype=torch.float32, device=device)).cpu().numpy()
    kmeans = KMeans(n_clusters=cfg["n_clusters"], n_init=20, random_state=config["protocol"]["split_seed"])
    kmeans.fit(z_train)
    centers = torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    model.cluster_centers = nn.Parameter(centers)
    train_index_to_local = {int(idx): int(pos) for pos, idx in enumerate(train["indices"])}
    ml_pairs = np.asarray(
        [
            (train_index_to_local[int(pair["i"])], train_index_to_local[int(pair["j"])])
            for pair in dataset.pairwise_constraints["must_link"]
            if int(pair["i"]) in train_index_to_local and int(pair["j"]) in train_index_to_local
        ],
        dtype=np.int64,
    )
    cl_pairs = np.asarray(
        [
            (train_index_to_local[int(pair["i"])], train_index_to_local[int(pair["j"])])
            for pair in dataset.pairwise_constraints["cannot_link"]
            if int(pair["i"]) in train_index_to_local and int(pair["j"]) in train_index_to_local
        ],
        dtype=np.int64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = _make_scaler(device, performance_cfg)
    best_metric = -1.0
    best_epoch = -1
    wait = 0
    history = []
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    best_path = exp_path / "checkpoints" / "best.ckpt"
    last_path = exp_path / "checkpoints" / "last.ckpt"
    start_epoch = 0
    best_score = (-1.0, -1.0, -1.0)
    if last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)
        optimizer.load_state_dict(state["optimizer_state"])
        history = state.get("history", history)
        start_epoch = int(state.get("epoch", -1)) + 1
        best_score = tuple(state.get("best_score", best_score))
        best_metric = float(best_score[0])
    elif best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)
        history = state.get("history", history)
        best_score = tuple(state.get("selection_score", best_score))
        best_metric = float(best_score[0])
        best_epoch = int(state.get("epoch", best_epoch))
        start_epoch = cfg["train_epochs"]
    for epoch in range(start_epoch, cfg["train_epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, performance_cfg):
            z = model.encode(x_train_tensor)
            q = model.soft_assign(z)
            p = _target_distribution(q.detach())
            kl = torch.mean(torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8)), dim=1))
            pairwise = _pairwise_loss(z, ml_pairs, cl_pairs)
            loss = kl + cfg["lambda_pairwise"] * pairwise
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        val_eval, _, _ = _evaluate_sdec(model, x_val, val["y"], device, config["data"]["num_classes"], performance_cfg)
        record = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_kl_loss": float(kl.item()),
            "train_pairwise_loss": float(pairwise.item()),
            "val_mapped_MA": float(val_eval["mapped_semantic_metrics"]["MA"]),
            "val_NMI": float(val_eval["clustering_metrics"]["NMI"]),
            "val_ARI": float(val_eval["clustering_metrics"]["ARI"]),
        }
        history.append(record)
        score = (record["val_mapped_MA"], record["val_NMI"], record["val_ARI"])
        if score > best_score:
            best_metric = record["val_mapped_MA"]
            best_epoch = epoch
            wait = 0
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "selection_score": score,
                    "history": history,
                },
                best_path,
            )
        else:
            wait += 1
        torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_score": best_score,
                    "history": history,
                },
                last_path,
            )
        if wait >= cfg["patience"]:
            break
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)
    train_eval, train_embeddings, train_assignments = _evaluate_sdec(model, x_train, train["y"], device, config["data"]["num_classes"], performance_cfg)
    val_eval, val_embeddings, val_assignments = _evaluate_sdec(model, x_val, val["y"], device, config["data"]["num_classes"], performance_cfg)
    test_eval, test_embeddings, test_assignments = _evaluate_sdec(model, x_test, test["y"], device, config["data"]["num_classes"], performance_cfg)
    plot_paths = save_embedding_plots(exp_path, test_embeddings, test["y"])
    np.save(exp_path / "checkpoints" / "test_embeddings.npy", test_embeddings)
    save_json(exp_path / "mapping.json", {"val": val_eval["mapping"], "test": test_eval["mapping"]})
    eval_summary = {
        "variant": "sdec",
        "evaluation_type": "clustering",
        "splits": {
            "train": {"evaluation_type": "clustering", **train_eval},
            "val": {"evaluation_type": "clustering", **val_eval},
            "test": {"evaluation_type": "clustering", **test_eval},
        },
        "artifacts": plot_paths,
    }
    train_summary = {
        "stage1_pretrain": {
            "best_epoch": int(best_pretrain_epoch),
            "best_val_reconstruction_loss": float(best_pretrain),
            "history": pretrain_history,
            "best_checkpoint": str(best_pretrain_path),
            "last_checkpoint": str(last_pretrain_path),
            "data_stream": train_stream_info,
        },
        "stage2_clustering": {
            "best_epoch": int(best_epoch),
            "selection_rule": {"primary": "val_mapped_MA", "tie_break": ["val_NMI", "val_ARI"]},
            "history": history,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "pairwise_counts": {"must_link": int(len(ml_pairs)), "cannot_link": int(len(cl_pairs))},
            "performance": performance_cfg,
        },
    }
    write_experiment_payload(
        exp_path,
        resolved_config={"variant": "sdec", "sdec": cfg, "protocol": config["protocol"], "data": config["data"], "performance": performance_cfg},
        train_summary=train_summary,
        eval_summary=eval_summary,
    )
    return eval_summary
