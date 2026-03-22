from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from formal_semisup.data.dataset import CanonicalDataset
from formal_semisup.evaluation.metrics import classification_metrics
from formal_semisup.methods.common import copy_canonical_artifacts, write_experiment_payload
from formal_semisup.reporting.visualization import save_embedding_plots
from formal_semisup.utils.io import ensure_dir, save_json
from formal_semisup.utils.repro import set_global_seed


def _torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("torch is required for supervised variants") from exc
    return torch, nn, DataLoader, Dataset


class TimeSeriesDataset:
    def __init__(self, x_seq: np.ndarray, mask: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.x_seq = x_seq.astype(np.float32)
        self.mask = mask.astype(bool)
        self.y = y.astype(np.int64)
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {
            "x_seq": self.x_seq[idx],
            "mask": self.mask[idx],
            "y": self.y[idx],
            "index": self.indices[idx],
        }


def _make_loader(dataset: TimeSeriesDataset, batch_size: int, shuffle: bool):
    torch, _, DataLoader, Dataset = _torch()

    class _WrappedDataset(Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, idx):
            item = dataset[idx]
            return {
                "x_seq": torch.from_numpy(item["x_seq"]),
                "mask": torch.from_numpy(item["mask"]),
                "y": torch.tensor(item["y"], dtype=torch.long),
                "index": torch.tensor(item["index"], dtype=torch.long),
            }

    return DataLoader(_WrappedDataset(), batch_size=batch_size, shuffle=shuffle)


def masked_mean_pool(sequence_embeddings, mask):
    torch, _, _, _ = _torch()
    valid = (~mask).unsqueeze(-1).float()
    summed = (sequence_embeddings * valid).sum(dim=1)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return summed / denom


def reconstruction_mse(pred, target, mask):
    torch, _, _, _ = _torch()
    valid = (~mask).unsqueeze(-1).float()
    sq = ((pred - target) ** 2) * valid
    denom = valid.sum().clamp_min(1.0) * pred.shape[-1]
    return sq.sum() / denom


class RecurrentClassifier:
    def __init__(self, variant: str, input_dim: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int):
        torch, nn, _, _ = _torch()
        self.variant = variant
        rnn_cls = {"supervised_lstm": nn.LSTM, "supervised_rnn": nn.RNN, "supervised_gru": nn.GRU}[variant]
        self.model = nn.Module()
        self.model.encoder = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.model.dropout = nn.Dropout(dropout)
        self.model.head = nn.Linear(hidden_size, num_classes)

        def forward(x_seq, mask):
            output, _ = self.model.encoder(x_seq)
            pooled = masked_mean_pool(output, mask)
            embedding = self.model.dropout(pooled)
            logits = self.model.head(embedding)
            return logits, embedding

        self.model.forward = forward


class TransformerClassifier:
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, ffn_dim: int, dropout: float, num_classes: int):
        torch, nn, _, _ = _torch()
        self.model = nn.Module()
        self.model.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.model.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.model.dropout = nn.Dropout(dropout)
        self.model.head = nn.Linear(d_model, num_classes)

        def forward(x_seq, mask):
            x = self.model.input_proj(x_seq)
            encoded = self.model.encoder(x, src_key_padding_mask=mask)
            pooled = masked_mean_pool(encoded, mask)
            embedding = self.model.dropout(pooled)
            logits = self.model.head(embedding)
            return logits, embedding

        self.model.forward = forward


class ConvAutoencoderClassifier:
    def __init__(self, input_dim: int, latent_channels: int, num_classes: int):
        torch, nn, _, _ = _torch()
        self.model = nn.Module()
        self.model.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.model.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, input_dim, kernel_size=3, padding=1),
        )
        self.model.classifier = nn.Linear(latent_channels, num_classes)

        def encode(x_seq, mask):
            x = x_seq.transpose(1, 2)
            encoded = self.model.encoder(x).transpose(1, 2)
            pooled = masked_mean_pool(encoded, mask)
            return encoded, pooled

        def reconstruct(x_seq):
            x = x_seq.transpose(1, 2)
            decoded = self.model.decoder(self.model.encoder(x)).transpose(1, 2)
            return decoded

        def classify(x_seq, mask):
            encoded, pooled = encode(x_seq, mask)
            logits = self.model.classifier(pooled)
            return logits, pooled, encoded

        self.model.encode = encode
        self.model.reconstruct = reconstruct
        self.model.classify = classify


def _to_device(batch, device):
    torch, _, _, _ = _torch()
    return {
        "x_seq": batch["x_seq"].to(device=device, dtype=torch.float32),
        "mask": batch["mask"].to(device=device, dtype=torch.bool),
        "y": batch["y"].to(device=device, dtype=torch.long),
        "index": batch["index"].to(device=device, dtype=torch.long),
    }


def _evaluate_classifier(model, loader, device: str, num_classes: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    torch, _, _, _ = _torch()
    model.eval()
    all_logits = []
    all_embeddings = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            logits, embeddings = model(batch["x_seq"], batch["mask"])
            all_logits.append(logits.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
            all_targets.append(batch["y"].cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    embeddings_np = np.concatenate(all_embeddings, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    predictions = logits_np.argmax(axis=1)
    metrics = classification_metrics(targets_np, predictions, num_classes=num_classes)
    return metrics, logits_np, embeddings_np, predictions


def _run_training_loop(
    *,
    model,
    optimizer,
    train_loader,
    val_loader,
    max_epochs: int,
    patience: int,
    device: str,
    checkpoint_dir: Path,
    variant: str,
) -> dict[str, Any]:
    torch, nn, _, _ = _torch()
    criterion = nn.CrossEntropyLoss()
    best_metric = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.ckpt"
    last_path = checkpoint_dir / "last.ckpt"
    start_epoch = 0
    if last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        best_metric = float(state["best_metric"])
        best_epoch = int(state["best_epoch"])
        start_epoch = int(state["epoch"]) + 1
        history = state.get("history", [])
    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch["x_seq"], batch["mask"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_metrics, _, _, _ = _evaluate_classifier(model, val_loader, device, num_classes=model.head.out_features)
        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "val_OA": float(val_metrics["OA"]),
            "val_MA": float(val_metrics["MA"]),
        }
        history.append(epoch_record)
        improved = val_metrics["OA"] > best_metric
        if improved:
            best_metric = float(val_metrics["OA"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "history": history,
                    "variant": variant,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "history": history,
                "variant": variant,
            },
            last_path,
        )
        if epochs_without_improvement >= patience:
            break
    if best_path.exists():
        best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state["model_state"])
    return {
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "history": history,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }


def _copy_protocol(exp_dir: Path, canonical_dir: str | Path) -> None:
    copy_canonical_artifacts(canonical_dir, exp_dir)
    ensure_dir(exp_dir / "logs")
    ensure_dir(exp_dir / "checkpoints")


def _base_resolved_config(config: dict[str, Any], variant: str, device: str) -> dict[str, Any]:
    return {
        "variant": variant,
        "device": device,
        "protocol": config["protocol"],
        "data": config["data"],
        "supervised": config["supervised"],
        "cae": config["cae"],
    }


def run_supervised_experiment(
    *,
    variant: str,
    dataset: CanonicalDataset,
    config: dict[str, Any],
    exp_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    torch, nn, _, _ = _torch()
    set_global_seed(config["protocol"]["split_seed"])
    exp_path = Path(exp_dir)
    _copy_protocol(exp_path, exp_path.parent / "canonical")
    supervised_cfg = config["supervised"]
    x_train = dataset.split_arrays("train")
    x_val = dataset.split_arrays("val")
    x_test = dataset.split_arrays("test")
    labeled_indices = set(dataset.labeled_train_indices())
    labeled_mask = np.asarray([int(idx) in labeled_indices for idx in x_train["indices"]], dtype=bool)
    train_labeled = {
        key: value[labeled_mask] if isinstance(value, np.ndarray) and len(value) == len(labeled_mask) else value
        for key, value in x_train.items()
    }
    train_loader = _make_loader(
        TimeSeriesDataset(train_labeled["x_seq"], train_labeled["mask"], train_labeled["y"], train_labeled["indices"]),
        batch_size=supervised_cfg["batch_size"],
        shuffle=True,
    )
    val_loader = _make_loader(
        TimeSeriesDataset(x_val["x_seq"], x_val["mask"], x_val["y"], x_val["indices"]),
        batch_size=supervised_cfg["batch_size"],
        shuffle=False,
    )
    test_loader = _make_loader(
        TimeSeriesDataset(x_test["x_seq"], x_test["mask"], x_test["y"], x_test["indices"]),
        batch_size=supervised_cfg["batch_size"],
        shuffle=False,
    )
    train_eval_loader = _make_loader(
        TimeSeriesDataset(x_train["x_seq"], x_train["mask"], x_train["y"], x_train["indices"]),
        batch_size=supervised_cfg["batch_size"],
        shuffle=False,
    )
    if variant in {"supervised_lstm", "supervised_rnn", "supervised_gru"}:
        wrapper = RecurrentClassifier(
            variant,
            input_dim=dataset.x_seq.shape[-1],
            hidden_size=supervised_cfg["hidden_size"],
            num_layers=supervised_cfg["num_layers"],
            dropout=supervised_cfg["dropout"],
            num_classes=config["data"]["num_classes"],
        )
        model = wrapper.model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=supervised_cfg["lr"], weight_decay=supervised_cfg["weight_decay"])
        train_summary = _run_training_loop(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=supervised_cfg["max_epochs"],
            patience=supervised_cfg["patience"],
            device=device,
            checkpoint_dir=exp_path / "checkpoints",
            variant=variant,
        )
        train_metrics, _, train_embeddings, train_preds = _evaluate_classifier(model, train_eval_loader, device, config["data"]["num_classes"])
        val_metrics, _, val_embeddings, val_preds = _evaluate_classifier(model, val_loader, device, config["data"]["num_classes"])
        test_metrics, _, test_embeddings, test_preds = _evaluate_classifier(model, test_loader, device, config["data"]["num_classes"])
    elif variant == "supervised_transformer":
        wrapper = TransformerClassifier(
            input_dim=dataset.x_seq.shape[-1],
            d_model=supervised_cfg["hidden_size"],
            nhead=supervised_cfg["transformer_heads"],
            num_layers=supervised_cfg["num_layers"],
            ffn_dim=supervised_cfg["transformer_ffn"],
            dropout=supervised_cfg["transformer_dropout"],
            num_classes=config["data"]["num_classes"],
        )
        model = wrapper.model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=supervised_cfg["lr"], weight_decay=supervised_cfg["weight_decay"])
        train_summary = _run_training_loop(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=supervised_cfg["max_epochs"],
            patience=supervised_cfg["patience"],
            device=device,
            checkpoint_dir=exp_path / "checkpoints",
            variant=variant,
        )
        train_metrics, _, train_embeddings, train_preds = _evaluate_classifier(model, train_eval_loader, device, config["data"]["num_classes"])
        val_metrics, _, val_embeddings, val_preds = _evaluate_classifier(model, val_loader, device, config["data"]["num_classes"])
        test_metrics, _, test_embeddings, test_preds = _evaluate_classifier(model, test_loader, device, config["data"]["num_classes"])
    elif variant == "cae_pretrain_classifier":
        cae_cfg = config["cae"]
        wrapper = ConvAutoencoderClassifier(input_dim=dataset.x_seq.shape[-1], latent_channels=cae_cfg["latent_channels"], num_classes=config["data"]["num_classes"])
        model = wrapper.model.to(device)
        checkpoint_dir = exp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ae_optimizer = torch.optim.AdamW(model.parameters(), lr=cae_cfg["lr"], weight_decay=cae_cfg["weight_decay"])
        unsup_loader = _make_loader(
            TimeSeriesDataset(x_train["x_seq"], x_train["mask"], x_train["y"], x_train["indices"]),
            batch_size=supervised_cfg["batch_size"],
            shuffle=True,
        )
        unsup_val_loader = _make_loader(
            TimeSeriesDataset(x_val["x_seq"], x_val["mask"], x_val["y"], x_val["indices"]),
            batch_size=supervised_cfg["batch_size"],
            shuffle=False,
        )
        best_recon = float("inf")
        best_recon_epoch = -1
        pretrain_history = []
        best_ae_path = checkpoint_dir / "best_pretrain.ckpt"
        last_ae_path = checkpoint_dir / "last_pretrain.ckpt"
        start_epoch = 0
        if last_ae_path.exists():
            state = torch.load(last_ae_path, map_location=device)
            model.load_state_dict(state["model_state"])
            ae_optimizer.load_state_dict(state["optimizer_state"])
            best_recon = float(state.get("best_recon", best_recon))
            best_recon_epoch = int(state.get("best_recon_epoch", best_recon_epoch))
            pretrain_history = state.get("history", pretrain_history)
            start_epoch = int(state.get("epoch", -1)) + 1
        elif best_ae_path.exists():
            state = torch.load(best_ae_path, map_location=device)
            model.load_state_dict(state["model_state"])
            best_recon = float(state.get("best_recon", best_recon))
            best_recon_epoch = int(state.get("epoch", best_recon_epoch))
            start_epoch = cae_cfg["pretrain_epochs"]
        for epoch in range(start_epoch, cae_cfg["pretrain_epochs"]):
            model.train()
            losses = []
            for batch in unsup_loader:
                batch = _to_device(batch, device)
                ae_optimizer.zero_grad(set_to_none=True)
                recon = model.reconstruct(batch["x_seq"])
                loss = reconstruction_mse(recon, batch["x_seq"], batch["mask"])
                loss.backward()
                ae_optimizer.step()
                losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                val_losses = []
                for batch in unsup_val_loader:
                    batch = _to_device(batch, device)
                    recon = model.reconstruct(batch["x_seq"])
                    val_losses.append(float(reconstruction_mse(recon, batch["x_seq"], batch["mask"]).item()))
            val_recon = float(np.mean(val_losses)) if val_losses else float("inf")
            pretrain_history.append({"epoch": epoch, "train_recon_loss": float(np.mean(losses)), "val_reconstruction_loss": val_recon})
            if val_recon < best_recon:
                best_recon = val_recon
                best_recon_epoch = epoch
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "best_recon": best_recon}, best_ae_path)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": ae_optimizer.state_dict(),
                    "epoch": epoch,
                    "best_recon": best_recon,
                    "best_recon_epoch": best_recon_epoch,
                    "history": pretrain_history,
                },
                last_ae_path,
            )
        if best_ae_path.exists():
            state = torch.load(best_ae_path, map_location=device)
            model.load_state_dict(state["model_state"])

        class _CAEClassifier(nn.Module):
            def __init__(self, cae_model):
                super().__init__()
                self.cae_model = cae_model
                self.head = cae_model.classifier

            def forward(self, x_seq, mask):
                logits, pooled, _ = self.cae_model.classify(x_seq, mask)
                return logits, pooled

        classifier_model = _CAEClassifier(model).to(device)
        optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=cae_cfg["lr"], weight_decay=cae_cfg["weight_decay"])
        finetune_summary = _run_training_loop(
            model=classifier_model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=cae_cfg["finetune_epochs"],
            patience=cae_cfg["patience"],
            device=device,
            checkpoint_dir=checkpoint_dir,
            variant=variant,
        )
        train_metrics, _, train_embeddings, train_preds = _evaluate_classifier(classifier_model, train_eval_loader, device, config["data"]["num_classes"])
        val_metrics, _, val_embeddings, val_preds = _evaluate_classifier(classifier_model, val_loader, device, config["data"]["num_classes"])
        test_metrics, _, test_embeddings, test_preds = _evaluate_classifier(classifier_model, test_loader, device, config["data"]["num_classes"])
        train_summary = {
            "stage1_pretrain": {
                "best_epoch": best_recon_epoch,
                "best_val_reconstruction_loss": best_recon,
                "history": pretrain_history,
                "best_checkpoint": str(best_ae_path),
                "last_checkpoint": str(last_ae_path),
            },
            "stage2_finetune": finetune_summary,
        }
    else:
        raise ValueError(f"unknown supervised variant: {variant}")

    plot_paths = save_embedding_plots(exp_path, test_embeddings, x_test["y"])
    eval_summary = {
        "variant": variant,
        "evaluation_type": "classification",
        "splits": {
            "train": {"evaluation_type": "classification", "classification_metrics": train_metrics},
            "val": {"evaluation_type": "classification", "classification_metrics": val_metrics},
            "test": {"evaluation_type": "classification", "classification_metrics": test_metrics},
        },
        "artifacts": plot_paths,
    }
    write_experiment_payload(
        exp_path,
        resolved_config=_base_resolved_config(config, variant, device),
        train_summary=train_summary,
        eval_summary=eval_summary,
    )
    save_json(exp_path / "predictions_summary.json", {"train": train_preds.tolist(), "val": val_preds.tolist(), "test": test_preds.tolist()})
    return eval_summary
