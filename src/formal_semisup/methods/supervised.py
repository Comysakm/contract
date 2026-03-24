from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from formal_semisup.data.dataset import CanonicalDataset
from formal_semisup.evaluation.metrics import classification_metrics
from formal_semisup.methods.common import copy_canonical_artifacts, write_experiment_payload
from formal_semisup.reporting.visualization import save_embedding_plots
from formal_semisup.utils.experiment_logger import get_experiment_logger
from formal_semisup.utils.io import ensure_dir, save_json
from formal_semisup.utils.performance import make_tensor_batch_stream, resolve_amp_dtype, setup_torch_performance
from formal_semisup.utils.repro import set_global_seed


def _torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("torch is required for supervised variants") from exc
    return torch, nn


def masked_mean_pool(sequence_embeddings, mask):
    valid = (~mask).unsqueeze(-1).float()
    summed = (sequence_embeddings * valid).sum(dim=1)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return summed / denom


def reconstruction_mse(pred, target, mask):
    valid = (~mask).unsqueeze(-1).float()
    sq = ((pred - target) ** 2) * valid
    denom = valid.sum().clamp_min(1.0) * pred.shape[-1]
    return sq.sum() / denom


def _autocast_context(device: str, performance_cfg: dict[str, Any]):
    torch, _ = _torch()
    enabled = bool(performance_cfg.get("mixed_precision", True)) and str(device).startswith("cuda")
    dtype = resolve_amp_dtype(performance_cfg.get("amp_dtype", "bfloat16"))
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def _make_scaler(device: str, performance_cfg: dict[str, Any]):
    torch, _ = _torch()
    enabled = bool(performance_cfg.get("mixed_precision", True)) and str(device).startswith("cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(enabled=enabled)
        except Exception:
            return None


def _maybe_compile(model, device: str, performance_cfg: dict[str, Any]):
    torch, _ = _torch()
    if not (str(device).startswith("cuda") and bool(performance_cfg.get("torch_compile", False))):
        return model
    try:
        return torch.compile(model, mode=performance_cfg.get("compile_mode", "max-autotune"))
    except Exception:
        return model


def _to_device(batch, device: str, performance_cfg: dict[str, Any]):
    current_device = str(batch["x_seq"].device)
    if current_device.startswith(device):
        return batch
    non_blocking = bool(performance_cfg.get("non_blocking_transfer", True))
    return {
        "x_seq": batch["x_seq"].to(device=device, dtype=batch["x_seq"].dtype, non_blocking=non_blocking),
        "mask": batch["mask"].to(device=device, dtype=batch["mask"].dtype, non_blocking=non_blocking),
        "y": batch["y"].to(device=device, dtype=batch["y"].dtype, non_blocking=non_blocking),
        "index": batch["index"].to(device=device, dtype=batch["index"].dtype, non_blocking=non_blocking),
    }


class RecurrentClassifier:
    def __init__(self, variant: str, input_dim: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int):
        _, nn = _torch()
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
        _, nn = _torch()
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
        _, nn = _torch()
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


def _evaluate_classifier(model, batch_stream, device: str, num_classes: int, performance_cfg: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    torch, _ = _torch()
    model.eval()
    all_logits = []
    all_embeddings = []
    all_targets = []
    with torch.no_grad():
        for batch in batch_stream:
            batch = _to_device(batch, device, performance_cfg)
            with _autocast_context(device, performance_cfg):
                logits, embeddings = model(batch["x_seq"], batch["mask"])
            all_logits.append(logits.detach().float().cpu().numpy())
            all_embeddings.append(embeddings.detach().float().cpu().numpy())
            all_targets.append(batch["y"].detach().cpu().numpy())
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
    train_stream,
    val_stream,
    max_epochs: int,
    patience: int,
    device: str,
    checkpoint_dir: Path,
    variant: str,
    num_classes: int,
    performance_cfg: dict[str, Any],
    logger,
    stage_name: str,
) -> dict[str, Any]:
    torch, nn = _torch()
    criterion = nn.CrossEntropyLoss()
    scaler = _make_scaler(device, performance_cfg)
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
        for batch in train_stream:
            batch = _to_device(batch, device, performance_cfg)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, performance_cfg):
                logits, _ = model(batch["x_seq"], batch["mask"])
                loss = criterion(logits, batch["y"])
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_losses.append(float(loss.item()))
        val_metrics, _, _, _ = _evaluate_classifier(model, val_stream, device, num_classes=num_classes, performance_cfg=performance_cfg)
        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "val_OA": float(val_metrics["OA"]),
            "val_MA": float(val_metrics["MA"]),
        }
        history.append(epoch_record)
        logger.log_metrics(
            stage_name,
            epoch=f"{epoch + 1}/{max_epochs}",
            train_loss=epoch_record["train_loss"],
            val_OA=epoch_record["val_OA"],
            val_MA=epoch_record["val_MA"],
            best_OA=max(best_metric, float(val_metrics["OA"])),
            wait=epochs_without_improvement,
        )
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
        "performance": config.get("performance", {}),
    }


def _build_streams(x_split: dict[str, np.ndarray], batch_size: int, shuffle: bool, device: str, performance_cfg: dict[str, Any]):
    return make_tensor_batch_stream(
        x=x_split["x_seq"],
        mask=x_split["mask"],
        y=x_split["y"],
        indices=x_split["indices"],
        batch_size=batch_size,
        shuffle=shuffle,
        device=device,
        performance_cfg=performance_cfg,
    )


def run_supervised_experiment(
    *,
    variant: str,
    dataset: CanonicalDataset,
    config: dict[str, Any],
    exp_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    torch, nn = _torch()
    set_global_seed(config["protocol"]["split_seed"])
    performance_cfg = config.get("performance", {})
    setup_torch_performance(device, performance_cfg)
    exp_path = Path(exp_dir)
    _copy_protocol(exp_path, exp_path.parent / "canonical")
    logger = get_experiment_logger(exp_path, variant)
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
    train_stream, train_stream_info = _build_streams(train_labeled, supervised_cfg["batch_size"], True, device, performance_cfg)
    val_stream, val_stream_info = _build_streams(x_val, supervised_cfg["batch_size"], False, device, performance_cfg)
    test_stream, test_stream_info = _build_streams(x_test, supervised_cfg["batch_size"], False, device, performance_cfg)
    train_eval_stream, train_eval_stream_info = _build_streams(x_train, supervised_cfg["batch_size"], False, device, performance_cfg)
    stream_info = {
        "train_labeled": train_stream_info,
        "train_full_eval": train_eval_stream_info,
        "val": val_stream_info,
        "test": test_stream_info,
    }
    logger.log(f"device={device} performance={performance_cfg}")
    logger.log(f"data_stream={stream_info}")
    if variant in {"supervised_lstm", "supervised_rnn", "supervised_gru"}:
        wrapper = RecurrentClassifier(
            variant,
            input_dim=dataset.x_seq.shape[-1],
            hidden_size=supervised_cfg["hidden_size"],
            num_layers=supervised_cfg["num_layers"],
            dropout=supervised_cfg["dropout"],
            num_classes=config["data"]["num_classes"],
        )
        model = _maybe_compile(wrapper.model.to(device), device, performance_cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=supervised_cfg["lr"], weight_decay=supervised_cfg["weight_decay"])
        train_summary = _run_training_loop(
            model=model,
            optimizer=optimizer,
            train_stream=train_stream,
            val_stream=val_stream,
            max_epochs=supervised_cfg["max_epochs"],
            patience=supervised_cfg["patience"],
            device=device,
            checkpoint_dir=exp_path / "checkpoints",
            variant=variant,
            num_classes=config["data"]["num_classes"],
            performance_cfg=performance_cfg,
            logger=logger,
            stage_name="train",
        )
        train_metrics, _, train_embeddings, train_preds = _evaluate_classifier(model, train_eval_stream, device, config["data"]["num_classes"], performance_cfg)
        val_metrics, _, val_embeddings, val_preds = _evaluate_classifier(model, val_stream, device, config["data"]["num_classes"], performance_cfg)
        test_metrics, _, test_embeddings, test_preds = _evaluate_classifier(model, test_stream, device, config["data"]["num_classes"], performance_cfg)
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
        model = _maybe_compile(wrapper.model.to(device), device, performance_cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=supervised_cfg["lr"], weight_decay=supervised_cfg["weight_decay"])
        train_summary = _run_training_loop(
            model=model,
            optimizer=optimizer,
            train_stream=train_stream,
            val_stream=val_stream,
            max_epochs=supervised_cfg["max_epochs"],
            patience=supervised_cfg["patience"],
            device=device,
            checkpoint_dir=exp_path / "checkpoints",
            variant=variant,
            num_classes=config["data"]["num_classes"],
            performance_cfg=performance_cfg,
            logger=logger,
            stage_name="train",
        )
        train_metrics, _, train_embeddings, train_preds = _evaluate_classifier(model, train_eval_stream, device, config["data"]["num_classes"], performance_cfg)
        val_metrics, _, val_embeddings, val_preds = _evaluate_classifier(model, val_stream, device, config["data"]["num_classes"], performance_cfg)
        test_metrics, _, test_embeddings, test_preds = _evaluate_classifier(model, test_stream, device, config["data"]["num_classes"], performance_cfg)
    elif variant == "cae_pretrain_classifier":
        cae_cfg = config["cae"]
        wrapper = ConvAutoencoderClassifier(input_dim=dataset.x_seq.shape[-1], latent_channels=cae_cfg["latent_channels"], num_classes=config["data"]["num_classes"])
        model = _maybe_compile(wrapper.model.to(device), device, performance_cfg)
        checkpoint_dir = exp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ae_optimizer = torch.optim.AdamW(model.parameters(), lr=cae_cfg["lr"], weight_decay=cae_cfg["weight_decay"])
        unsup_loader, unsup_stream_info = _build_streams(x_train, supervised_cfg["batch_size"], True, device, performance_cfg)
        unsup_val_loader, unsup_val_stream_info = _build_streams(x_val, supervised_cfg["batch_size"], False, device, performance_cfg)
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
        pretrain_scaler = _make_scaler(device, performance_cfg)
        for epoch in range(start_epoch, cae_cfg["pretrain_epochs"]):
            model.train()
            losses = []
            for batch in unsup_loader:
                batch = _to_device(batch, device, performance_cfg)
                ae_optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, performance_cfg):
                    recon = model.reconstruct(batch["x_seq"])
                    loss = reconstruction_mse(recon, batch["x_seq"], batch["mask"])
                if pretrain_scaler is not None:
                    pretrain_scaler.scale(loss).backward()
                    pretrain_scaler.step(ae_optimizer)
                    pretrain_scaler.update()
                else:
                    loss.backward()
                    ae_optimizer.step()
                losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                val_losses = []
                for batch in unsup_val_loader:
                    batch = _to_device(batch, device, performance_cfg)
                    with _autocast_context(device, performance_cfg):
                        recon = model.reconstruct(batch["x_seq"])
                        val_losses.append(float(reconstruction_mse(recon, batch["x_seq"], batch["mask"]).item()))
            val_recon = float(np.mean(val_losses)) if val_losses else float("inf")
            pretrain_history.append({"epoch": epoch, "train_recon_loss": float(np.mean(losses)), "val_reconstruction_loss": val_recon})
            logger.log_metrics(
                "cae_pretrain",
                epoch=f"{epoch + 1}/{cae_cfg['pretrain_epochs']}",
                train_recon_loss=float(np.mean(losses)) if losses else None,
                val_reconstruction_loss=val_recon,
                best_val_reconstruction_loss=min(best_recon, val_recon),
            )
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

        classifier_model = _maybe_compile(_CAEClassifier(model).to(device), device, performance_cfg)
        optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=cae_cfg["lr"], weight_decay=cae_cfg["weight_decay"])
        finetune_summary = _run_training_loop(
            model=classifier_model,
            optimizer=optimizer,
            train_stream=train_stream,
            val_stream=val_stream,
            max_epochs=cae_cfg["finetune_epochs"],
            patience=cae_cfg["patience"],
            device=device,
            checkpoint_dir=checkpoint_dir,
            variant=variant,
            num_classes=config["data"]["num_classes"],
            performance_cfg=performance_cfg,
            logger=logger,
            stage_name="cae_finetune",
        )
        train_metrics, _, train_embeddings, train_preds = _evaluate_classifier(classifier_model, train_eval_stream, device, config["data"]["num_classes"], performance_cfg)
        val_metrics, _, val_embeddings, val_preds = _evaluate_classifier(classifier_model, val_stream, device, config["data"]["num_classes"], performance_cfg)
        test_metrics, _, test_embeddings, test_preds = _evaluate_classifier(classifier_model, test_stream, device, config["data"]["num_classes"], performance_cfg)
        train_summary = {
            "stage1_pretrain": {
                "best_epoch": best_recon_epoch,
                "best_val_reconstruction_loss": best_recon,
                "history": pretrain_history,
                "best_checkpoint": str(best_ae_path),
                "last_checkpoint": str(last_ae_path),
                "data_stream": {"train": unsup_stream_info, "val": unsup_val_stream_info},
            },
            "stage2_finetune": finetune_summary,
        }
    else:
        raise ValueError(f"unknown supervised variant: {variant}")

    if "data_stream" not in train_summary:
        train_summary["data_stream"] = stream_info
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
    logger.log(f"completed test_OA={test_metrics['OA']:.6f} test_MA={test_metrics['MA']:.6f}")
    return eval_summary
