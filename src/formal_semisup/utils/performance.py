from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for performance utilities") from exc
    return torch


def is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


def estimate_numpy_bytes(arrays: list[np.ndarray]) -> int:
    return int(sum(int(arr.nbytes) for arr in arrays))


def should_preload_to_device(device: str, arrays: list[np.ndarray], performance_cfg: dict[str, Any]) -> bool:
    policy = performance_cfg.get("gpu_preload_policy", "auto")
    if not is_cuda_device(device):
        return False
    if policy == "always":
        return True
    if policy == "never":
        return False
    threshold_bytes = int(float(performance_cfg.get("gpu_preload_max_gb", 32.0)) * (1024**3))
    return estimate_numpy_bytes(arrays) <= threshold_bytes


def setup_torch_performance(device: str, performance_cfg: dict[str, Any]) -> None:
    torch = _torch()
    if not is_cuda_device(device):
        return
    allow_tf32 = bool(performance_cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_amp_dtype(dtype_name: str):
    torch = _torch()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(str(dtype_name).lower(), torch.bfloat16)


@dataclass
class TensorBatchStream:
    tensors: dict[str, Any]
    batch_size: int
    shuffle: bool

    def __iter__(self):
        torch = _torch()
        n_items = int(next(iter(self.tensors.values())).shape[0])
        base_device = next(iter(self.tensors.values())).device
        if self.shuffle:
            indices = torch.randperm(n_items, device=base_device)
        else:
            indices = torch.arange(n_items, device=base_device)
        for start in range(0, n_items, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            yield {key: value.index_select(0, batch_indices) for key, value in self.tensors.items()}

    def __len__(self) -> int:
        n_items = int(next(iter(self.tensors.values())).shape[0])
        return (n_items + self.batch_size - 1) // self.batch_size


def make_tensor_batch_stream(
    *,
    x: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: str,
    performance_cfg: dict[str, Any],
) -> tuple[TensorBatchStream, dict[str, Any]]:
    torch = _torch()
    preload = should_preload_to_device(device, [x, mask.astype(np.uint8), y, indices], performance_cfg)
    target_device = device if preload else "cpu"
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=target_device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=target_device)
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=target_device)
    idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=target_device)
    if target_device == "cpu" and is_cuda_device(device) and bool(performance_cfg.get("pin_memory", True)):
        x_tensor = x_tensor.pin_memory()
        mask_tensor = mask_tensor.pin_memory()
        y_tensor = y_tensor.pin_memory()
        idx_tensor = idx_tensor.pin_memory()
    stream = TensorBatchStream(
        tensors={"x_seq": x_tensor, "mask": mask_tensor, "y": y_tensor, "index": idx_tensor},
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return stream, {"storage": "gpu" if preload else "cpu", "estimated_bytes": estimate_numpy_bytes([x, mask.astype(np.uint8), y, indices])}


def make_flat_tensor_batch_stream(
    *,
    x: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: str,
    performance_cfg: dict[str, Any],
) -> tuple[TensorBatchStream, dict[str, Any]]:
    torch = _torch()
    preload = should_preload_to_device(device, [x], performance_cfg)
    target_device = device if preload else "cpu"
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=target_device)
    if target_device == "cpu" and is_cuda_device(device) and bool(performance_cfg.get("pin_memory", True)):
        x_tensor = x_tensor.pin_memory()
    stream = TensorBatchStream(tensors={"x": x_tensor}, batch_size=batch_size, shuffle=shuffle)
    return stream, {"storage": "gpu" if preload else "cpu", "estimated_bytes": estimate_numpy_bytes([x])}
