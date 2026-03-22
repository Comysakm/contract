from __future__ import annotations

import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from formal_semisup.utils.io import load_json, save_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_server_name(config_server_name: str = "auto") -> str:
    if config_server_name != "auto":
        return config_server_name
    hostname = socket.gethostname().strip() or "unknown-host"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in hostname)


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def dependency_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import matplotlib

        versions["matplotlib"] = matplotlib.__version__
    except Exception:
        pass
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = "missing"
    try:
        import umap

        versions["umap_learn"] = umap.__version__
    except Exception:
        versions["umap_learn"] = "missing"
    return versions


def read_status(exp_dir: str | Path) -> dict[str, Any] | None:
    status_path = Path(exp_dir) / "status.json"
    if not status_path.exists():
        return None
    return load_json(status_path)


def write_status(
    exp_dir: str | Path,
    *,
    state: str,
    command: str,
    seeds: dict[str, int],
    failure_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    exp_path = Path(exp_dir)
    current = read_status(exp_path) or {}
    payload = {
        "state": state,
        "command": command,
        "seeds": seeds,
        "dependency_versions": dependency_versions(),
        "started_at": current.get("started_at") or utc_now_iso(),
        "ended_at": utc_now_iso() if state in {"completed", "failed"} else None,
        "failure_reason": failure_reason,
    }
    if extra:
        payload.update(extra)
    save_json(exp_path / "status.json", payload)


def experiment_complete(exp_dir: str | Path) -> bool:
    status = read_status(exp_dir)
    if not status or status.get("state") != "completed":
        return False
    return (Path(exp_dir) / "eval_summary.json").exists()
