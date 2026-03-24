from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from formal_semisup.utils.io import ensure_dir
from formal_semisup.utils.runtime import utc_now_iso


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


@dataclass
class ExperimentLogger:
    variant: str
    log_path: Path

    def log(self, message: str) -> None:
        line = f"[{utc_now_iso()}] [{self.variant}] {message}"
        ensure_dir(self.log_path.parent)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def log_metrics(self, stage: str, **metrics: Any) -> None:
        fields = [f"{key}={_format_value(value)}" for key, value in metrics.items()]
        self.log(f"{stage} " + " ".join(fields))


def get_experiment_logger(exp_dir: str | Path, variant: str) -> ExperimentLogger:
    exp_path = Path(exp_dir)
    return ExperimentLogger(variant=variant, log_path=exp_path / "logs" / "terminal.log")
