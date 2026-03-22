from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(path: str | Path, data: Any) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_text(path: str | Path, text: str) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(text, encoding="utf-8")


def save_csv_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    if not rows:
        with path_obj.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path_obj.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

