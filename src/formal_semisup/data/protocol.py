from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from formal_semisup.utils.io import save_csv_rows, save_json


@dataclass
class CanonicalProtocolArtifacts:
    split_manifest_path: Path
    normalization_stats_path: Path
    label_subset_csv_path: Path
    label_subset_json_path: Path
    pairwise_constraints_path: Path
    invalid_samples_path: Path


def load_npy_inputs(x_path: str | Path, y_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    x_data = np.load(Path(x_path), allow_pickle=True)
    y_data = np.load(Path(y_path), allow_pickle=True)
    if len(x_data) != len(y_data):
        raise ValueError(f"x/y length mismatch: {len(x_data)} vs {len(y_data)}")
    return x_data, np.asarray(y_data)


def _safe_sample_array(sample: Any) -> np.ndarray:
    arr = np.asarray(sample)
    if arr.dtype == np.dtype("O"):
        arr = np.array(sample, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def validate_raw_samples(x_data: np.ndarray, y_data: np.ndarray, spectral_dim: int) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, Any]]]:
    valid_x: list[np.ndarray] = []
    valid_y: list[int] = []
    invalid: list[dict[str, Any]] = []
    for idx, (sample, label) in enumerate(zip(x_data, y_data)):
        try:
            arr = _safe_sample_array(sample)
        except Exception as exc:
            invalid.append({"index": int(idx), "reason": f"array_cast_failed:{exc}"})
            continue
        if arr.ndim != 2:
            invalid.append({"index": int(idx), "reason": "not_2d"})
            continue
        if arr.shape[0] <= 0:
            invalid.append({"index": int(idx), "reason": "non_positive_length"})
            continue
        if arr.shape[1] < spectral_dim:
            invalid.append({"index": int(idx), "reason": f"input_dim_lt_{spectral_dim}"})
            continue
        try:
            label_int = int(label)
        except Exception:
            invalid.append({"index": int(idx), "reason": "label_invalid"})
            continue
        valid_x.append(arr)
        valid_y.append(label_int)
    return valid_x, np.asarray(valid_y, dtype=np.int64), invalid


def canonicalize_sample(sample: np.ndarray, *, seq_len: int, spectral_dim: int) -> dict[str, np.ndarray | int | bool]:
    length = int(sample.shape[0])
    use_len = min(length, seq_len)
    trimmed = sample[:use_len]
    x_spec = np.zeros((seq_len, spectral_dim), dtype=np.float32)
    x_spec[:use_len] = trimmed[:, :spectral_dim]
    if sample.shape[1] >= 11:
        doy_values = np.asarray(trimmed[:, 10], dtype=np.float32) / 365.0
        has_doy = True
    else:
        doy_values = np.linspace(0.0, 1.0, num=use_len, dtype=np.float32)
        has_doy = False
    x_doy = np.zeros((seq_len, 1), dtype=np.float32)
    x_doy[:use_len, 0] = doy_values
    mask = np.ones((seq_len,), dtype=bool)
    mask[:use_len] = False
    has_cloud = sample.shape[1] >= 10
    if has_cloud:
        cloud_mask = trimmed[:, 9] > 0.5
        mask[:use_len] = cloud_mask.astype(bool)
    return {
        "x_spec": x_spec,
        "x_doy": x_doy,
        "mask": mask,
        "orig_len": length,
        "used_len": use_len,
        "has_cloud_flag": has_cloud,
        "has_doy_column": has_doy,
    }


def stack_canonical_samples(samples: list[np.ndarray], *, seq_len: int, spectral_dim: int) -> dict[str, np.ndarray]:
    x_spec_all = []
    x_doy_all = []
    x_seq_all = []
    mask_all = []
    meta_rows: list[dict[str, Any]] = []
    for sample in samples:
        item = canonicalize_sample(sample, seq_len=seq_len, spectral_dim=spectral_dim)
        x_spec = item["x_spec"]
        x_doy = item["x_doy"]
        mask = item["mask"]
        x_seq = np.concatenate([x_spec, x_doy], axis=1)
        x_spec_all.append(x_spec)
        x_doy_all.append(x_doy)
        x_seq_all.append(x_seq)
        mask_all.append(mask)
        meta_rows.append(
            {
                "orig_len": int(item["orig_len"]),
                "used_len": int(item["used_len"]),
                "has_cloud_flag": bool(item["has_cloud_flag"]),
                "has_doy_column": bool(item["has_doy_column"]),
            }
        )
    return {
        "x_spec": np.stack(x_spec_all, axis=0).astype(np.float32),
        "x_doy": np.stack(x_doy_all, axis=0).astype(np.float32),
        "x_seq": np.stack(x_seq_all, axis=0).astype(np.float32),
        "mask": np.stack(mask_all, axis=0).astype(bool),
        "meta": np.asarray(meta_rows, dtype=object),
    }


def create_canonical_split(n_samples: int, train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(indices)
    train_end = int(round(n_samples * train_ratio))
    val_end = train_end + int(round(n_samples * val_ratio))
    val_end = min(val_end, n_samples)
    train_idx = np.sort(indices[:train_end]).tolist()
    val_idx = np.sort(indices[train_end:val_end]).tolist()
    test_idx = np.sort(indices[val_end:]).tolist()
    if not train_idx or not val_idx or not test_idx:
        raise ValueError("split produced an empty partition")
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def compute_normalization_stats(x_spec: np.ndarray, mask: np.ndarray, train_indices: list[int]) -> dict[str, Any]:
    train_spec = x_spec[np.asarray(train_indices)]
    train_mask = mask[np.asarray(train_indices)]
    valid = ~train_mask
    if valid.sum() == 0:
        raise ValueError("no valid pixels found in train split for normalization")
    pixels = train_spec[valid]
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist(), "clip_min": -3.0, "clip_max": 3.0}


def apply_normalization(x_spec: np.ndarray, mask: np.ndarray, stats: dict[str, Any], clip_min: float, clip_max: float) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, -1)
    normalized = (x_spec - mean) / std
    normalized = np.clip(normalized, clip_min, clip_max).astype(np.float32)
    normalized[mask] = 0.0
    return normalized


def _allocate_exact_subset(counts_by_class: dict[int, int], subset_size: int) -> tuple[dict[int, int], list[int]]:
    present_classes = sorted(counts_by_class)
    allocations = {cls: 0 for cls in present_classes}
    uncovered: list[int] = []
    if subset_size <= 0:
        return allocations, present_classes
    if subset_size >= len(present_classes):
        for cls in present_classes:
            allocations[cls] = 1
        remaining = subset_size - len(present_classes)
        if remaining <= 0:
            return allocations, uncovered
        frequencies = np.asarray([counts_by_class[cls] for cls in present_classes], dtype=np.float64)
        proportions = frequencies / frequencies.sum()
        raw = proportions * remaining
        floors = np.floor(raw).astype(int)
        for cls, floor_value in zip(present_classes, floors):
            allocations[cls] += int(floor_value)
        left = remaining - int(floors.sum())
        remainders = sorted(
            ((raw[i] - floors[i], present_classes[i]) for i in range(len(present_classes))),
            key=lambda item: (-item[0], item[1]),
        )
        for _, cls in remainders[:left]:
            allocations[cls] += 1
        return allocations, uncovered
    sorted_classes = sorted(present_classes, key=lambda cls: (-counts_by_class[cls], cls))
    covered = sorted_classes[:subset_size]
    uncovered = sorted_classes[subset_size:]
    for cls in covered:
        allocations[cls] = 1
    return allocations, uncovered


def create_label_subset(train_indices: list[int], y: np.ndarray, label_fraction: float, seed: int) -> dict[str, Any]:
    subset_size = max(1, int(round(len(train_indices) * label_fraction)))
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx in train_indices:
        class_to_indices[int(y[idx])].append(int(idx))
    allocations, uncovered = _allocate_exact_subset({k: len(v) for k, v in class_to_indices.items()}, subset_size)
    remaining_capacity = {cls: max(0, len(indices) - allocations[cls]) for cls, indices in class_to_indices.items()}
    allocated_total = sum(allocations.values())
    if allocated_total < subset_size:
        deficit = subset_size - allocated_total
        expandable = [cls for cls, cap in remaining_capacity.items() if cap > 0]
        cursor = 0
        while deficit > 0 and expandable:
            cls = expandable[cursor % len(expandable)]
            if remaining_capacity[cls] > 0:
                allocations[cls] += 1
                remaining_capacity[cls] -= 1
                deficit -= 1
            expandable = [candidate for candidate in expandable if remaining_capacity[candidate] > 0]
            cursor += 1
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    class_rows: list[dict[str, Any]] = []
    for cls in sorted(class_to_indices):
        indices = np.asarray(class_to_indices[cls], dtype=np.int64)
        rng.shuffle(indices)
        take = min(len(indices), allocations[cls])
        selected.extend(indices[:take].tolist())
        class_rows.append(
            {
                "class_id": int(cls),
                "train_count": int(len(class_to_indices[cls])),
                "allocated_count": int(allocations[cls]),
                "sampled_count": int(take),
            }
        )
    selected = sorted(selected)
    label_rows = [{"index": int(idx), "label": int(y[idx]), "split": "train"} for idx in selected]
    return {
        "subset_size_target": int(subset_size),
        "subset_size_actual": int(len(selected)),
        "selected_indices": selected,
        "selected_rows": label_rows,
        "class_allocation_rows": class_rows,
        "uncovered_present_classes": [int(cls) for cls in uncovered],
        "seed": int(seed),
    }


def _round_robin_sample(group_to_pairs: dict[Any, list[tuple[int, int]]], target: int, seed: int) -> list[tuple[int, int]]:
    if target <= 0:
        return []
    rng = np.random.default_rng(seed)
    pools: dict[Any, list[tuple[int, int]]] = {}
    for key, pairs in group_to_pairs.items():
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        pools[key] = shuffled
    selected: list[tuple[int, int]] = []
    group_keys = sorted(pools, key=lambda item: str(item))
    while len(selected) < target and any(pools[key] for key in group_keys):
        for key in group_keys:
            if not pools[key]:
                continue
            selected.append(pools[key].pop())
            if len(selected) >= target:
                break
    return selected


def create_pairwise_constraints(
    selected_indices: list[int],
    y: np.ndarray,
    seed: int,
    must_link_multiplier: int = 4,
    cannot_link_multiplier: int = 4,
) -> dict[str, Any]:
    selected_indices = sorted(int(idx) for idx in selected_indices)
    labels = {int(idx): int(y[idx]) for idx in selected_indices}
    must_by_class: dict[int, list[tuple[int, int]]] = defaultdict(list)
    cannot_by_class_pair: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for offset_i, idx_i in enumerate(selected_indices):
        for idx_j in selected_indices[offset_i + 1 :]:
            pair = (int(idx_i), int(idx_j))
            label_i = labels[idx_i]
            label_j = labels[idx_j]
            if label_i == label_j:
                must_by_class[label_i].append(pair)
            else:
                key = tuple(sorted((label_i, label_j)))
                cannot_by_class_pair[key].append(pair)
    all_must = sum(len(v) for v in must_by_class.values())
    all_cannot = sum(len(v) for v in cannot_by_class_pair.values())
    n_labeled = len(selected_indices)
    target_must = min(all_must, must_link_multiplier * n_labeled)
    target_cannot = min(all_cannot, cannot_link_multiplier * n_labeled)
    sampled_must = _round_robin_sample(must_by_class, target_must, seed)
    sampled_cannot = _round_robin_sample(cannot_by_class_pair, target_cannot, seed + 1)
    must_set = {tuple(sorted(pair)) for pair in sampled_must}
    cannot_set = {tuple(sorted(pair)) for pair in sampled_cannot}
    conflicts = sorted(must_set & cannot_set)
    if conflicts:
        must_set -= set(conflicts)
        cannot_set -= set(conflicts)
    sampled_must = sorted(must_set)
    sampled_cannot = sorted(cannot_set)
    must_count_by_class = {str(cls): len([pair for pair in sampled_must if labels[pair[0]] == cls]) for cls in sorted(must_by_class)}
    cannot_count_by_pair = {
        f"{pair[0]}-{pair[1]}": len(
            [candidate for candidate in sampled_cannot if tuple(sorted((labels[candidate[0]], labels[candidate[1]]))) == pair]
        )
        for pair in sorted(cannot_by_class_pair)
    }
    return {
        "seed": int(seed),
        "n_labeled": int(n_labeled),
        "must_link_multiplier": int(must_link_multiplier),
        "cannot_link_multiplier": int(cannot_link_multiplier),
        "all_feasible_must_link": int(all_must),
        "all_feasible_cannot_link": int(all_cannot),
        "target_must_link": int(target_must),
        "target_cannot_link": int(target_cannot),
        "actual_must_link": int(len(sampled_must)),
        "actual_cannot_link": int(len(sampled_cannot)),
        "must_link": [{"i": int(i), "j": int(j)} for i, j in sampled_must],
        "cannot_link": [{"i": int(i), "j": int(j)} for i, j in sampled_cannot],
        "must_link_count_by_class": must_count_by_class,
        "cannot_link_count_by_class_pair": cannot_count_by_pair,
        "conflict_count": int(len(conflicts)),
        "conflicts": [{"i": int(i), "j": int(j)} for i, j in conflicts],
        "deduplicated_pairs": int(len(sampled_must) + len(sampled_cannot)),
    }


def save_canonical_protocol(
    *,
    canonical_dir: str | Path,
    split_manifest: dict[str, Any],
    normalization_stats: dict[str, Any],
    label_subset: dict[str, Any],
    pairwise_constraints: dict[str, Any],
    invalid_samples: list[dict[str, Any]],
) -> CanonicalProtocolArtifacts:
    canonical_path = Path(canonical_dir)
    split_path = canonical_path / "split_manifest.json"
    stats_path = canonical_path / "normalization_stats.json"
    label_csv_path = canonical_path / "label_subset.csv"
    label_json_path = canonical_path / "label_subset.json"
    constraints_path = canonical_path / "pairwise_constraints.json"
    invalid_path = canonical_path / "invalid_samples.json"
    save_json(split_path, split_manifest)
    save_json(stats_path, normalization_stats)
    save_csv_rows(label_csv_path, label_subset["selected_rows"])
    save_json(label_json_path, label_subset)
    save_json(constraints_path, pairwise_constraints)
    save_json(invalid_path, invalid_samples)
    return CanonicalProtocolArtifacts(
        split_manifest_path=split_path,
        normalization_stats_path=stats_path,
        label_subset_csv_path=label_csv_path,
        label_subset_json_path=label_json_path,
        pairwise_constraints_path=constraints_path,
        invalid_samples_path=invalid_path,
    )
