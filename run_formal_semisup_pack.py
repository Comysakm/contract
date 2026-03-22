from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.prepare_canonical_protocol import prepare_canonical_protocol

from formal_semisup.data.dataset import load_canonical_dataset
from formal_semisup.methods.cop_kmeans import run_cop_kmeans
from formal_semisup.methods.semi_supervised_spectral import run_semi_supervised_spectral
from formal_semisup.methods.sdec import run_sdec
from formal_semisup.methods.supervised import run_supervised_experiment
from formal_semisup.reporting.aggregate import build_pack_report
from formal_semisup.utils.config import load_config
from formal_semisup.utils.io import ensure_dir, save_json
from formal_semisup.utils.runtime import experiment_complete, resolve_device, resolve_server_name, write_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "formal_semisup_pack.yaml"))
    parser.add_argument("--pack-id", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--data-x", default=None)
    parser.add_argument("--data-y", default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_variants(config: dict, variants_arg: str | None) -> list[str]:
    if variants_arg:
        return [item.strip() for item in variants_arg.split(",") if item.strip()]
    return list(config["variants"]["table1"]) + list(config["variants"]["table2"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pack_id = args.pack_id or config["project"]["default_pack_id"]
    server_name = resolve_server_name(config["project"]["default_server_name"])
    device = resolve_device(args.device)
    pack_root = ensure_dir(ROOT / "runs" / "formal_semisup_pack" / pack_id / server_name)
    canonical_dir = ensure_dir(pack_root / "canonical")
    x_path = args.data_x or config["data"]["x_path"]
    y_path = args.data_y or config["data"]["y_path"]
    if not (canonical_dir / "canonical_manifest.json").exists():
        prepare_canonical_protocol(args.config, canonical_dir, x_path, y_path)
    dataset = load_canonical_dataset(canonical_dir)
    variants = resolve_variants(config, args.variants)
    completed = []
    failed = []
    command = " ".join(sys.argv)
    for variant in variants:
        exp_dir = pack_root / variant
        if args.reuse_existing and experiment_complete(exp_dir):
            completed.append(variant)
            continue
        ensure_dir(exp_dir)
        write_status(
            exp_dir,
            state="running",
            command=command,
            seeds={
                "split_seed": config["protocol"]["split_seed"],
                "subset_seed": config["protocol"]["subset_seed"],
                "constraint_seed": config["protocol"]["constraint_seed"],
            },
        )
        try:
            if variant in {"supervised_lstm", "supervised_rnn", "supervised_gru", "supervised_transformer", "cae_pretrain_classifier"}:
                run_supervised_experiment(variant=variant, dataset=dataset, config=config, exp_dir=exp_dir, device=device)
            elif variant == "cop_kmeans":
                run_cop_kmeans(dataset=dataset, config=config, exp_dir=exp_dir)
            elif variant == "semi_supervised_spectral":
                run_semi_supervised_spectral(dataset=dataset, config=config, exp_dir=exp_dir)
            elif variant == "sdec":
                run_sdec(dataset=dataset, config=config, exp_dir=exp_dir, device=device)
            else:
                raise ValueError(f"unknown variant: {variant}")
            write_status(
                exp_dir,
                state="completed",
                command=command,
                seeds={
                    "split_seed": config["protocol"]["split_seed"],
                    "subset_seed": config["protocol"]["subset_seed"],
                    "constraint_seed": config["protocol"]["constraint_seed"],
                },
            )
            completed.append(variant)
        except Exception as exc:
            write_status(
                exp_dir,
                state="failed",
                command=command,
                seeds={
                    "split_seed": config["protocol"]["split_seed"],
                    "subset_seed": config["protocol"]["subset_seed"],
                    "constraint_seed": config["protocol"]["constraint_seed"],
                },
                failure_reason=str(exc),
            )
            failed.append({"variant": variant, "error": str(exc)})
    manifest = {
        "pack_id": pack_id,
        "server_name": server_name,
        "pack_root": str(pack_root),
        "canonical_dir": str(canonical_dir),
        "device": device,
        "variants_requested": variants,
        "completed_variants": completed,
        "failed_variants": failed,
        "data_x": x_path,
        "data_y": y_path,
    }
    save_json(pack_root / "manifest.json", manifest)
    experiment_dirs = [pack_root / variant for variant in variants if (pack_root / variant).exists()]
    build_pack_report(pack_root, manifest, experiment_dirs)
    if failed:
        raise RuntimeError(f"pack completed with failures: {failed}")


if __name__ == "__main__":
    main()
