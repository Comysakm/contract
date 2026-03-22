# TimeSen2Crop Formal Semi-Supervised Pack

Formal comparison framework for TimeSen2Crop-style remote sensing time-series classification and semi-supervised clustering under a unified low-label protocol.

## Scope

This project implements:

- `supervised_lstm`
- `supervised_rnn`
- `supervised_gru`
- `supervised_transformer`
- `cae_pretrain_classifier`
- `cop_kmeans`
- `semi_supervised_spectral`
- `sdec`

The protocol is fixed to:

- `full-data / unfiltered`
- `clean_cloud_threshold = 1.0`
- `min_valid_steps = 0`
- `train/val/test = 0.70 / 0.15 / 0.15`
- `split_seed = 42`
- `label_fraction = 0.01`
- `strict_label_training_only = 1`

## Default Data Paths

- `/home/mw/input/TimeSen2Crop1854/trainx9.npy`
- `/home/mw/input/TimeSen2Crop1854/trainy9.npy`

You can override them with CLI flags.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python run_formal_semisup_pack.py --pack-id formal_v1
python run_formal_semisup_pack.py --pack-id formal_v1 --variants supervised_lstm,sdec --reuse-existing
python run_formal_semisup_pack.py --pack-id formal_v1 --data-x /custom/trainx9.npy --data-y /custom/trainy9.npy
```

## Outputs

Results are written under:

```text
runs/formal_semisup_pack/<pack-id>/<server-name>/
```

Key pack-level outputs:

- `manifest.json`
- `summary.csv`
- `summary_table1_supervised.csv`
- `summary_table2_semisup_clustering.csv`
- `report.md`
