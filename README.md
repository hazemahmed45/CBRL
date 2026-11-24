# CBRL: Contrastive Bootstrap Representation Learning

Self-supervised pre-training of a ViT-L/16 for agricultural images, followed by fine-tuning on
10% of the labels. CBRL trains a BYOL network (online encoder, projector and predictor with an
EMA target) and adds an NT-Xent (InfoNCE) loss between the online projections of the two views:

```
loss = 1.0 * BYOL(p1, p2 | t1, t2) + 1.0 * NT-Xent(z1, z2; tau = 0.5)
```

The whole method is two files: `train.py` (pre-training and fine-tuning) and `prepare_data.py`
(datasets). Every setting is in `config.yaml`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned versions used for the thesis runs
```

## Data

```bash
python prepare_data.py deepweeds            # direct download
python prepare_data.py plantdoc             # direct download
python prepare_data.py plantvillage         # Kaggle CLI, or --source-dir <downloaded copy>
python prepare_data.py new_plant_diseases   # Kaggle CLI, or --source-dir <downloaded copy>
python prepare_data.py lucas_cover_photos   # 195,814 photos from the JRC FTP, resumable
```

Each command builds `data/<Name>/{train,val,test}/<class>/`, writes the 10% label splits for seeds
0, 1 and 2 to `data/splits/<dataset>/`, and compares both with checksums of the thesis data. It
prints `OK` per split when the data are identical. Datasets without a test split are split
70/15/15 per class with seed 0; LUCAS uses the level-1 land-cover code as the label and the exact
photo list in `manifests/lucas2018_images.txt.gz`.

## Train

```bash
python train.py all --dataset deepweeds --seed 0        # pre-train, then fine-tune and test
python train.py pretrain --dataset deepweeds --seed 0   # pre-training only
python train.py finetune --dataset deepweeds --seed 0   # fine-tune outputs/deepweeds/cbrl_seed0/pretrain_best.pt
```

- Pre-training runs `round(train images x 10,000 / 38,012)` steps of 1,024 images (DeepWeeds 3,224;
  PlantVillage 10,000; LUCAS 35,948), AdamW at 5e-4 with 5% warm-up and cosine decay. The
  checkpoint with the lowest training loss (checked every 50 steps) is the one fine-tuned.
- Fine-tuning trains the encoder and a linear head for 50 epochs on the 10% label split of the
  same seed and reports the test metrics of the epoch with the best validation macro-F1.
- `--resume` continues an interrupted stage from its `*_last.pt`.

Ablation arms of the thesis:

```bash
python train.py all --dataset deepweeds --seed 0 --run-name contrastive_only_seed0 --bootstrap-loss-weight 0
python train.py all --dataset deepweeds --seed 0 --run-name bootstrap_only_seed0 --contrastive-loss-weight 0
```

Outputs go to `outputs/<dataset>/<run>/`: `pretrain_{best,last}.pt`, `pretrain.jsonl`,
`finetune_{best,last}.pt`, `finetune.jsonl` and `results.json` (test accuracy, macro-F1,
weighted F1 and balanced accuracy).

## Hardware

The batch of 1,024 ViT-L/16 images was trained on one 80 GB GPU (about 8 s per step with
gradient checkpointing). With less memory, lower `batch_size` and raise `gradient_accumulation` to
keep 1,024 images per step. This changes the BatchNorm batches and the InfoNCE negatives, so the
results are then close to, but not the same as, the thesis runs.

## Reproducibility

`train.py` is a port of the thesis code (method `cbrl` of the `agri_ssl` package). It was checked
against that code on CPU with the pinned packages:

- identical initial weights for the same seed;
- identical losses for five optimisation steps, and identical weights afterwards (EMA target and
  BatchNorm statistics included);
- identical augmented views under the same Albumentations seed;
- identical test metrics for a short fine-tuning run;
- identical dataset splits and label splits (the checksums in `prepare_data.py`).

The pre-training augmentations draw their randomness from Albumentations, which seeds itself from
the operating system, as the thesis code did. Two runs with the same seed therefore differ by
ordinary run-to-run noise; the thesis reports three seeds per setting for this reason.
