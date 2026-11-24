"""CBRL: Contrastive Bootstrap Representation Learning.

One file that pre-trains a ViT with CBRL and fine-tunes it for image classification.

    python train.py pretrain --dataset deepweeds --seed 0
    python train.py finetune --dataset deepweeds --seed 0
    python train.py all      --dataset deepweeds --seed 0     # both stages

CBRL trains a BYOL network (online encoder + projector + predictor, EMA target) and adds an
NT-Xent (InfoNCE) loss between the online projections of the two views:

    loss = bootstrap_weight * BYOL(p1, p2 | t1, t2) + contrastive_weight * NT-Xent(z1, z2)

This is a line-by-line port of the thesis code path (agri_ssl, method "cbrl"): the same modules
are built in the same order, random numbers are drawn in the same order and every
hyperparameter is in config.yaml. With the pinned packages in requirements.txt it reproduces
the thesis runs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

# name used in run names -> folder under data_root
DATASETS = {
    "deepweeds": "DeepWeeds",
    "plantvillage": "PlantVillage",
    "plantdoc": "PlantDoc",
    "new_plant_diseases": "NewPlantDiseases",
    "lucas_cover_photos": "LUCASCoverPhotos",
}
VIT_ARCHITECTURES = {
    "vit_t_16": "vit_tiny_patch16_224",
    "vit_s_16": "vit_small_patch16_224",
    "vit_b_16": "vit_base_patch16_224",
    "vit_l_16": "vit_large_patch16_224",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- utilities
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    warmup_steps, total_steps = max(0, warmup_steps), max(1, total_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def optimizer_to_device(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def log_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), **row}, sort_keys=True) + "\n")


def progress_bar(iterable=None, *, total=None, desc: str, unit: str = "batch", leave: bool = True):
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave, dynamic_ncols=True,
                mininterval=0.5, file=sys.stdout)


# ----------------------------------------------------------------------------- model
class VitEncoder(nn.Module):
    """timm ViT without its classifier; returns the pooled (class-token) feature."""

    def __init__(self, architecture: str, pretrained: bool = False, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        import timm

        name = VIT_ARCHITECTURES[architecture]
        try:
            self.model = timm.create_model(name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        except TypeError:
            self.model = timm.create_model(name, pretrained=pretrained, num_classes=0)
            if hasattr(self.model, "set_input_size"):
                self.model.set_input_size(img_size=224)
        self.feature_dim = int(self.model.num_features)
        if gradient_checkpointing:
            self.set_gradient_checkpointing(True)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.model.set_grad_checkpointing(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        if features.ndim > 2:
            features = features.mean(dim=tuple(range(2, features.ndim)))
        return features


class ProjectionHead(nn.Module):
    """Linear -> BatchNorm -> ReLU -> Linear."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def update_ema(source: nn.Module, target: nn.Module, momentum: float) -> None:
    for source_param, target_param in zip(source.parameters(), target.parameters(), strict=False):
        target_param.data.mul_(momentum).add_(source_param.data, alpha=1.0 - momentum)
    for source_buffer, target_buffer in zip(source.buffers(), target.buffers(), strict=False):
        if torch.is_floating_point(target_buffer):
            target_buffer.data.mul_(momentum).add_(source_buffer.data, alpha=1.0 - momentum)
        else:
            target_buffer.data.copy_(source_buffer.data)


def cosine_momentum(base: float, step: int, total_steps: int) -> float:
    progress = min(1.0, step / max(1, total_steps))
    return 1.0 - (1.0 - base) * (math.cos(math.pi * progress) + 1.0) / 2.0


class CBRLModel(nn.Module):
    """Online encoder/projector/predictor and an EMA target encoder/projector (BYOL layout)."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.online_encoder = VitEncoder(cfg["architecture"], gradient_checkpointing=bool(cfg["gradient_checkpointing"]))
        self.online_projector = ProjectionHead(self.online_encoder.feature_dim, cfg["projection_hidden_dim"],
                                               cfg["projection_dim"])
        self.predictor = ProjectionHead(cfg["projection_dim"], cfg["prediction_hidden_dim"], cfg["projection_dim"])
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.set_gradient_checkpointing(False)
        self.target_projector = copy.deepcopy(self.online_projector)
        for module in (self.target_encoder, self.target_projector):
            for param in module.parameters():
                param.requires_grad = False

    def online(self, x):
        features = self.online_encoder(x)
        projection = self.online_projector(features)
        return projection, self.predictor(projection)

    @torch.no_grad()
    def target(self, x):
        return self.target_projector(self.target_encoder(x))

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        update_ema(self.online_encoder, self.target_encoder, momentum)
        update_ema(self.online_projector, self.target_projector, momentum)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = z1.shape[0]
    with torch.amp.autocast(device_type=z1.device.type, enabled=False):
        z = F.normalize(torch.cat([z1.float(), z2.float()], dim=0), dim=1)
        logits = z @ z.T / temperature
        logits = logits.masked_fill(torch.eye(2 * batch_size, device=z.device, dtype=torch.bool),
                                    torch.finfo(logits.dtype).min)
        labels = (torch.arange(2 * batch_size, device=z.device) + batch_size) % (2 * batch_size)
        return F.cross_entropy(logits, labels)


def negative_cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target.detach(), dim=1)
    return 2.0 - 2.0 * (prediction * target).sum(dim=1).mean()


def cbrl_loss(model: CBRLModel, view1, view2, cfg: dict):
    z1, p1 = model.online(view1)
    z2, p2 = model.online(view2)
    t1, t2 = model.target(view1), model.target(view2)
    bootstrap = 0.5 * (negative_cosine_similarity(p1, t2) + negative_cosine_similarity(p2, t1))
    contrastive = nt_xent_loss(z1, z2, temperature=float(cfg["temperature"]))
    loss = float(cfg["bootstrap_loss_weight"]) * bootstrap + float(cfg["contrastive_loss_weight"]) * contrastive
    return loss, {"loss_bootstrap": float(bootstrap.detach()), "loss_contrastive": float(contrastive.detach())}


# ----------------------------------------------------------------------------- data
class TwoViewTransform:
    def __init__(self, first, second) -> None:
        self.first, self.second = first, second

    def __call__(self, image):
        return self.first(image), self.second(image)


class AlbumentationsView:
    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, image):
        return self.transform(image=np.asarray(image.convert("RGB")))["image"]


def byol_views(image_size: int, min_scale: float, vertical_flip_prob: float, noise_prob: float) -> TwoViewTransform:
    """Asymmetric BYOL views (Albumentations): view 1 always blurred, view 2 sometimes solarised."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    def crop():
        kwargs = {"scale": (min_scale, 1.0), "ratio": (0.75, 1.3333333333333333), "p": 1.0}
        try:
            return A.RandomResizedCrop(size=(image_size, image_size), **kwargs)
        except TypeError:
            return A.RandomResizedCrop(height=image_size, width=image_size, **kwargs)

    def view(blur_prob: float, solarize_prob: float):
        pipeline = A.Compose([
            crop(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=vertical_flip_prob),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=blur_prob),
            A.Solarize(p=solarize_prob),
            A.GaussNoise(p=noise_prob),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
        return AlbumentationsView(pipeline)

    return TwoViewTransform(view(blur_prob=1.0, solarize_prob=0.0), view(blur_prob=0.1, solarize_prob=0.2))


def supervised_train_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def supervised_eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


class UnlabeledImageDataset(Dataset):
    """All images under root (sorted); the label is the image index."""

    def __init__(self, root: Path, transform) -> None:
        self.transform = transform
        self.samples = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.samples:
            raise FileNotFoundError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.transform(Image.open(self.samples[index]).convert("RGB")), index


def align_classes(dataset: ImageFolder, reference: ImageFolder) -> ImageFolder:
    """Give an evaluation split the training split's class indices."""
    unknown = set(dataset.classes).difference(reference.class_to_idx)
    if unknown:
        raise ValueError(f"Evaluation split contains classes absent from training: {sorted(unknown)}")
    samples = [(path, reference.class_to_idx[dataset.classes[target]]) for path, target in dataset.samples]
    dataset.samples = dataset.imgs = samples
    dataset.targets = [target for _, target in samples]
    dataset.classes, dataset.class_to_idx = list(reference.classes), dict(reference.class_to_idx)
    return dataset


# ----------------------------------------------------------------------------- pre-training
def pretrain(args, config: dict) -> Path:
    cfg = dict(config["pretrain"])
    for key in ("contrastive_loss_weight", "bootstrap_loss_weight"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    out_dir = run_dir(args, config)
    device = resolve_device(args.device)
    seed_everything(args.seed)

    train_set = UnlabeledImageDataset(
        Path(config["data_root"]) / DATASETS[args.dataset] / "train",
        byol_views(int(cfg["image_size"]), float(cfg["min_crop_scale"]), float(cfg["vertical_flip_prob"]),
                   float(cfg["noise_prob"])),
    )
    max_steps = args.max_steps or max(1, round(len(train_set) * cfg["reference_steps"] / cfg["reference_dataset_size"]))
    batch_size, grad_accum = int(cfg["batch_size"]), int(cfg["gradient_accumulation"])
    optimizer_steps = math.ceil(max_steps / grad_accum)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=int(cfg["num_workers"]),
                        pin_memory=device.type == "cuda", drop_last=len(train_set) >= batch_size,
                        persistent_workers=int(cfg["num_workers"]) > 0,
                        prefetch_factor=2 if int(cfg["num_workers"]) > 0 else None)

    model = CBRLModel(cfg).to(device)
    print(f"device={device} images={len(train_set):,} max_steps={max_steps:,} batch_size={batch_size} "
          f"gradient_accumulation={grad_accum} parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["lr"]),
                                  weight_decay=float(cfg["weight_decay"]), betas=(0.9, 0.999))
    scheduler = cosine_with_warmup(optimizer, int(optimizer_steps * float(cfg["warmup_fraction"])), optimizer_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(cfg["amp"]))

    last_path, best_path = out_dir / "pretrain_last.pt", out_dir / "pretrain_best.pt"
    step, best_loss = 0, float("inf")
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, best_loss = int(state["step"]), float(state["best_metric"])
        print(f"Resumed pre-training from {last_path} at step {step:,}")

    def payload(**extra):
        return {"encoder": model.online_encoder.state_dict(), "encoder_source": "online",
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "config": cfg, "dataset": args.dataset, "seed": args.seed, "step": step, **extra}

    every = int(cfg["checkpoint_every_steps"])
    interval = math.lcm(every, grad_accum)
    log_path = out_dir / "pretrain.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = progress_bar(total=max_steps, desc=f"Pretrain {args.dataset} seed {args.seed}", unit="step")
    progress.update(min(step, max_steps))
    try:
        while step < max_steps:
            for (view1, view2), _ in loader:
                view1 = view1.to(device, non_blocking=True)
                view2 = view2.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    raw_loss, parts = cbrl_loss(model, view1, view2, cfg)
                    loss = raw_loss / grad_accum
                scaler.scale(loss).backward()

                stepped = (step + 1) % grad_accum == 0 or step + 1 >= max_steps
                if stepped:
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    model.update_target(cosine_momentum(float(cfg["ema_base_momentum"]), step, max_steps))

                step += 1
                loss_value = float(raw_loss.detach())
                progress.update()
                progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                if step == 1 or step % every == 0 or step >= max_steps:
                    log_jsonl(log_path, {"step": step, "loss": loss_value,
                                         "learning_rate": optimizer.param_groups[0]["lr"], **parts})
                # Checkpoints every `every` steps; "best" = lowest training loss seen at a checkpoint step.
                save_now = stepped and (step <= grad_accum or step % interval == 0 or step >= max_steps)
                if save_now and loss_value < best_loss:
                    best_loss = loss_value
                    save_checkpoint(best_path, payload(monitor={"name": "train/loss", "mode": "min", "value": best_loss}))
                if save_now:
                    save_checkpoint(last_path, payload(best_metric=best_loss))
                if step >= max_steps:
                    break
        save_checkpoint(last_path, payload(best_metric=best_loss))
    finally:
        progress.close()
    return best_path


# ----------------------------------------------------------------------------- fine-tuning
@torch.no_grad()
def classification_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict:
    preds = logits.argmax(dim=1)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=logits.device)
    for target, pred in zip(targets, preds, strict=False):
        confusion[target.long(), pred.long()] += 1
    tp = confusion.diag()
    precision = tp / confusion.sum(dim=0).clamp_min(1.0)
    recall = tp / confusion.sum(dim=1).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    support = confusion.sum(dim=1)
    return {"accuracy": (preds == targets).float().mean().item(), "macro_f1": f1.mean().item(),
            "weighted_f1": (f1 * support / support.sum().clamp_min(1.0)).sum().item(),
            "balanced_accuracy": recall.mean().item()}


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, desc: str) -> dict:
    model.eval()
    all_logits, all_targets, total_loss = [], [], 0.0
    loss_fn = nn.CrossEntropyLoss()
    for images, targets in progress_bar(loader, desc=desc, leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(loss_fn(logits, targets).item()) * images.shape[0]
        all_logits.append(logits.detach())
        all_targets.append(targets.detach())
    metrics = classification_metrics(torch.cat(all_logits), torch.cat(all_targets), num_classes)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: VitEncoder, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.feature_dim, num_classes))

    def forward(self, x):
        return self.head(self.encoder(x))


def finetune(args, config: dict, checkpoint: Path | None = None) -> dict:
    cfg = config["finetune"]
    out_dir = run_dir(args, config)
    checkpoint = Path(args.checkpoint) if args.checkpoint else (checkpoint or out_dir / "pretrain_best.pt")
    fraction = f"{round(float(cfg['label_fraction']) * 100)}percent"
    split = Path(args.split) if args.split else Path(config["data_root"]) / "splits" / args.dataset / f"seed_{args.seed}_{fraction}.json"
    if not split.exists():
        raise FileNotFoundError(f"{split} not found. Run: python prepare_data.py {args.dataset}")
    seed_everything(args.seed)
    device = resolve_device(args.device)

    image_size = int(cfg["image_size"])
    root = Path(config["data_root"]) / DATASETS[args.dataset]
    train_set = ImageFolder(str(root / "train"), transform=supervised_train_transform(image_size))
    val_set = align_classes(ImageFolder(str(root / "val"), transform=supervised_eval_transform(image_size)), train_set)
    test_set = align_classes(ImageFolder(str(root / "test"), transform=supervised_eval_transform(image_size)), train_set)
    num_classes = len(train_set.classes)
    train_set = Subset(train_set, json.loads(split.read_text())["indices"])

    loader = lambda dataset, batch_size, shuffle: DataLoader(  # noqa: E731
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda")
    train_loader = loader(train_set, int(cfg["batch_size"]), True)
    val_loader = loader(val_set, int(cfg["eval_batch_size"]), False)
    test_loader = loader(test_set, int(cfg["eval_batch_size"]), False)

    encoder = VitEncoder(cfg["architecture"], pretrained=False, gradient_checkpointing=False)
    last_path, best_path = out_dir / "finetune_last.pt", out_dir / "finetune_best.pt"
    resume = args.resume and last_path.exists()
    if not resume:
        state = torch.load(checkpoint, map_location="cpu")
        missing, unexpected = encoder.load_state_dict(state.get("encoder", state), strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected encoder checkpoint keys: {unexpected[:10]}")
        if missing:
            print(f"Warning: missing encoder keys while loading {checkpoint}: {missing[:10]}")
        print(f"Loaded pre-trained encoder from {checkpoint}")
    model = EncoderClassifier(encoder, num_classes, float(cfg["dropout"])).to(device)

    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": float(cfg["encoder_lr"])},
                                   {"params": model.head.parameters(), "lr": float(cfg["head_lr"])}],
                                  weight_decay=float(cfg["weight_decay"]))
    epochs, steps_per_epoch = int(cfg["epochs"]), max(1, len(train_loader))
    scheduler = cosine_with_warmup(optimizer, int(cfg["warmup_epochs"]) * steps_per_epoch, epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(cfg["amp"]))
    loss_fn = nn.CrossEntropyLoss()
    grad_accum = int(cfg["gradient_accumulation"])

    best_f1, start_epoch, latest_val = float("-inf"), 0, None
    if resume:
        state = torch.load(last_path, map_location="cpu")
        model.encoder.load_state_dict(state["encoder"])
        model.head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, latest_val, best_f1 = int(state["epoch"]), state["val"], float(state["best_metric"])
        print(f"Resumed fine-tuning from {last_path} at epoch {start_epoch}")

    def payload(epoch: int, val: dict, **extra):
        return {"encoder": model.encoder.state_dict(), "head": model.head.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "config": cfg, "num_classes": num_classes,
                "epoch": epoch, "val": val, **extra}

    log_path = out_dir / "finetune.jsonl"
    bar = progress_bar(total=epochs * len(train_loader), desc=f"Fine-tune {args.dataset} seed {args.seed}")
    bar.update(min(start_epoch, epochs) * len(train_loader))
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for batch_index, (images, targets) in enumerate(train_loader, start=1):
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    loss = loss_fn(model(images), targets) / grad_accum
                scaler.scale(loss).backward()
                if batch_index % grad_accum == 0 or batch_index == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                running_loss += float(loss.item()) * grad_accum * images.shape[0]
                bar.update()
                bar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{running_loss / max(1, batch_index * images.shape[0]):.4f}")

            val = evaluate(model, val_loader, device, num_classes, f"Validate epoch {epoch + 1}")
            latest_val = val
            log_jsonl(log_path, {"epoch": epoch + 1, "train_loss": running_loss / max(1, len(train_set)),
                                 "learning_rate": {"encoder": optimizer.param_groups[0]["lr"],
                                                   "head": optimizer.param_groups[1]["lr"]},
                                 "val": val, "split": str(split)})
            bar.set_postfix(epoch=f"{epoch + 1}/{epochs}", val_f1=f"{val['macro_f1']:.4f}")
            if val["macro_f1"] > best_f1:  # keep the epoch with the best validation macro-F1
                best_f1 = val["macro_f1"]
                save_checkpoint(best_path, payload(epoch + 1, val))
            save_checkpoint(last_path, payload(epoch + 1, latest_val, best_metric=best_f1))

        best = torch.load(best_path, map_location="cpu")
        model.encoder.load_state_dict(best["encoder"], strict=True)
        model.head.load_state_dict(best["head"], strict=True)
        test = evaluate(model, test_loader, device, num_classes, "Test classification")
        log_jsonl(log_path, {"test": test, "split": str(split), "test_checkpoint": "best", "test_epoch": best["epoch"]})
        (out_dir / "results.json").write_text(json.dumps(
            {"dataset": args.dataset, "seed": args.seed, "split": str(split), "pretrained_checkpoint": str(checkpoint),
             "best_epoch": best["epoch"], "val": best["val"], "test": test}, indent=2) + "\n")
        print(f"test={test}")
        return test
    finally:
        bar.close()


# ----------------------------------------------------------------------------- command line
def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu only when CPU training is intentional.")
    return torch.device(requested)


def run_dir(args, config: dict) -> Path:
    name = args.run_name or f"cbrl_seed{args.seed}"
    return Path(config["output_dir"]) / args.dataset / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["pretrain", "finetune", "all"])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--seed", type=int, default=0, help="pre-training seed, fine-tuning seed and label split")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--run-name", help="output folder under outputs/<dataset>/ (default cbrl_seed<seed>)")
    parser.add_argument("--contrastive-loss-weight", type=float, help="ablation: 0 gives bootstrap-only")
    parser.add_argument("--bootstrap-loss-weight", type=float, help="ablation: 0 gives contrastive-only")
    parser.add_argument("--checkpoint", help="encoder checkpoint to fine-tune (default <run>/pretrain_best.pt)")
    parser.add_argument("--split", help="label-fraction split JSON (default data/splits/<dataset>/seed_<seed>_10percent.json)")
    parser.add_argument("--max-steps", type=int, help="override the pre-training length (for smoke tests)")
    parser.add_argument("--resume", action="store_true", help="continue from <run>/*_last.pt")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint = pretrain(args, config) if args.stage in {"pretrain", "all"} else None
    if args.stage in {"finetune", "all"}:
        finetune(args, config, checkpoint)


if __name__ == "__main__":
    main()
