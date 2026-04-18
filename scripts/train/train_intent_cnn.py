"""Training script for Intent CNN.

Trains the MobileNetV3-Small + dual-head model defined in
src/perception/intent_cnn.py using the ROI dataset collected from the Jetson
and auto-labeled by scripts/autolabel.py.

Dataset structure expected:
    intent_dataset/
    ├── STATIONARY/  roi_*.jpg  → STATIONARY (0), dx=0.0, dy=0.0
    ├── APPROACHING/ roi_*.jpg  → APPROACHING (1), dx=0.0, dy=-0.6
    ├── DEPARTING/   roi_*.jpg  → DEPARTING (2), dx=0.0, dy=+0.6
    ├── CROSSING/    roi_*.jpg  → CROSSING (3), dx=±0.8, dy=0.0 (dx from cx shift)
    ├── FOLLOWING/   roi_*.jpg  → FOLLOWING (4), dx=0.0, dy=-0.3
    ├── ERRATIC/     roi_*.jpg  → ERRATIC (5), dx=0.0, dy=0.0
    └── uncertain/   roi_*.jpg  → SKIPPED (excluded from training)

Label mapping rationale
-----------------------
ROI autolabel uses depth and bbox spatial analysis to determine coarse 
intent. We map it to the 6 intent classes AND synthesise a plausible 
(dx, dy) ground-truth vector so the direction head gets a meaningful 
regression target.

For CROSSING, the direction (left vs right) is inferred dynamically by 
comparing the `cx` metadata across consecutive frames for the same track_id 
during dataset loading.

Usage:
    python scripts/train_intent_cnn.py
    python scripts/train_intent_cnn.py --dataset D:/nckh/context-aware/intent_dataset --epochs 30
    python scripts/train_intent_cnn.py --resume models/cnn_intent/checkpoint_ep10.pt

Output:
    models/cnn_intent/intent_v1.pt          ← best model (by val loss)
    models/cnn_intent/checkpoint_ep<N>.pt   ← periodic checkpoints
    models/cnn_intent/training_log.csv      ← per-epoch metrics
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [train] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CNN_INPUT_W = 128
CNN_INPUT_H = 256
NUM_INTENT_CLASSES = 6

LABEL_MAP = {
    # folder_name → (intent_class, default_dx, default_dy)
    "STATIONARY":  (0,  0.0,  0.0),
    "APPROACHING": (1,  0.0, -0.6),
    "DEPARTING":   (2,  0.0,  0.6),
    "CROSSING":    (3,  0.0,  0.0),  # dx will be resolved dynamically
    "FOLLOWING":   (4,  0.0, -0.3),
    "ERRATIC":     (5,  0.0,  0.0),
}

INTENT_NAMES = [
    "STATIONARY", "APPROACHING", "DEPARTING",
    "CROSSING", "FOLLOWING", "ERRATIC",
]


class ROIDataset(Dataset):
    """Reads labeled ROI images from the intent_dataset folder structure."""

    def __init__(self, root: Path, transform=None, max_samples: Optional[int] = None) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int, float, float]] = []

        from collections import defaultdict
        
        # Helper to parse cx and frame_id
        def parse_meta(p: Path) -> dict:
            parts = p.stem.split("_")
            if len(parts) >= 7 and parts[0] == "roi":
                return {"tid": parts[1], "cx": int(parts[2]), "fid": int(parts[6] if len(parts) == 8 else parts[3])}
            return None

        for folder, (intent_cls, base_dx, base_dy) in LABEL_MAP.items():
            folder_path = root / folder
            if not folder_path.exists():
                logger.warning("Label folder not found: %s — skipping", folder_path)
                continue
            imgs = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
            
            if folder != "CROSSING":
                for img_path in imgs:
                    self.samples.append((img_path, intent_cls, base_dx, base_dy))
            else:
                # Dynamic DX resolution for CROSSING
                tracks = defaultdict(list)
                for img_path in imgs:
                    meta = parse_meta(img_path)
                    if meta:
                        tracks[meta["tid"]].append((meta["fid"], meta["cx"], img_path))
                    else:
                        self.samples.append((img_path, intent_cls, base_dx, base_dy))
                
                for tid, frames in tracks.items():
                    frames.sort(key=lambda x: x[0])  # sort by frame_id
                    for i, (fid, cx, img_path) in enumerate(frames):
                        # determine direction from future frames if possible
                        lookahead = min(i + 15, len(frames) - 1)
                        if lookahead > i:
                            future_cx = frames[lookahead][1]
                            dx = 0.8 if future_cx > cx else -0.8
                        else:
                            dx = 0.0 # fallback
                        self.samples.append((img_path, intent_cls, dx, base_dy))

        if max_samples and len(self.samples) > max_samples:
            # Replay Buffer: keep all HUMAN labels, randomly subsample AUTO labels
            human_samples = [s for s in self.samples if s[0].name.startswith("human_")]
            auto_samples  = [s for s in self.samples if not s[0].name.startswith("human_")]
            
            auto_quota = max(0, max_samples - len(human_samples))
            if len(auto_samples) > auto_quota:
                random.shuffle(auto_samples)
                auto_samples = auto_samples[:auto_quota]
                
            self.samples = human_samples + auto_samples
            random.shuffle(self.samples)

        logger.info(
            "ROIDataset loaded: %d images from %s",
            len(self.samples), root,
        )
        self._log_class_distribution()

    def _log_class_distribution(self) -> None:
        from collections import Counter
        counts = Counter(s[1] for s in self.samples)
        for cls_id, n in sorted(counts.items()):
            logger.info("  Class %d (%s): %d images", cls_id, INTENT_NAMES[cls_id], n)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        img_path, intent_cls, dx, dy = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        intent_label = torch.tensor(intent_cls, dtype=torch.long)
        direction_gt = torch.tensor([dx, dy], dtype=torch.float32)
        return img, intent_label, direction_gt


def build_transforms(augment: bool):
    """ImageNet-normalised transforms.  Training uses augmentation."""
    base = [
        transforms.Resize((CNN_INPUT_H, CNN_INPUT_W)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ]
    if augment:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.05),
        ]
        # Apply augmentation BEFORE ToTensor
        return transforms.Compose(aug + base)
    return transforms.Compose(base)



def build_model(feature_dim: int = 576, freeze_backbone_blocks: int = 10) -> nn.Module:
    """Build MobileNetV3-Small + dual heads. Partially freeze backbone."""
    import torchvision.models as tv_models

    backbone = tv_models.mobilenet_v3_small(
        weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
    )
    backbone.classifier = nn.Identity()

    model = _IntentModel(backbone, feature_dim=feature_dim)

    # Freeze first N feature blocks — fine-tune only upper layers + heads.
    # This prevents catastrophic forgetting of ImageNet features with small data.
    params = list(backbone.features.parameters())
    for i, param in enumerate(params):
        if i < freeze_backbone_blocks:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d / %d parameters trainable (%.1f%%)", trainable, total, trainable/total*100)
    return model


class _IntentModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.intent_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_INTENT_CLASSES),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        feats = self.backbone.features(x)
        feats = self.pool(feats).flatten(1)
        return self.intent_head(feats), self.direction_head(feats)



class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def compute_class_weights(dataset: ROIDataset, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights to handle label imbalance."""
    from collections import Counter
    counts = Counter(s[1] for s in dataset.samples)
    total  = len(dataset)
    weights = torch.zeros(NUM_INTENT_CLASSES, dtype=torch.float32)
    for cls_id, n in counts.items():
        weights[cls_id] = total / (len(counts) * n)
    # Classes not in dataset get weight 1.0
    for cls_id in range(NUM_INTENT_CLASSES):
        if weights[cls_id] == 0:
            weights[cls_id] = 1.0
    return weights.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    intent_criterion: nn.Module,
    dir_criterion: nn.Module,
    lambda_dir: float,
    device: torch.device,
    scaler: Optional["torch.cuda.amp.GradScaler"],
    model_anchor: Optional[nn.Module] = None,
    ewc_lambda: float = 0.0,
) -> dict:
    model.train()
    loss_m   = AverageMeter()
    intent_m = AverageMeter()
    dir_m    = AverageMeter()
    ewc_m    = AverageMeter()
    acc_m    = AverageMeter()

    anchor_params = None
    if model_anchor is not None and ewc_lambda > 0:
        anchor_params = list(model_anchor.parameters())

    for imgs, intent_labels, dir_gt in loader:
        imgs          = imgs.to(device)
        intent_labels = intent_labels.to(device)
        dir_gt        = dir_gt.to(device)

        optimizer.zero_grad()

        if anchor_params is not None:
            # Simplified EWC: L2 distance to old weights (Starting Point regularization)
            # F_i is approximated as 1.0 here for stability on small datasets
            ewc_loss = 0.0
            for p, p_old in zip(model.parameters(), anchor_params):
                if p.requires_grad:
                    ewc_loss += torch.sum((p - p_old) ** 2)
        else:
            ewc_loss = torch.tensor(0.0).to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, dir_pred  = model(imgs)
                loss_intent       = intent_criterion(logits, intent_labels)
                loss_dir          = dir_criterion(torch.tanh(dir_pred), dir_gt)
                loss              = loss_intent + lambda_dir * loss_dir + ewc_lambda * ewc_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, dir_pred = model(imgs)
            loss_intent      = intent_criterion(logits, intent_labels)
            loss_dir         = dir_criterion(torch.tanh(dir_pred), dir_gt)
            loss             = loss_intent + lambda_dir * loss_dir + ewc_lambda * ewc_loss
            loss.backward()
            optimizer.step()

        acc = (logits.argmax(dim=1) == intent_labels).float().mean().item()
        bs  = imgs.size(0)
        loss_m.update(loss.item(), bs)
        intent_m.update(loss_intent.item(), bs)
        dir_m.update(loss_dir.item(), bs)
        ewc_m.update(ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss, bs)
        acc_m.update(acc, bs)

    return {
        "loss":        loss_m.avg,
        "intent_loss": intent_m.avg,
        "dir_loss":    dir_m.avg,
        "ewc_loss":    ewc_m.avg,
        "accuracy":    acc_m.avg,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    intent_criterion: nn.Module,
    dir_criterion: nn.Module,
    lambda_dir: float,
    device: torch.device,
) -> dict:
    model.eval()
    loss_m   = AverageMeter()
    intent_m = AverageMeter()
    dir_m    = AverageMeter()
    acc_m    = AverageMeter()

    for imgs, intent_labels, dir_gt in loader:
        imgs          = imgs.to(device)
        intent_labels = intent_labels.to(device)
        dir_gt        = dir_gt.to(device)

        logits, dir_pred = model(imgs)
        loss_intent  = intent_criterion(logits, intent_labels)
        loss_dir     = dir_criterion(torch.tanh(dir_pred), dir_gt)
        loss         = loss_intent + lambda_dir * loss_dir

        acc = (logits.argmax(dim=1) == intent_labels).float().mean().item()
        bs  = imgs.size(0)
        loss_m.update(loss.item(), bs)
        intent_m.update(loss_intent.item(), bs)
        dir_m.update(loss_dir.item(), bs)
        acc_m.update(acc, bs)

    return {
        "loss":        loss_m.avg,
        "intent_loss": intent_m.avg,
        "dir_loss":    dir_m.avg,
        "accuracy":    acc_m.avg,
    }



def train(args: argparse.Namespace) -> None:
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    use_amp = device.type == "cuda" and args.amp
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed-precision training (AMP) enabled")

    # Dataset
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        logger.error("Dataset directory not found: %s", dataset_root)
        sys.exit(1)

    full_dataset = ROIDataset(dataset_root, transform=build_transforms(augment=True), max_samples=args.replay_buffer)
    if len(full_dataset) == 0:
        logger.error("No images found in dataset. Check label folders: %s", list(LABEL_MAP))
        sys.exit(1)

    val_size   = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Val dataset should use inference-time transforms (no augmentation)
    val_ds.dataset = ROIDataset(dataset_root, transform=build_transforms(augment=False))

    logger.info("Train: %d  Val: %d  (total: %d)", train_size, val_size, len(full_dataset))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    # Model
    model = build_model(freeze_backbone_blocks=args.freeze_blocks).to(device)

    # Resume from checkpoint
    start_epoch = 1
    best_val    = float("inf")
    model_anchor = None
    
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt  = torch.load(ckpt_path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val    = ckpt.get("best_val_loss", float("inf"))
                logger.info("Resumed from %s (epoch %d)", ckpt_path, start_epoch - 1)
            else:
                model.load_state_dict(ckpt, strict=False)
                logger.info("Loaded weights from %s", ckpt_path)
            
            # Create anchor model for EWC anti-forgetting
            if args.ewc_lambda > 0:
                import copy
                model_anchor = copy.deepcopy(model)
                model_anchor.eval()
                for p in model_anchor.parameters():
                    p.requires_grad = False
                logger.info("EWC: Anti-forgetting enabled (lambda=%g) anchored to %s", args.ewc_lambda, ckpt_path.name)
        else:
            logger.warning("Resume path not found: %s — starting fresh", ckpt_path)

    # Loss & Optimiser
    class_weights = compute_class_weights(full_dataset, device)
    intent_criterion = nn.CrossEntropyLoss(weight=class_weights)
    dir_criterion    = nn.MSELoss()
    lambda_dir       = args.lambda_dir

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )

    # Cosine annealing LR — smooth decay across all epochs
    total_epochs = args.epochs - start_epoch + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6,
    )

    # Output dir
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "training_log.csv"
    best_pt = out_dir / "intent_v1.pt"

    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc",
                         "val_loss", "val_acc", "train_intent_loss",
                         "train_dir_loss", "val_intent_loss", "val_dir_loss"])

    # Training loop
    logger.info("---")
    logger.info("  Training INTENT CNN")
    logger.info("  Epochs   : %d → %d", start_epoch, args.epochs)
    logger.info("  Batch    : %d  |  LR: %g  |  λ_dir: %g", args.batch_size, args.lr, lambda_dir)
    logger.info("  Output   : %s", out_dir)
    logger.info("---")

    for epoch in range(start_epoch, args.epochs + 1):
        t0  = time.monotonic()
        tr  = train_one_epoch(
            model, train_loader, optimizer, intent_criterion, dir_criterion, 
            lambda_dir, device, scaler, model_anchor, args.ewc_lambda
        )
        val = validate(model, val_loader, intent_criterion, dir_criterion,
                       lambda_dir, device)
        scheduler.step()
        elapsed = time.monotonic() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Ep %3d/%d  train[loss=%.4f acc=%.1f%%]  val[loss=%.4f acc=%.1f%%]  "
            "lr=%.2e  %.0fs",
            epoch, args.epochs,
            tr["loss"], tr["accuracy"] * 100,
            val["loss"], val["accuracy"] * 100,
            current_lr, elapsed,
        )

        # Log CSV
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, current_lr,
                tr["loss"], tr["accuracy"],
                val["loss"], val["accuracy"],
                tr["intent_loss"], tr["dir_loss"],
                val["intent_loss"], val["dir_loss"],
            ])

        # Save best model
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save({
                "epoch":           epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":        val["loss"],
                "val_accuracy":    val["accuracy"],
                "best_val_loss":   best_val,
                "label_map":       LABEL_MAP,
            }, best_pt)
            logger.info("   New best model saved → %s (val_loss=%.4f)", best_pt.name, best_val)

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = out_dir / f"checkpoint_ep{epoch:03d}.pt"
            torch.save({
                "epoch":           epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":        val["loss"],
                "best_val_loss":   best_val,
            }, ckpt_path)
            logger.info("  Checkpoint saved → %s", ckpt_path.name)

    logger.info("Training done. Best val_loss=%.4f → %s", best_val, best_pt)
    logger.info("Deploy: copy %s to Jetson at models/cnn_intent/intent_v1.pt", best_pt)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train INTENT CNN from ROI dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",      type=str, default="D:/nckh/context-aware/intent_dataset",
                        help="Path to labeled dataset root (must contain intent class folders)")
    parser.add_argument("--output",       type=str, default="models/cnn_intent",
                        help="Directory to save model checkpoints and logs")
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--lr",           type=float, default=3e-4,
                        help="Initial learning rate (Adam)")
    parser.add_argument("--lambda-dir",   type=float, default=0.5,
                        help="Weight for direction regression loss")
    parser.add_argument("--val-split",    type=float, default=0.15,
                        help="Fraction of data to use for validation")
    parser.add_argument("--workers",      type=int, default=4,
                        help="DataLoader worker threads (set 0 on Windows if issues)")
    parser.add_argument("--device",       type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp",          action="store_true", default=True,
                        help="Use Automatic Mixed Precision (CUDA only)")
    parser.add_argument("--freeze-blocks", type=int, default=10,
                        help="Number of backbone parameter tensors to freeze")
    parser.add_argument("--save-every",   type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume",       type=str, default=None,
                        help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--ewc-lambda",   type=float, default=5000.0,
                        help="EWC regularization strength to prevent forgetting")
    parser.add_argument("--replay-buffer",type=int, default=5000,
                        help="Max dataset size to randomly subsample to (Replay Buffer)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
