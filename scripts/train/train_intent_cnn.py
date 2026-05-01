"""Training script for Intent CNN.

Trains the MobileNetV3-Small + dual-head model defined in
src/perception/intent_cnn.py using the ROI dataset collected from the Jetson
and auto-labeled by scripts/autolabel.py.

Dataset structure expected:
    intent_dataset/
    ├── stationary/  roi_*.jpg  → STATIONARY (0), dx=0.0, dy=0.0
    ├── approaching/ roi_*.jpg  → APPROACHING (1), dx=0.0, dy=-0.6
    ├── departing/   roi_*.jpg  → DEPARTING (2), dx=0.0, dy=+0.6
    ├── crossing/    roi_*.jpg  → CROSSING (3), dx=±0.8, dy=0.0 (dx from cx shift)
    ├── erratic/     roi_*.jpg  → ERRATIC (4), dx=0.0, dy=0.0
    └── uncertain/   roi_*.jpg  → review/abstain only, excluded from training

Label mapping rationale
-----------------------
ROI autolabel uses depth and bbox spatial analysis to determine coarse
intent. We map it to the 5 trainable intent classes AND synthesise a plausible
(dx, dy) ground-truth vector so the direction head gets a meaningful
regression target.

`FOLLOWING` is intentionally not a class. Ambiguous residual motion is
`UNCERTAIN` and must be reviewed or excluded before phase-2 model export.

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
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.intent_labels import (  # noqa: E402
    REVIEW_ACCEPTED_STATUSES,
    TRAINABLE_INTENT_NAMES,
    TRAINABLE_LABEL_TO_ID,
    canonical_label,
    is_trainable_label,
    needs_human_review,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [train] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CNN_INPUT_W = 128
CNN_INPUT_H = 256
NUM_INTENT_CLASSES = len(TRAINABLE_INTENT_NAMES)

LABEL_MAP = {
    # folder_name → (intent_class, default_dx, default_dy)
    "STATIONARY": (TRAINABLE_LABEL_TO_ID["STATIONARY"], 0.0, 0.0),
    "APPROACHING": (TRAINABLE_LABEL_TO_ID["APPROACHING"], 0.0, -0.6),
    "DEPARTING": (TRAINABLE_LABEL_TO_ID["DEPARTING"], 0.0, 0.6),
    "CROSSING": (TRAINABLE_LABEL_TO_ID["CROSSING"], 0.0, 0.0),  # dx resolved dynamically
    "ERRATIC": (TRAINABLE_LABEL_TO_ID["ERRATIC"], 0.0, 0.0),
}

INTENT_NAMES = list(TRAINABLE_INTENT_NAMES)

Sample = tuple[list[Path], int, float, float, str]
KNOWN_NON_LABEL_DIRS = {"reports", "_extracted", "review_queue", ".git"}
METADATA_FILENAMES = ("metadata.jsonl", "imported_metadata.jsonl")


def _frame_id_from_name(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.startswith("f") and part[1:].isdigit():
            return int(part[1:])
    return 0


def _resolve_crossing_dx(frames: list[dict], idx: int) -> float:
    row = frames[idx]
    cx = row.get("cx")
    if cx is None:
        return 0.0
    lookahead = min(idx + 3, len(frames) - 1)
    if lookahead > idx and frames[lookahead].get("cx") is not None:
        future_cx = float(frames[lookahead]["cx"])
        return 0.8 if future_cx > float(cx) else -0.8
    if idx > 0 and frames[idx - 1].get("cx") is not None:
        prev_cx = float(frames[idx - 1]["cx"])
        return 0.8 if float(cx) > prev_cx else -0.8
    return 0.0


def _load_sample(
    samples: list[Sample],
    sample_idx: int,
    transform,
    hflip_p: float = 0.0,
):
    img_paths, intent_cls, dx, dy, _track_uid = samples[sample_idx]
    should_flip = hflip_p > 0.0 and random.random() < hflip_p

    frames = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        if should_flip:
            img = TF.hflip(img)
        if transform:
            img = transform(img)
        else:
            img = TF.to_tensor(img)
        frames.append(img)

    if should_flip:
        dx = -dx

    img = torch.stack(frames, dim=0)
    intent_label = torch.tensor(intent_cls, dtype=torch.long)
    direction_gt = torch.tensor([dx, dy], dtype=torch.float32)
    return img, intent_label, direction_gt


class ROIDataset(Dataset):
    """Reads labeled ROI sequences from the intent_dataset folder structure.

    Temporal API contract:
        each sample image tensor has shape `(T, C, H, W)`.
        Even when `temporal_window == 1`, the time dimension is preserved.
    """

    def __init__(
        self,
        root: Path,
        transform=None,
        max_samples: int | None = None,
        temporal_window: int = 15,
        require_reviewed_erratic: bool = True,
    ) -> None:
        self.transform = transform
        self.samples: list[Sample] = []
        self.temporal_window = max(1, int(temporal_window))

        # Load sidecar metadata once — keyed by resolved image path.
        # This is the primary source for track_uid, frame_id, and cx used to
        # build temporal sequences and resolve CROSSING direction.
        meta_index = self._load_metadata(root)

        def image_meta(img_path: Path) -> dict | None:
            return meta_index.get(str(img_path.resolve()))

        tracks: dict[str, list[dict]] = defaultdict(list)
        skipped_review = 0
        skipped_legacy = 0

        candidate_dirs = [
            p for p in root.iterdir() if p.is_dir() and p.name not in KNOWN_NON_LABEL_DIRS
        ]
        for folder_path in candidate_dirs:
            label = canonical_label(folder_path.name)
            if not is_trainable_label(label):
                skipped_legacy += len(list(folder_path.glob("*.jpg"))) + len(
                    list(folder_path.glob("*.png"))
                )
                continue
            intent_cls, base_dx, base_dy = LABEL_MAP[label]
            imgs = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))

            for img_path in imgs:
                meta = image_meta(img_path) or {}
                meta_label = canonical_label(meta.get("label", label))
                if meta_label != label or not is_trainable_label(meta_label):
                    skipped_legacy += 1
                    continue

                review_status = str(meta.get("review_status") or "")
                if (
                    require_reviewed_erratic
                    and needs_human_review(label)
                    and review_status not in REVIEW_ACCEPTED_STATUSES
                ):
                    skipped_review += 1
                    continue

                track_uid = str(
                    meta.get("track_uid")
                    or meta.get("session_id")
                    or meta.get("tid")
                    or img_path.stem
                )
                frame_id = int(meta.get("frame_id", _frame_id_from_name(img_path)))
                cx = float(meta["cx"]) if "cx" in meta and meta["cx"] is not None else None
                tracks[track_uid].append(
                    {
                        "path": img_path,
                        "label": label,
                        "intent_cls": intent_cls,
                        "base_dx": base_dx,
                        "base_dy": base_dy,
                        "frame_id": frame_id,
                        "cx": cx,
                    }
                )

        for track_uid, frames in tracks.items():
            frames.sort(key=lambda x: x["frame_id"])
            for i, row in enumerate(frames):
                history = frames[max(0, i - self.temporal_window + 1) : i + 1]
                seq_paths = [h["path"] for h in history]
                while len(seq_paths) < self.temporal_window:
                    seq_paths.insert(0, seq_paths[0])

                dx = float(row["base_dx"])
                if row["label"] == "CROSSING":
                    dx = _resolve_crossing_dx(frames, i)
                self.samples.append(
                    (
                        seq_paths[-self.temporal_window :],
                        int(row["intent_cls"]),
                        dx,
                        float(row["base_dy"]),
                        track_uid,
                    )
                )

        if max_samples and len(self.samples) > max_samples:
            # Replay Buffer: keep all HUMAN labels, randomly subsample AUTO labels
            human_samples = [s for s in self.samples if s[0][-1].name.startswith("human_")]
            auto_samples = [s for s in self.samples if not s[0][-1].name.startswith("human_")]

            auto_quota = max(0, max_samples - len(human_samples))
            if len(auto_samples) > auto_quota:
                random.shuffle(auto_samples)
                auto_samples = auto_samples[:auto_quota]

            self.samples = human_samples + auto_samples
            random.shuffle(self.samples)

        logger.info(
            "ROIDataset loaded: %d temporal samples from %s",
            len(self.samples),
            root,
        )
        if skipped_review:
            logger.warning(
                "Skipped %d ERRATIC/UNCERTAIN samples pending human review", skipped_review
            )
        if skipped_legacy:
            logger.warning("Skipped %d non-trainable/legacy samples", skipped_legacy)
        self._log_class_distribution()

    def _log_class_distribution(self) -> None:
        from collections import Counter

        counts = Counter(s[1] for s in self.samples)
        for cls_id, n in sorted(counts.items()):
            logger.info("  Class %d (%s): %d images", cls_id, INTENT_NAMES[cls_id], n)

    @staticmethod
    def _load_metadata(root: Path) -> dict[str, dict]:
        """Build {resolved_img_path_str -> metadata_dict} index from sidecar metadata.jsonl.

        Scans both the dataset root and every label sub-folder so the index
        covers images that have already been moved by autolabel.py.
        """
        index: dict[str, dict] = {}

        candidate_dirs = [root] + [
            d for d in root.iterdir() if d.is_dir() and d.name not in KNOWN_NON_LABEL_DIRS
        ]
        for directory in candidate_dirs:
            for metadata_name in METADATA_FILENAMES:
                meta_path = directory / metadata_name
                if not meta_path.exists():
                    continue
                with open(meta_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        file_value = row.get("file")
                        if not file_value:
                            continue
                        # Resolve against the directory that owns this JSONL file.
                        img_path = (directory / file_value).resolve()
                        index[str(img_path)] = row
        return index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return _load_sample(self.samples, idx, self.transform)


class _SplitView(Dataset):
    """View of a shared sample list with its own transform.

    Allows train and val to share the *same* subsampled sample list while
    applying different transforms — eliminating the index-mismatch that
    occurred when val_ds.dataset was replaced by a fresh ROIDataset.
    """

    def __init__(
        self,
        samples: list[Sample],
        indices: list[int],
        transform,
        hflip_p: float = 0.0,
    ) -> None:
        self.samples = samples
        self.indices = indices
        self.transform = transform
        self.hflip_p = hflip_p

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return _load_sample(self.samples, self.indices[idx], self.transform, self.hflip_p)


def _split_by_track(samples: list[Sample], val_split: float) -> tuple[list[int], list[int]]:
    by_track: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        by_track[sample[4]].append(idx)

    tracks = list(by_track.keys())
    random.shuffle(tracks)
    target_val = max(1, int(len(samples) * val_split))

    val_tracks: set[str] = set()
    val_count = 0
    for track in tracks:
        if len(tracks) - len(val_tracks) <= 1:
            break
        val_tracks.add(track)
        val_count += len(by_track[track])
        if val_count >= target_val:
            break

    val_indices = [idx for track in val_tracks for idx in by_track[track]]
    train_indices = [
        idx for track, indices in by_track.items() if track not in val_tracks for idx in indices
    ]
    if not val_indices and train_indices:
        fallback_n = max(1, int(len(train_indices) * val_split))
        val_indices = train_indices[:fallback_n]
        train_indices = train_indices[fallback_n:]
        logger.warning(
            "Track split produced empty val set; falling back to sample-level split for %d items",
            len(val_indices),
        )
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    return train_indices, val_indices


def build_transforms(augment: bool):
    """ImageNet-normalised transforms.  Training uses augmentation."""
    base = [
        transforms.Resize((CNN_INPUT_H, CNN_INPUT_W)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    if augment:
        aug = [
            # NOTE: HorizontalFlip is intentionally ABSENT here.
            # It is handled in __getitem__ / _SplitView so that dx can be
            # negated atomically with the image flip (avoids direction label corruption).
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

    backbone = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
    backbone.classifier = nn.Identity()

    model = _IntentModel(backbone, feature_dim=feature_dim)

    # Freeze first N feature blocks — fine-tune only upper layers + heads.
    # This prevents catastrophic forgetting of ImageNet features with small data.
    params = list(backbone.features.parameters())
    for i, param in enumerate(params):
        if i < freeze_backbone_blocks:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %d / %d parameters trainable (%.1f%%)", trainable, total, trainable / total * 100
    )
    return model


class _IntentModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.temporal = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.Conv1d(feature_dim, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.intent_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_INTENT_CLASSES),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(
                "Temporal API requires input shape (B, T, C, H, W); "
                f"got tensor with shape {tuple(x.shape)}"
            )
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feats = self.backbone.features(x)
        feats = self.pool(feats).flatten(1)
        feats = feats.reshape(b, t, -1).transpose(1, 2)
        temporal_feats = self.temporal(feats).mean(dim=-1)
        return self.intent_head(temporal_feats), self.direction_head(temporal_feats)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_class_weights(dataset, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights to handle label imbalance."""
    from collections import Counter

    counts = Counter(s[1] for s in dataset.samples)
    total = len(dataset)
    weights = torch.zeros(NUM_INTENT_CLASSES, dtype=torch.float32)
    for cls_id, n in counts.items():
        weights[cls_id] = total / (len(counts) * n)
    # Classes not in dataset get weight 1.0
    for cls_id in range(NUM_INTENT_CLASSES):
        if weights[cls_id] == 0:
            weights[cls_id] = 1.0
    return weights.to(device)


def _checkpoint_epoch(path: Path, device: torch.device) -> int:
    if not path.exists():
        return 0
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as exc:
        logger.warning("Could not inspect checkpoint epoch from %s: %s", path, exc)
        return 0
    return int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0


def _load_compatible_state_dict(model: nn.Module, state_dict: dict) -> None:
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    skipped = sorted(set(state_dict.keys()) - set(compatible.keys()))
    model.load_state_dict(compatible, strict=False)
    if skipped:
        logger.warning(
            "Skipped %d incompatible checkpoint tensors; old intent heads will be retrained",
            len(skipped),
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    intent_criterion: nn.Module,
    dir_criterion: nn.Module,
    lambda_dir: float,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    model_anchor: nn.Module | None = None,
    ewc_lambda: float = 0.0,
) -> dict:
    model.train()
    loss_m = AverageMeter()
    intent_m = AverageMeter()
    dir_m = AverageMeter()
    ewc_m = AverageMeter()
    acc_m = AverageMeter()

    anchor_params = None
    if model_anchor is not None and ewc_lambda > 0:
        anchor_params = list(model_anchor.parameters())

    for imgs, intent_labels, dir_gt in loader:
        imgs = imgs.to(device)
        intent_labels = intent_labels.to(device)
        dir_gt = dir_gt.to(device)

        optimizer.zero_grad()

        if anchor_params is not None:
            # Simplified EWC: L2 distance to old weights (Starting Point regularization)
            # F_i is approximated as 1.0 here for stability on small datasets
            ewc_loss = torch.tensor(0.0, device=device)
            for p, p_old in zip(model.parameters(), anchor_params):
                if p.requires_grad:
                    ewc_loss += torch.sum((p - p_old) ** 2)
        else:
            ewc_loss = torch.tensor(0.0, device=device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, dir_pred = model(imgs)
                loss_intent = intent_criterion(logits, intent_labels)
                loss_dir = dir_criterion(torch.tanh(dir_pred), dir_gt)
                loss = loss_intent + lambda_dir * loss_dir + ewc_lambda * ewc_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, dir_pred = model(imgs)
            loss_intent = intent_criterion(logits, intent_labels)
            loss_dir = dir_criterion(torch.tanh(dir_pred), dir_gt)
            loss = loss_intent + lambda_dir * loss_dir + ewc_lambda * ewc_loss
            loss.backward()
            optimizer.step()

        acc = (logits.argmax(dim=1) == intent_labels).float().mean().item()
        bs = imgs.size(0)
        loss_m.update(loss.item(), bs)
        intent_m.update(loss_intent.item(), bs)
        dir_m.update(loss_dir.item(), bs)
        ewc_m.update(ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss, bs)
        acc_m.update(acc, bs)

    return {
        "loss": loss_m.avg,
        "intent_loss": intent_m.avg,
        "dir_loss": dir_m.avg,
        "ewc_loss": ewc_m.avg,
        "accuracy": acc_m.avg,
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
    loss_m = AverageMeter()
    intent_m = AverageMeter()
    dir_m = AverageMeter()
    acc_m = AverageMeter()

    for imgs, intent_labels, dir_gt in loader:
        imgs = imgs.to(device)
        intent_labels = intent_labels.to(device)
        dir_gt = dir_gt.to(device)

        logits, dir_pred = model(imgs)
        loss_intent = intent_criterion(logits, intent_labels)
        loss_dir = dir_criterion(torch.tanh(dir_pred), dir_gt)
        loss = loss_intent + lambda_dir * loss_dir

        acc = (logits.argmax(dim=1) == intent_labels).float().mean().item()
        bs = imgs.size(0)
        loss_m.update(loss.item(), bs)
        intent_m.update(loss_intent.item(), bs)
        dir_m.update(loss_dir.item(), bs)
        acc_m.update(acc, bs)

    return {
        "loss": loss_m.avg,
        "intent_loss": intent_m.avg,
        "dir_loss": dir_m.avg,
        "accuracy": acc_m.avg,
    }


@torch.no_grad()
def collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all = []
    labels_all = []
    for imgs, intent_labels, _dir_gt in loader:
        imgs = imgs.to(device)
        logits, _ = model(imgs)
        logits_all.append(logits.float().cpu())
        labels_all.append(intent_labels.long().cpu())
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def fit_temperature(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Fit a scalar temperature on validation logits for confidence calibration."""
    logits, labels = collect_logits_and_labels(model, loader, device)
    if logits.numel() == 0:
        return 1.0

    temperature = torch.ones(1, requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature.clamp_min(1e-3), labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        return 1.0

    return float(temperature.detach().clamp(0.5, 10.0).item())


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> float:
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=1)
    confs, preds = probs.max(dim=1)
    correct = preds.eq(labels)
    ece = torch.zeros(1)
    for lower in torch.linspace(0, 0.9, 10):
        upper = lower + 0.1
        mask = (confs > lower) & (confs <= upper)
        if mask.any():
            acc = correct[mask].float().mean()
            conf = confs[mask].mean()
            ece += mask.float().mean() * torch.abs(conf - acc)
    return float(ece.item())


def train(args: argparse.Namespace) -> None:
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    use_amp = device.type == "cuda" and args.amp
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed-precision training (AMP) enabled")

    resume_epoch = _checkpoint_epoch(Path(args.resume), device) if args.resume else 0
    if args.resume and getattr(args, "epochs_are_additional", False) and resume_epoch > 0:
        args.epochs += resume_epoch
        logger.info(
            "Resume mode: training %d additional epochs (target epoch %d)",
            args.epochs - resume_epoch,
            args.epochs,
        )

    # Dataset — load sample list ONCE (replay buffer applied), then split by index.
    # _SplitView ensures val and train point into the *same* subsampled list so
    # indices are never stale (fixes the bug where val_ds.dataset was replaced
    # by a fresh full-size ROIDataset after random_split).
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        logger.error("Dataset directory not found: %s", dataset_root)
        sys.exit(1)

    index_ds = ROIDataset(
        dataset_root,
        transform=None,
        max_samples=args.replay_buffer,
        temporal_window=args.temporal_window,
        require_reviewed_erratic=not args.allow_unreviewed_erratic,
    )
    if len(index_ds) == 0:
        logger.error("No images found in dataset. Check label folders: %s", list(LABEL_MAP))
        sys.exit(1)
    if len(index_ds) < 2:
        logger.error("Need at least 2 trainable images for a train/val split; found %d", len(index_ds))
        sys.exit(1)

    train_indices, val_indices = _split_by_track(index_ds.samples, args.val_split)
    if not train_indices or not val_indices:
        logger.error("Track-level split failed. Need at least 2 distinct tracks.")
        sys.exit(1)

    train_ds = _SplitView(
        index_ds.samples, train_indices,
        transform=build_transforms(augment=True), hflip_p=0.5,
    )
    val_ds = _SplitView(
        index_ds.samples, val_indices,
        transform=build_transforms(augment=False), hflip_p=0.0,
    )

    logger.info(
        "Train: %d  Val: %d  (total: %d)",
        len(train_ds), len(val_ds), len(index_ds),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = build_model(freeze_backbone_blocks=args.freeze_blocks).to(device)

    # Resume from checkpoint
    start_epoch = 1
    best_val = float("inf")
    model_anchor = None

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            if "model_state_dict" in ckpt:
                _load_compatible_state_dict(model, ckpt["model_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val = ckpt.get("best_val_loss", float("inf"))
                logger.info("Resumed from %s (epoch %d)", ckpt_path, start_epoch - 1)
            else:
                _load_compatible_state_dict(model, ckpt)
                logger.info("Loaded weights from %s", ckpt_path)

            # Create anchor model for EWC anti-forgetting
            if args.ewc_lambda > 0:
                import copy

                model_anchor = copy.deepcopy(model)
                model_anchor.eval()
                for p in model_anchor.parameters():
                    p.requires_grad = False
                logger.info(
                    "EWC: Anti-forgetting enabled (lambda=%g) anchored to %s",
                    args.ewc_lambda,
                    ckpt_path.name,
                )
        else:
            logger.warning("Resume path not found: %s — starting fresh", ckpt_path)

    # Loss & Optimiser — class weights derived from the same subsampled index_ds
    class_weights = compute_class_weights(index_ds, device)
    intent_criterion = nn.CrossEntropyLoss(weight=class_weights)
    dir_criterion = nn.MSELoss()
    lambda_dir = args.lambda_dir

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Cosine annealing LR — smooth decay across all epochs
    total_epochs = args.epochs - start_epoch + 1
    if total_epochs <= 0:
        logger.error(
            "No epochs left to train: start_epoch=%d, target_epoch=%d. "
            "Increase --epochs or pass --epochs-are-additional when resuming.",
            start_epoch,
            args.epochs,
        )
        sys.exit(1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=1e-6,
    )

    # Output dir
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "training_log.csv"
    best_pt = out_dir / "intent_v1.pt"

    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "train_intent_loss",
                "train_dir_loss",
                "val_intent_loss",
                "val_dir_loss",
            ]
        )

    # Training loop
    logger.info("---")
    logger.info("  Training INTENT CNN")
    logger.info("  Epochs   : %d → %d", start_epoch, args.epochs)
    logger.info("  Batch    : %d  |  LR: %g  |  λ_dir: %g", args.batch_size, args.lr, lambda_dir)
    logger.info("  Output   : %s", out_dir)
    logger.info("---")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.monotonic()
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            intent_criterion,
            dir_criterion,
            lambda_dir,
            device,
            scaler,
            model_anchor,
            args.ewc_lambda,
        )
        val = validate(model, val_loader, intent_criterion, dir_criterion, lambda_dir, device)
        scheduler.step()
        elapsed = time.monotonic() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Ep %3d/%d  train[loss=%.4f acc=%.1f%%]  val[loss=%.4f acc=%.1f%%]  lr=%.2e  %.0fs",
            epoch,
            args.epochs,
            tr["loss"],
            tr["accuracy"] * 100,
            val["loss"],
            val["accuracy"] * 100,
            current_lr,
            elapsed,
        )

        # Log CSV
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    current_lr,
                    tr["loss"],
                    tr["accuracy"],
                    val["loss"],
                    val["accuracy"],
                    tr["intent_loss"],
                    tr["dir_loss"],
                    val["intent_loss"],
                    val["dir_loss"],
                ]
            )

        # Save best model
        if val["loss"] < best_val:
            best_val = val["loss"]
            temperature = fit_temperature(model, val_loader, device)
            val_logits, val_labels = collect_logits_and_labels(model, val_loader, device)
            ece = expected_calibration_error(val_logits, val_labels, temperature)
            metadata = {
                "architecture": "mobilenetv3_small_tcn",
                "temporal_window": args.temporal_window,
                "trainable_intent_names": INTENT_NAMES,
                "runtime_intent_names": INTENT_NAMES + ["UNCERTAIN"],
                "num_trainable_classes": NUM_INTENT_CLASSES,
                "temperature": temperature,
                "ece": ece,
                "confidence_threshold": args.confidence_threshold,
                "margin_threshold": args.margin_threshold,
                "label_policy": "FOLLOWING removed; UNCERTAIN/ERRATIC require human review",
                "continual_learning": {
                    "ewc_lambda": args.ewc_lambda,
                    "replay_buffer": args.replay_buffer,
                },
                "optimization_targets": {
                    "distillation": args.distill_from or "",
                    "quantization": "export/benchmark scripts; checkpoint remains PyTorch",
                },
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val["loss"],
                    "val_accuracy": val["accuracy"],
                    "best_val_loss": best_val,
                    "label_map": LABEL_MAP,
                    "metadata": metadata,
                    "temperature": temperature,
                },
                best_pt,
            )
            logger.info(
                "   New best model saved -> %s (val_loss=%.4f, T=%.3f, ECE=%.4f)",
                best_pt.name,
                best_val,
                temperature,
                ece,
            )

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = out_dir / f"checkpoint_ep{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val["loss"],
                    "best_val_loss": best_val,
                },
                ckpt_path,
            )
            logger.info("  Checkpoint saved → %s", ckpt_path.name)

    logger.info("Training done. Best val_loss=%.4f → %s", best_val, best_pt)
    logger.info("Deploy: copy %s to Jetson at models/cnn_intent/intent_v1.pt", best_pt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train INTENT CNN from ROI dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="D:/nckh/context-aware/intent_dataset",
        help="Path to labeled dataset root (must contain intent class folders)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/cnn_intent",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (Adam)")
    parser.add_argument(
        "--lambda-dir", type=float, default=0.5, help="Weight for direction regression loss"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15, help="Fraction of data to use for validation"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker threads (set 0 on Windows if issues)",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--amp", action="store_true", default=True, help="Use Automatic Mixed Precision (CUDA only)"
    )
    parser.add_argument(
        "--freeze-blocks",
        type=int,
        default=10,
        help="Number of backbone parameter tensors to freeze",
    )
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from"
    )
    parser.add_argument(
        "--epochs-are-additional",
        action="store_true",
        help="When resuming, interpret --epochs as extra epochs instead of final target epoch",
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=5000.0,
        help="EWC regularization strength to prevent forgetting",
    )
    parser.add_argument(
        "--replay-buffer",
        type=int,
        default=5000,
        help="Max dataset size to randomly subsample to (Replay Buffer)",
    )
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=15,
        help="Number of ROI frames per track sample for temporal CNN/TCN input",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
        help="Runtime confidence threshold below which prediction abstains as UNCERTAIN",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.12,
        help="Runtime top-1/top-2 probability margin threshold for UNCERTAIN abstention",
    )
    parser.add_argument(
        "--allow-unreviewed-erratic",
        action="store_true",
        help="Allow ERRATIC samples still marked needs_review into training",
    )
    parser.add_argument(
        "--distill-from",
        type=str,
        default=None,
        help="Optional teacher checkpoint path reserved for distillation experiments",
    )
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
