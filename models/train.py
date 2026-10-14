"""
Kaggle-style training for VisionCare (CPU-only).

Matches project spec:
  - Epochs: 30, Batch: 8, Optimizer: AdamW, LR: 0.0001
  - Albumentations augmentation on train split only
  - Hold-out test split for validation (no random leakage)
  - Saves best checkpoint by validation accuracy

Usage:
  python scripts/prepare_kaggle_split.py
  python models/train.py --data datasets/kermany --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Allow CUDA if available; do not override CUDA visibility here.

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch.nn.functional as F

from models.architectures import EfficientNetClassifier, MobileNetClassifier
from utils.config import CLASS_LABELS, MODELS_DIR
from utils.metrics import compute_metrics

IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def load_samples_from_split(root: Path, split: str, max_per_class: int | None = None) -> list[tuple[Path, int]]:
    """Load images from root/split/CNV style (Kaggle layout)."""
    samples: list[tuple[Path, int]] = []
    base = root / split
    if not base.exists():
        base = root

    for idx, cls in enumerate(CLASS_LABELS):
        paths: list[Path] = []
        for folder_name in (cls, cls.lower()):
            folder = base / folder_name
            if folder.is_dir():
                for p in folder.iterdir():
                    if p.is_file() and p.exists() and p.suffix.lower() in IMAGE_EXTS:
                        paths.append(p)
            if paths:
                break
        paths = list(dict.fromkeys(paths))
        if not paths and split == "":
            for folder_name in (cls, cls.lower()):
                folder = root / folder_name
                if folder.is_dir():
                    paths.extend(
                        p for p in folder.rglob("*")
                        if p.is_file() and p.exists() and p.suffix.lower() in IMAGE_EXTS
                    )
        random.shuffle(paths)
        if max_per_class:
            paths = paths[:max_per_class]
        for p in paths:
            samples.append((p, idx))
    return samples


class AlbumentationsOCTDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], augment: bool = False):
        self.samples = samples
        if augment:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = np.array(Image.open(path).convert("RGB"))
        return self.transform(image=img)["image"], label


def mixup_data(x, y, alpha=0.2, device=None):
    """Returns mixed inputs, and mixed label vectors, plus lambda."""
    if alpha <= 0:
        # return one-hot labels
        y_onehot = F.one_hot(y, num_classes=len(y.unique())) if y.dim() == 1 else y
        return x, y_onehot.float(), 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    if device is None:
        device = x.device
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a = F.one_hot(y, num_classes=len(CLASS_LABELS)).float().to(device)
    y_b = F.one_hot(y[index], num_classes=len(CLASS_LABELS)).float().to(device)
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y, lam


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> tuple[float, dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds, truths, probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item()
        prob = torch.softmax(out, dim=1)
        pred = out.argmax(1)
        preds.extend(pred.cpu().numpy())
        truths.extend(y.cpu().numpy())
        probs.extend(prob.cpu().numpy())
    y_true = np.array(truths)
    y_pred = np.array(preds)
    y_prob = np.array(probs)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    acc = 100.0 * (y_pred == y_true).mean() if len(y_true) else 0.0
    avg_loss = total_loss / max(len(loader), 1)
    return acc, {"loss": avg_loss, **metrics}, y_true, y_pred, y_prob


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    name: str,
    patience: int = 7,
    class_weights: torch.Tensor | None = None,
    mixup_alpha: float = 0.0,
) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_metrics: dict = {}
    stale = 0
    history = []
    weights_path = MODELS_DIR / f"{name}.pth"

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if mixup_alpha and mixup_alpha > 0.0:
                x_m, y_mixed, lam = mixup_data(x, y, alpha=mixup_alpha, device=device)
                out = model(x_m)
                logp = F.log_softmax(out, dim=1)
                loss_per_sample = - (y_mixed * logp).sum(dim=1)
                if class_weights is not None:
                    cw = class_weights.to(device)
                    sample_weights = (y_mixed * cw).sum(dim=1)
                    loss = (loss_per_sample * sample_weights).mean()
                else:
                    loss = loss_per_sample.mean()
                preds = out.argmax(1)
                train_correct += (preds == y).sum().item()
                train_n += len(y)
            else:
                out = model(x)
                loss = criterion(out, y)
                train_loss += loss.item()
                train_correct += (out.argmax(1) == y).sum().item()
                train_n += len(y)

            # Backpropagate and update (single place to avoid double steps)
            loss.backward()
            opt.step()

        val_acc, val_metrics, _, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        train_acc = 100 * train_correct / max(train_n, 1)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "val_f1": round(val_metrics.get("f1", 0), 4),
            "val_precision": round(val_metrics.get("precision", 0), 4),
            "val_recall": round(val_metrics.get("recall", 0), 4),
            "val_roc_auc": round(val_metrics.get("roc_auc", 0), 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"[{name}] Epoch {epoch+1}/{epochs} "
            f"loss={row['train_loss']:.4f} train={train_acc:.1f}% val={val_acc:.1f}% "
            f"f1={row['val_f1']:.3f} auc={row['val_roc_auc']:.3f} ({row['seconds']}s)"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = val_metrics
            stale = 0
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), weights_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"[{name}] Early stopping at epoch {epoch+1} (patience={patience})")
                break

    return {
        "best_val_acc": best_acc,
        "best_metrics": best_metrics,
        "history": history,
        "weights": str(weights_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/kermany")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None, help="Cap train images per class")
    parser.add_argument("--test-max-per-class", type=int, default=250, help="Cap test images per class")
    parser.add_argument(
        "--only",
        type=str,
        default="both",
        choices=("both", "efficientnet", "mobilenet"),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose .pth already exists",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        default=True,
        help="Balance classes to min count per split (default: on)",
    )
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="MixUp alpha (0 disables)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prefer GPU when available for speed and improved accuracy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

    data_root = ROOT / args.data
    train_samples = load_samples_from_split(data_root, "train", args.max_per_class)
    test_samples = load_samples_from_split(data_root, "test", args.test_max_per_class)

    if not test_samples:
        print("No test/ split found — using 15% stratified hold-out from train (run prepare_kaggle_split.py for Kaggle layout).")
        all_samples = train_samples or load_samples_from_split(data_root, "", args.max_per_class)
        if not all_samples:
            print("No images. Run: python scripts/prepare_kaggle_split.py")
            return
        by_class: dict[int, list] = {i: [] for i in range(len(CLASS_LABELS))}
        for p, l in all_samples:
            by_class[l].append((p, l))
        train_samples, test_samples = [], []
        for cls_idx, items in by_class.items():
            random.shuffle(items)
            n_val = max(1, int(len(items) * 0.15))
            test_samples.extend(items[:n_val])
            train_samples.extend(items[n_val:])

    if args.balance:
        by_class: dict[int, list] = {i: [] for i in range(len(CLASS_LABELS))}
        for p, l in train_samples:
            by_class[l].append((p, l))
        min_train = min(len(v) for v in by_class.values()) if train_samples else 0
        train_samples = []
        for cls_idx, items in by_class.items():
            random.shuffle(items)
            train_samples.extend(items[:min_train])

        by_class_t: dict[int, list] = {i: [] for i in range(len(CLASS_LABELS))}
        for p, l in test_samples:
            by_class_t[l].append((p, l))
        min_test = min(len(v) for v in by_class_t.values()) if test_samples else 0
        test_samples = []
        for cls_idx, items in by_class_t.items():
            random.shuffle(items)
            test_samples.extend(items[:min_test])

    counts_train = {CLASS_LABELS[i]: sum(1 for _, l in train_samples if l == i) for i in range(len(CLASS_LABELS))}
    counts_test = {CLASS_LABELS[i]: sum(1 for _, l in test_samples if l == i) for i in range(len(CLASS_LABELS))}
    print(f"Kaggle-style training (device={device})")
    print(f"  Train: {len(train_samples)} {counts_train}")
    print(f"  Test:  {len(test_samples)} {counts_test}")

    # Compute normalized inverse-frequency class weights for loss
    train_counts_list = [counts_train.get(c, 0) for c in CLASS_LABELS]
    class_weights = None
    if any(train_counts_list):
        inv = [1.0 / c if c > 0 else 0.0 for c in train_counts_list]
        s = sum(inv) or 1.0
        norm = [v / s for v in inv]
        class_weights = torch.tensor(norm, dtype=torch.float)

    train_ds = AlbumentationsOCTDataset(train_samples, augment=True)
    val_ds = AlbumentationsOCTDataset(test_samples, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    eff_path = MODELS_DIR / "efficientnet_b0_oct.pth"
    mob_path = MODELS_DIR / "mobilenet_v3_oct.pth"
    eff_results = {"skipped": True, "best_val_acc": 0, "weights": str(eff_path)}
    mob_results = {"skipped": True, "best_val_acc": 0, "weights": str(mob_path)}

    train_eff = args.only in ("both", "efficientnet")
    train_mob = args.only in ("both", "mobilenet")
    if args.skip_existing and eff_path.exists():
        train_eff = False
        print(f"Skipping EfficientNet — exists: {eff_path}")
    if args.skip_existing and mob_path.exists():
        train_mob = False
        print(f"Skipping MobileNet — exists: {mob_path}")

    if train_eff:
        eff = EfficientNetClassifier(len(CLASS_LABELS), pretrained=True).to(device)
        print("\n=== EfficientNet-B0 (primary, 70% ensemble weight) ===")
        eff_results = train_classifier(
            eff, train_loader, val_loader, device, args.epochs, args.lr,
            "efficientnet_b0_oct", args.patience, class_weights=class_weights,
            mixup_alpha=args.mixup_alpha,
        )

    if train_mob:
        mob = MobileNetClassifier(len(CLASS_LABELS), pretrained=True).to(device)
        print("\n=== MobileNetV3-Small (validation model, 30% ensemble) ===")
        mob_results = train_classifier(
            mob, train_loader, val_loader, device, args.epochs, args.lr,
            "mobilenet_v3_oct", args.patience, class_weights=class_weights,
            mixup_alpha=args.mixup_alpha,
        )

    summary = {
        "layout": "kaggle_train_test",
        "train_counts": counts_train,
        "test_counts": counts_test,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "loss": "CrossEntropyLoss",
            "augmentation": "flip, rotate, brightness, noise, CLAHE",
        },
        "efficientnet_b0": eff_results,
        "mobilenet_v3": mob_results,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = MODELS_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== Training complete ===")
    print(f"  EfficientNet best val acc: {eff_results['best_val_acc']:.1f}%")
    print(f"  MobileNet best val acc:    {mob_results['best_val_acc']:.1f}%")
    print(f"  Weights: {MODELS_DIR}")
    print(f"  Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
