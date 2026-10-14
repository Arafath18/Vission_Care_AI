"""Train Attention U-Net for lesion segmentation (CPU, fast pseudo-mask run)."""

import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.architectures import AttentionUNet
from utils.config import CLASS_LABELS, MODELS_DIR

IMAGE_EXTS = {".jpeg", ".jpg", ".png"}


class SegDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], augment: bool = False):
        self.samples = samples
        aug = [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
        val = [A.Resize(224, 224), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
        self.t = A.Compose(aug if augment else val)

    def __len__(self):
        return len(self.samples)

    def _pseudo_mask(self, gray: np.ndarray, label: int) -> np.ndarray:
        if label == CLASS_LABELS.index("NORMAL"):
            return np.zeros_like(gray, dtype=np.float32)
        _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask.astype(np.float32)
        if label == CLASS_LABELS.index("DRUSEN"):
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return np.clip(mask, 0, 1)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = np.array(Image.open(path).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = self._pseudo_mask(gray, label)
        out = self.t(image=img, mask=mask)
        return out["image"], out["mask"].unsqueeze(0).float()


def collect_samples(root: Path, max_per_class: int = 200) -> list:
    samples = []
    for split in ("train", ""):
        base = root / split if (root / split).exists() else root
        for idx, cls in enumerate(CLASS_LABELS):
            folder = base / cls
            if not folder.exists():
                continue
            files = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS]
            random.shuffle(files)
            for f in files[:max_per_class]:
                samples.append((f, idx))
    return samples


def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum()
    return 1 - (2 * inter + 1) / (pred.sum() + target.sum() + 1)


def main():
    random.seed(42)
    data = ROOT / "datasets" / "kermany"
    samples = collect_samples(data, 150)
    if len(samples) < 50:
        print("Not enough images for segmentation training.")
        return

    n_val = max(20, len(samples) // 10)
    random.shuffle(samples)
    train_s, val_s = samples[n_val:], samples[:n_val]

    train_loader = DataLoader(SegDataset(train_s, True), batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(SegDataset(val_s, False), batch_size=8, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = AttentionUNet(3, 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best = 1e9
    for epoch in range(8):
        model.train()
        loss_sum = 0.0
        for x, m in train_loader:
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            out = model(x)
            loss = bce(out, m) + dice_loss(out, m)
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, m in val_loader:
                x, m = x.to(device), m.to(device)
                out = model(x)
                val_loss += (bce(out, m) + dice_loss(out, m)).item()

        val_loss /= max(len(val_loader), 1)
        print(f"Epoch {epoch+1}/8 train={loss_sum/len(train_loader):.4f} val={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / "attention_unet_seg.pth")

    print(f"Saved {MODELS_DIR / 'attention_unet_seg.pth'}")


if __name__ == "__main__":
    main()
