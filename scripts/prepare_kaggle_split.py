"""
Prepare Kaggle-style train/test folder layout from Hugging Face splits.

Output:
  datasets/kermany/train/{CNV,DME,DRUSEN,NORMAL}/
  datasets/kermany/test/{CNV,DME,DRUSEN,NORMAL}/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from utils.config import CLASS_LABELS

DEST = ROOT / "datasets" / "kermany"
CLASSES = CLASS_LABELS
HF_REPO = "zacharielegault/Kermany2017-OCT"


def export_split(split_name: str, max_per_class: int, dest_root: Path, preserve_existing: bool = True) -> dict[str, int]:
    from datasets import load_dataset

    counts = {c: 0 for c in CLASSES}
    out_base = dest_root / split_name
    for c in CLASSES:
        folder = out_base / c
        if folder.exists():
            if preserve_existing:
                counts[c] = sum(
                    1 for f in folder.iterdir()
                    if f.is_file() and f.suffix.lower() in {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
                )
            else:
                shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(HF_REPO, split=split_name, streaming=True)
    for row in ds:
        label = CLASSES[int(row["label"])]
        if counts[label] >= max_per_class:
            if all(counts[x] >= max_per_class for x in CLASSES):
                break
            continue
        idx = counts[label]
        path = out_base / label / f"{split_name}_{label}_{idx:05d}.jpeg"
        row["image"].convert("RGB").save(path, "JPEG", quality=92)
        counts[label] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-class", type=int, default=800)
    parser.add_argument("--test-per-class", type=int, default=250)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Reset existing train/test outputs before exporting. By default, existing images are preserved.",
    )
    args = parser.parse_args()
    preserve_existing = not args.clear

    print("Exporting Kaggle-style train split...")
    train_counts = export_split("train", args.train_per_class, DEST, preserve_existing=preserve_existing)
    print("Train:", train_counts)

    print("Exporting Kaggle-style test split (hold-out validation)...")
    test_counts = export_split("test", args.test_per_class, DEST, preserve_existing=preserve_existing)
    print("Test:", test_counts)

    print(f"\nReady: {DEST / 'train'} and {DEST / 'test'}")


if __name__ == "__main__":
    main()
