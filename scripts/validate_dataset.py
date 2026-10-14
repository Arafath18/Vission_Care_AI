"""Verify Kermany dataset layout before training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.train import load_samples_from_split
from utils.config import CLASS_LABELS

DATA = ROOT / "datasets" / "kermany"
MIN_PER_CLASS = 50


def main() -> int:
    if not DATA.exists():
        print(f"Missing dataset folder: {DATA}")
        print("Run: python scripts/download_datasets.py --streaming --max-per-class 500")
        return 1

    train = load_samples_from_split(DATA, "train", max_per_class=None)
    test = load_samples_from_split(DATA, "test", max_per_class=None)

    if not train:
        train = load_samples_from_split(DATA, "", max_per_class=None)

    if not train:
        print("No training images found under datasets/kermany/train or class folders.")
        return 1

    missing_classes = []
    for cls in CLASS_LABELS:
        n_train = sum(1 for _, l in train if CLASS_LABELS[l] == cls)
        n_test = sum(1 for _, l in test if CLASS_LABELS[l] == cls) if test else 0
        print(f"  {cls}: train={n_train} test={n_test}")
        if n_train < MIN_PER_CLASS:
            print(f"Warning: {cls} has fewer than {MIN_PER_CLASS} train images.")
        if n_train == 0 and n_test == 0:
            missing_classes.append(cls)
    if missing_classes:
        print("Warning: the following classes have no images:", ", ".join(missing_classes))
        print("Please populate datasets/kermany with these class folders for 11-disease analysis.")

    print(f"OK — {len(train)} train, {len(test)} test images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
