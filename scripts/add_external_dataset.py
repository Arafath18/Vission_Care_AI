"""Copy externally labeled retinal OCT images into the project dataset without removing existing data."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from utils.config import CLASS_LABELS

DEST = ROOT / "datasets" / "kermany"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

DEFAULT_MAPPING = {
    "CNV": "CNV",
    "DME": "DME",
    "DRUSEN": "DRUSEN",
    "NORMAL": "NORMAL",
    "AMD": "AMD",
    "AGE RELATED MACULAR DEGENERATION": "AMD",
    "AGE-RELATED MACULAR DEGENERATION": "AMD",
    "CSCR": "CSCR",
    "CENTRAL SEROUS CHORIORETINOPATHY": "CSCR",
    "ERM": "ERM",
    "EPIRRETINAL MEMBRANE": "ERM",
    "MH": "MH",
    "MACULAR HOLE": "MH",
    "RD": "RD",
    "RETINAL DETACHMENT": "RD",
    "GLAUCOMA": "GLAUCOMA",
    "RP": "RP",
    "RETINITIS PIGMENTOSA": "RP",
}


def normalize_name(name: str) -> str:
    return name.strip().upper().replace("-", " ").replace("_", " ")


def load_mapping(mapping_path: Path | None) -> dict[str, str]:
    mapping = DEFAULT_MAPPING.copy()
    if mapping_path is None:
        return mapping
    with mapping_path.open("r", encoding="utf-8") as handle:
        custom = json.load(handle)
    for key, value in custom.items():
        mapping[normalize_name(str(key))] = str(value).upper()
    return mapping


def resolve_label(folder_name: str, mapping: dict[str, str]) -> str | None:
    normalized = normalize_name(folder_name)
    if normalized in mapping:
        return mapping[normalized]
    if normalized in CLASS_LABELS:
        return normalized
    return None


def find_image_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def split_image_paths(paths: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if train_ratio <= 0:
        return [], list(paths)
    if train_ratio >= 1:
        return list(paths), []
    rng = random.Random(seed)
    paths_copy = list(paths)
    rng.shuffle(paths_copy)
    n_train = int(len(paths_copy) * train_ratio)
    return paths_copy[:n_train], paths_copy[n_train:]


def copy_images(source_dir: Path, mapping: dict[str, str], dest_root: Path, dry_run: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {cls: 0 for cls in CLASS_LABELS}
    for folder in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
        label = resolve_label(folder.name, mapping)
        if label is None:
            print(f"Skipping unknown source class folder: {folder.name}")
            continue
        if label not in CLASS_LABELS:
            print(f"Skipping unsupported label: {label}")
            continue
        target_dir = dest_root / label
        target_dir.mkdir(parents=True, exist_ok=True)
        for src_path in sorted(folder.rglob("*")):
            if not src_path.is_file() or src_path.suffix.lower() not in IMAGE_EXTS:
                continue
            dest_path = target_dir / src_path.name
            if dest_path.exists():
                dest_path = target_dir / f"{src_path.stem}_{counts[label]:04d}{src_path.suffix.lower()}"
            if dry_run:
                print(f"DRY RUN: {src_path} -> {dest_path}")
            else:
                shutil.copy2(src_path, dest_path)
            counts[label] += 1
    return counts


def copy_split_images(source_dir: Path, mapping: dict[str, str], dest_root: Path, train_ratio: float, seed: int, dry_run: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {cls: 0 for cls in CLASS_LABELS}
    for folder in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
        label = resolve_label(folder.name, mapping)
        if label is None:
            print(f"Skipping unknown source class folder: {folder.name}")
            continue
        if label not in CLASS_LABELS:
            print(f"Skipping unsupported label: {label}")
            continue
        images = find_image_files(folder)
        if not images:
            continue
        train_paths, test_paths = split_image_paths(images, train_ratio, seed)
        for split_name, paths in ("train", train_paths), ("test", test_paths):
            target_dir = dest_root / split_name / label
            target_dir.mkdir(parents=True, exist_ok=True)
            for src_path in paths:
                dest_path = target_dir / src_path.name
                if dest_path.exists():
                    dest_path = target_dir / f"{src_path.stem}_{counts[label]:04d}{src_path.suffix.lower()}"
                if dry_run:
                    print(f"DRY RUN: {src_path} -> {dest_path}")
                else:
                    shutil.copy2(src_path, dest_path)
                counts[label] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Import external labeled disease images into datasets/kermany while preserving existing data.")
    parser.add_argument("--source-dir", required=True, help="Path to the external dataset root folder containing disease class subfolders.")
    parser.add_argument("--mapping", help="Optional JSON file mapping external folder names to internal class labels.")
    parser.add_argument("--split", choices=("auto", "none", "train-test"), default="auto",
        help="auto: preserve source split if present, otherwise split class folders into train/test. none: copy into root class folders only. train-test: require source train/test folders.")
    parser.add_argument("--train-ratio", type=float, default=0.8,
        help="Ratio of source class images to place in train when splitting class folders into train/test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to split images into train and test.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without writing files.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    mapping = load_mapping(Path(args.mapping)) if args.mapping else DEFAULT_MAPPING.copy()
    print(f"Using source: {source_dir}")
    print(f"Mapping keys: {len(mapping)} entries")

    source_has_split = any((source_dir / split_name).is_dir() for split_name in ("train", "test"))
    if args.split == "train-test":
        if not source_has_split:
            raise SystemExit("Source directory does not contain train/test split folders.")
        counts_train = copy_images(source_dir / "train", mapping, DEST / "train", dry_run=args.dry_run)
        counts_test = copy_images(source_dir / "test", mapping, DEST / "test", dry_run=args.dry_run)
        counts = {cls: counts_train.get(cls, 0) + counts_test.get(cls, 0) for cls in CLASS_LABELS}
    elif args.split == "auto" and source_has_split:
        counts_train = copy_images(source_dir / "train", mapping, DEST / "train", dry_run=args.dry_run) if (source_dir / "train").is_dir() else {cls: 0 for cls in CLASS_LABELS}
        counts_test = copy_images(source_dir / "test", mapping, DEST / "test", dry_run=args.dry_run) if (source_dir / "test").is_dir() else {cls: 0 for cls in CLASS_LABELS}
        counts = {cls: counts_train.get(cls, 0) + counts_test.get(cls, 0) for cls in CLASS_LABELS}
    elif args.split in ("auto", "none"):
        if args.split == "auto":
            print("Source has no train/test structure; splitting class folders into train/test using train_ratio.")
            counts = copy_split_images(source_dir, mapping, DEST, args.train_ratio, args.seed, dry_run=args.dry_run)
        else:
            counts = copy_images(source_dir, mapping, DEST, dry_run=args.dry_run)
    else:
        raise SystemExit(f"Unsupported split mode: {args.split}")

    print("\nImported images:")
    for cls in CLASS_LABELS:
        print(f"  {cls}: {counts[cls]}")


if __name__ == "__main__":
    main()
