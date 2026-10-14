"""
Download Kermany OCT2017 dataset for VisionCare training.

Sources:
  - Kaggle: andrewmvd/retinal-oct-categorization (requires ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)
  - Hugging Face: zacharielegault/Kermany2017-OCT (streaming or parquet)

Exports images to datasets/kermany/{CNV,DME,DRUSEN,NORMAL}/

Usage:
  python scripts/download_datasets.py --max-per-class 1000
  python scripts/download_datasets.py --source hf-stream --max-per-class 800
  python scripts/download_datasets.py --source kaggle --kaggle-dataset fawazahmed0/retinal-oct-images
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from utils.config import CLASS_LABELS

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "datasets" / "kermany"
CLASS_NAMES = CLASS_LABELS
HF_REPO = "zacharielegault/Kermany2017-OCT"
KAGGLE_DATASET = "andrewmvd/retinal-oct-categorization"
SOURCE_CHOICES = ("auto", "kaggle", "hf-stream", "hf-parquet")


def _prepare_destination(dest: Path, merge: bool = False) -> dict[str, int]:
    dest.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for cls in CLASS_NAMES:
        folder = dest / cls
        folder.mkdir(parents=True, exist_ok=True)
        if merge:
            counts[cls] = sum(1 for _ in folder.iterdir() if _.is_file() and _.suffix.lower() in IMAGE_EXTS)
        else:
            for child in folder.iterdir() if folder.exists() else []:
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
            counts[cls] = 0
    return counts


def _normalize_label(label) -> str | None:
    """HF ClassLabel order: CNV=0, DME=1, DRUSEN=2, NORMAL=3."""
    if isinstance(label, int):
        if 0 <= label < len(CLASS_NAMES):
            return CLASS_NAMES[label]
        return None
    label = str(label).upper().strip()
    return label if label in CLASS_NAMES else None


def export_from_streaming(max_per_class: int, hf_repo: str, merge: bool = False) -> dict[str, int]:
    from datasets import load_dataset

    print(f"Streaming from Hugging Face ({hf_repo}) (no full 84k cache required)...")
    counts = _prepare_destination(DEST, merge=merge)

    for split in ("train", "test"):
        if all(counts[c] >= max_per_class for c in CLASS_NAMES):
            break
        print(f"  Processing split: {split}")
        try:
            ds = load_dataset(hf_repo, split=split, streaming=True)
        except Exception as e:
            print(f"  split {split} skip: {e}")
            continue

        for row in ds:
            label = _normalize_label(row.get("label"))
            if label is None or counts[label] >= max_per_class:
                continue

            img = row.get("image")
            if img is None:
                continue
            idx = counts[label]
            out = DEST / label / f"{split}_{idx:05d}.jpeg"
            img.convert("RGB").save(out, "JPEG", quality=92)
            counts[label] += 1

            if all(counts[c] >= max_per_class for c in CLASS_NAMES):
                print("  All classes reached max-per-class limit.")
                return counts

    return counts


def export_from_parquet_shards(max_per_class: int, max_shards: int = 4, hf_repo: str = HF_REPO, merge: bool = False) -> dict[str, int]:
    """Download a few Parquet shards only (faster than full dataset)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    counts: dict[str, int] = defaultdict(int, _prepare_destination(DEST, merge=merge))

    for i in range(max_shards):
        fname = f"data/train-{i:05d}-of-00016.parquet"
        print(f"Downloading shard {fname} from {hf_repo}...")
        path = hf_hub_download(hf_repo, fname, repo_type="dataset")
        table = pq.read_table(path)
        col_names = table.column_names
        label_col = "label" if "label" in col_names else None
        image_col = "image" if "image" in col_names else None
        if not label_col or not image_col:
            print("  Unknown schema:", col_names)
            continue

        for row_idx in range(table.num_rows):
            if all(counts[c] >= max_per_class for c in CLASS_NAMES):
                return dict(counts)

            label_val = table.column(label_col)[row_idx].as_py()
            if isinstance(label_val, int):
                label = CLASS_NAMES[label_val]
            else:
                label = str(label_val).upper()
                if label not in CLASS_NAMES:
                    continue

            if counts[label] >= max_per_class:
                continue

            img_data = table.column(image_col)[row_idx].as_py()
            if isinstance(img_data, dict) and "bytes" in img_data:
                from io import BytesIO
                from PIL import Image
                img = Image.open(BytesIO(img_data["bytes"]))
            else:
                img = img_data

            out = DEST / label / f"shard{i}_{counts[label]:05d}.jpeg"
            img.convert("RGB").save(out, "JPEG", quality=92)
            counts[label] += 1

    return dict(counts)


def kaggle_credentials_available() -> bool:
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def download_kaggle(dest: Path, dataset_id: str) -> bool:
    if not kaggle_credentials_available():
        return False
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", dataset_id,
        "-p", str(dest),
        "--unzip",
    ]
    print("Running Kaggle download into", dest)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr or result.stdout)
        return False
    _reorganize_nested(dest)
    return True


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SPLIT_DIRS = ("train", "val", "valid", "validation", "test")


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.symlink(src, dest)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(src, dest)


def _reorganize_nested(root: Path) -> int:
    """Flatten train/CNV, val/CNV, test/CNV into root/CNV (or root/kermany/CNV)."""
    targets = [root]
    if (root / "kermany").is_dir():
        targets.append(root / "kermany")
    merged = 0
    for dest in targets:
        dest.mkdir(parents=True, exist_ok=True)
        for cls in CLASS_NAMES:
            out_dir = dest / cls
            out_dir.mkdir(exist_ok=True)
            for base in (dest, root, root.parent):
                for split in SPLIT_DIRS:
                    for name in (cls, cls.lower()):
                        src_dir = base / split / name
                        if not src_dir.is_dir():
                            continue
                        for f in src_dir.iterdir():
                            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                                continue
                            target = out_dir / f.name
                            if target.exists():
                                continue
                            _link_or_copy(f, target)
                            merged += 1
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="auto",
        help="Dataset source: auto, kaggle, hf-stream, or hf-parquet")
    parser.add_argument("--kaggle", action="store_true",
        help="Alias for --source kaggle (attempt Kaggle download if credentials are available)")
    parser.add_argument("--kaggle-dataset", default=KAGGLE_DATASET,
        help="Kaggle dataset slug to download, e.g. andrewmvd/retinal-oct-categorization")
    parser.add_argument("--hf-repo", default=HF_REPO,
        help="Hugging Face dataset repo slug to use for HF sources")
    parser.add_argument("--streaming", action="store_true", help="Use HF streaming API (alias for --source hf-stream)")
    parser.add_argument("--parquet-shards", type=int, default=0, help="Use N parquet shards (0=auto)")
    parser.add_argument("--merge", dest="merge", action="store_true", help="Keep existing images in destination and add new images without clearing")
    parser.add_argument("--clear", dest="merge", action="store_false", help="Remove existing images in destination before exporting.")
    parser.set_defaults(merge=True)
    args = parser.parse_args()

    if args.streaming:
        args.source = "hf-stream"
    if args.kaggle:
        args.source = "kaggle"

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    use_kaggle = args.source == "kaggle" or (args.source == "auto" and kaggle_credentials_available())

    if use_kaggle:
        if download_kaggle(DEST, args.kaggle_dataset):
            merged = _reorganize_nested(DEST)
            if merged:
                print(f"Merged {merged} nested split images into class folders.")
            counts = {c: len(list((DEST / c).glob("*"))) for c in CLASS_NAMES}
        elif args.source == "kaggle":
            print("Kaggle download requested but failed or credentials were not available.")
            sys.exit(1)
        else:
            print("Kaggle download unavailable; falling back to Hugging Face.")
            args.source = "hf-parquet" if args.source == "auto" else args.source
            counts = export_from_parquet_shards(args.max_per_class, args.parquet_shards or 3, args.hf_repo, merge=args.merge) if args.source == "hf-parquet" else export_from_streaming(args.max_per_class, args.hf_repo, merge=args.merge)
    elif args.source == "hf-stream":
        counts = export_from_streaming(args.max_per_class, args.hf_repo, merge=args.merge)
    elif args.source == "hf-parquet":
        counts = export_from_parquet_shards(args.max_per_class, args.parquet_shards or 3, args.hf_repo, merge=args.merge)
    else:
        # auto fallback path
        print(
            "No Kaggle credentials found. Using Hugging Face mirror (same Kermany OCT2017 data).\n"
            "For Kaggle: https://www.kaggle.com/settings → Create New Token → save as kaggle.json\n"
        )
        counts = export_from_parquet_shards(args.max_per_class, args.parquet_shards or 3, args.hf_repo, merge=args.merge)

    print("\nExport complete:", DEST)
    for cls in CLASS_NAMES:
        n = len(list((DEST / cls).glob("*"))) if (DEST / cls).exists() else 0
        print(f"  {cls}: {n} images")
    total = sum(len(list((DEST / c).glob("*"))) for c in CLASS_NAMES if (DEST / c).exists())
    if total == 0:
        print("ERROR: No images exported.")
        sys.exit(1)
    print(f"Total: {total}")


if __name__ == "__main__":
    main()
