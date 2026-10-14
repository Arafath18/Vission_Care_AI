"""Split flat datasets/kermany/{CLASS} into train/ and test/ (Kaggle-style)."""

import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "datasets" / "kermany"
from utils.config import CLASS_LABELS

CLASSES = CLASS_LABELS
TEST_PER_CLASS = 250
SEED = 42


def main():
    random.seed(SEED)
    for cls in CLASSES:
        flat = SRC / cls
        if not flat.exists():
            print(f"Missing {flat}")
            continue
        files = [f for f in flat.iterdir() if f.is_file()]
        random.shuffle(files)
        test_files = files[:TEST_PER_CLASS]
        train_files = files[TEST_PER_CLASS:]

        for split, subset in (("test", test_files), ("train", train_files)):
            dest = SRC / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in subset:
                target = dest / f.name
                if not target.exists():
                    shutil.copy2(f, target)

        print(f"{cls}: train={len(train_files)} test={len(test_files)}")


if __name__ == "__main__":
    main()
