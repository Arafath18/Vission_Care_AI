"""Initialize dataset folders for all configured retinal disease classes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import CLASS_LABELS

DATA_ROOT = ROOT / "datasets" / "kermany"


def main() -> None:
    for folder in (DATA_ROOT, DATA_ROOT / "train", DATA_ROOT / "test"):
        folder.mkdir(parents=True, exist_ok=True)

    for base in (DATA_ROOT, DATA_ROOT / "train", DATA_ROOT / "test"):
        for cls in CLASS_LABELS:
            (base / cls).mkdir(parents=True, exist_ok=True)

    print(f"Created folder structure for {len(CLASS_LABELS)} retina classes:")
    for cls in CLASS_LABELS:
        print(f"  - {cls}")
    print("\nNow add your images into datasets/kermany/train/<class>/ and datasets/kermany/test/<class>/.")


if __name__ == "__main__":
    main()
