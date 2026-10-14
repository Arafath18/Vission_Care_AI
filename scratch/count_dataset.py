import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from utils.config import CLASS_LABELS

for split in ['train', 'test']:
    print(f"=== {split} split ===")
    split_dir = root / 'datasets' / 'kermany' / split
    if not split_dir.exists():
        print(f"Directory {split_dir} does not exist.")
        continue
    for cls in CLASS_LABELS:
        folder = split_dir / cls
        if folder.exists():
            files = list(folder.glob('*'))
            print(f"  {cls}: {len(files)} files")
        else:
            print(f"  {cls}: directory missing")
