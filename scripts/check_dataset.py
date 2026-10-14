from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

dataset_root = root / 'datasets' / 'kermany' / 'test'
print('root', dataset_root)
from utils.config import CLASS_LABELS

for cls in CLASS_LABELS:
    folder = root / cls
    if not folder.exists():
        print('missing', cls)
        continue
    files = sorted(folder.glob('*'))
    print(cls, len(files), 'files')
    if files:
        print(' first', files[0].name, files[0].suffix, files[0].stat().st_size)
