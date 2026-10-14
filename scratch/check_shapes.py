import torch
from pathlib import Path
root = Path(__file__).resolve().parents[1]

for name in ['efficientnet_b0_oct.pth', 'mobilenet_v3_oct.pth']:
    path = root / 'models' / name
    if path.exists():
        state = torch.load(path, map_location='cpu')
        print(f"=== {name} ===")
        for k, v in state.items():
            if 'classifier' in k:
                print(k, v.shape)
