import torch
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for name in ['efficientnet_b0_oct.pth', 'mobilenet_v3_oct.pth', 'attention_unet_seg.pth']:
    path = root / 'models' / name
    print('CHECK', path, 'exists', path.exists(), 'size', path.stat().st_size if path.exists() else None)
    if path.exists():
        try:
            state = torch.load(path, map_location='cpu')
            print(' loaded type', type(state), 'keys', len(state) if hasattr(state, '__len__') else None)
            if hasattr(state, 'keys'):
                print(' first keys', list(state.keys())[:5])
        except Exception as exc:
            print(' load failed', exc)
