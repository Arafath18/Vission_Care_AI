import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]

print("Root path:", root)
for p in root.rglob('*'):
    if p.is_file() and p.suffix in ['.py', '.html', '.js']:
        try:
            content = p.read_text(encoding='utf-8')
            if 'qr' in content.lower() or 'verify' in content.lower():
                print(f"Found in {p.relative_to(root)}")
        except Exception:
            pass
