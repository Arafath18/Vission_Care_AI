from pathlib import Path
root = Path(__file__).resolve().parents[1]
frontend_dir = root / 'frontend'

print("Listing all files in", frontend_dir)
if frontend_dir.exists():
    for p in frontend_dir.rglob('*'):
        print(p.relative_to(frontend_dir))
else:
    print("Frontend directory not found.")
