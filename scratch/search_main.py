from pathlib import Path
root = Path(__file__).resolve().parents[1]
main_py = root / 'api' / 'main.py'

if main_py.exists():
    content = main_py.read_text(encoding='utf-8')
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if '@app.' in line:
            print(f"Line {idx+1}: {line.strip()}")
else:
    print("main.py not found.")
