from pathlib import Path
root = Path(__file__).resolve().parents[1]
js_path = root / 'frontend' / 'js' / 'api.js'

if js_path.exists():
    content = js_path.read_text(encoding='utf-8')
    lines = content.splitlines()
    print("Found occurrences in api.js:")
    for idx, line in enumerate(lines):
        if any(term in line.lower() for term in ['verify', 'qr', 'report']):
            print(f"Line {idx+1}: {line.strip()[:120]}")
else:
    print("api.js not found.")
