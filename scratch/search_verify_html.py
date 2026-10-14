from pathlib import Path
root = Path(__file__).resolve().parents[1]
html_path = root / 'frontend' / 'index.html'

if html_path.exists():
    content = html_path.read_text(encoding='utf-8')
    lines = content.splitlines()
    print("Found occurrences of verify/qr/result in index.html:")
    for idx, line in enumerate(lines):
        if any(term in line.lower() for term in ['verify', 'qr', 'scan-result', 'scanner']):
            print(f"Line {idx+1}: {line.strip()[:120]}")
else:
    print("index.html not found.")
