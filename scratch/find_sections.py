from pathlib import Path
root = Path(__file__).resolve().parents[1]
html_path = root / 'frontend' / 'index.html'

if html_path.exists():
    content = html_path.read_text(encoding='utf-8')
    print("HTML Length:", len(content))
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if 'id=' in line or 'class="modal' in line or 'class="section' in line or 'class="page' in line or '<nav' in line or '<!--' in line:
            if any(term in line.lower() for term in ['report', 'modal', 'card', 'view', 'download', 'qr', 'scan']):
                print(f"Line {idx+1}: {line.strip()[:100]}")
else:
    print("HTML not found.")
