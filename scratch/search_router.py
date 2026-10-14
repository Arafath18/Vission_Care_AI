from pathlib import Path
root = Path(__file__).resolve().parents[1]
js_path = root / 'frontend' / 'js' / 'api.js'
html_path = root / 'frontend' / 'index.html'

def search(p):
    if p.exists():
        content = p.read_text(encoding='utf-8')
        print(f"=== Search in {p.name} ===")
        for idx, line in enumerate(content.splitlines()):
            if any(term in line.lower() for term in ['location.', 'search', 'query', 'urlsearchparams', 'parameter', 'verify', 'scan', 'qrcode']):
                print(f"  Line {idx+1}: {line.strip()[:100]}")

search(js_path)
search(html_path)
