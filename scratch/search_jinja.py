from pathlib import Path
root = Path(__file__).resolve().parents[1]
main_py = root / 'api' / 'main.py'

if main_py.exists():
    content = main_py.read_text(encoding='utf-8')
    print("HTMLResponse in main.py:", "HTMLResponse" in content)
    print("Jinja2 in main.py:", "Jinja2" in content)
    print("templates in main.py:", "templates" in content)
    for line in content.splitlines():
        if 'response_class' in line or 'templates.' in line:
            print("  ", line.strip())
else:
    print("main.py not found.")
