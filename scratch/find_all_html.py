from pathlib import Path
root = Path(__file__).resolve().parents[1]

print("Listing all HTML files in the project:")
found = False
for p in root.rglob('*.html'):
    print(p.relative_to(root))
    found = True
if not found:
    print("No HTML files found in the project.")
