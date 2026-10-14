"""Download Noto Sans Tamil font for PDF reports."""

import urllib.request
from pathlib import Path

FONT_URL = (
    "https://github.com/googlefonts/noto-fonts/raw/main/"
    "hinted/ttf/NotoSansTamil/NotoSansTamil-Regular.ttf"
)


def main():
    root = Path(__file__).resolve().parent.parent
    fonts_dir = root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    dest = fonts_dir / "NotoSansTamil-Regular.ttf"
    if dest.exists():
        print(f"Font already exists: {dest}")
        return
    print(f"Downloading Tamil font to {dest}...")
    urllib.request.urlretrieve(FONT_URL, dest)
    print("Done.")


if __name__ == "__main__":
    main()
