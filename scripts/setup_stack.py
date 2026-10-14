"""Initialize DB, seed demo user, verify dataset and models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database.db import get_db
from models.train import load_samples_from_split
from utils.config import CLASS_LABELS, MODELS_DIR

DATA = ROOT / "datasets" / "kermany"
DEMO_EMAIL = "demo@visioncare.ai"
DEMO_PASSWORD = "demo123"
DEMO_NAME = "Dr. Arjun"


def count_split(split: str) -> dict[str, int]:
    samples = load_samples_from_split(DATA, split, max_per_class=None)
    return {CLASS_LABELS[i]: sum(1 for _, l in samples if l == i) for i in range(len(CLASS_LABELS))}


def main() -> None:
    db = get_db()
    print(f"Database: {db.db_path}")

    if not db.get_user_by_email(DEMO_EMAIL):
        ok, msg = db.create_user(DEMO_NAME, DEMO_EMAIL, DEMO_PASSWORD)
        print(f"Demo user: {msg}" if ok else f"Demo user error: {msg}")
    else:
        print(f"Demo user exists: {DEMO_EMAIL} / {DEMO_PASSWORD}")

    train = count_split("train") if (DATA / "train").exists() else {}
    test = count_split("test") if (DATA / "test").exists() else {}
    print(f"Dataset train: {train}")
    print(f"Dataset test:  {test}")

    for name in ("efficientnet_b0_oct.pth", "mobilenet_v3_oct.pth", "attention_unet_seg.pth"):
        path = MODELS_DIR / name
        print(f"Model {name}: {'OK' if path.exists() else 'MISSING'}")

    print("\nRun server:  run.bat")
    print(f"Login:       {DEMO_EMAIL} / {DEMO_PASSWORD}")


if __name__ == "__main__":
    main()
