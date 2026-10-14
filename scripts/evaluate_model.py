"""Evaluate VisionCare model performance on the Kermany OCT test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils.config import CLASS_LABELS, DATASETS_DIR, OUTPUTS_DIR
from utils.inference import get_predictor
from utils.metrics import compute_metrics


def find_image_paths(split_root: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(split_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        if label not in CLASS_LABELS:
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for path in sorted(class_dir.glob(ext)):
                samples.append((path, label))
    return samples


def load_dataset(dataset_root: Path, split: str) -> list[tuple[Path, str]]:
    split_root = dataset_root / "kermany" / split
    if not split_root.exists():
        raise FileNotFoundError(f"Dataset split folder not found: {split_root}")
    samples = find_image_paths(split_root)
    if not samples:
        raise ValueError(f"No images found in split: {split_root}")
    return samples


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = compute_metrics(y_true, y_pred, y_prob)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_LABELS))),
        target_names=CLASS_LABELS,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_LABELS))))
    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def run_evaluation(dataset_root: Path, split: str, save_report: bool) -> dict:
    samples = load_dataset(dataset_root, split)
    predictor = get_predictor()

    label_to_idx = {label: idx for idx, label in enumerate(CLASS_LABELS)}
    y_true = []
    y_pred = []
    y_prob = []

    start = time.perf_counter()
    for image_path, label in samples:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: failed to read image: {image_path}")
            continue
        result = predictor.predict(image)
        y_true.append(label_to_idx[label])
        predicted_label = result["label"]
        y_pred.append(label_to_idx.get(predicted_label, -1))
        y_prob.append(result.get("probs_raw", np.zeros(len(CLASS_LABELS), dtype=np.float32)))

    elapsed = time.perf_counter() - start
    if not y_true:
        raise RuntimeError("No valid predictions were generated.")

    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)
    y_prob_np = np.stack([np.asarray(row, dtype=np.float32) for row in y_prob], axis=0)

    evaluation = evaluate_predictions(y_true_np, y_pred_np, y_prob_np)
    evaluation["dataset"] = {
        "root": str(dataset_root),
        "split": split,
        "samples": len(y_true_np),
    }
    evaluation["timing"] = {
        "total_seconds": round(elapsed, 3),
        "seconds_per_image": round(elapsed / len(y_true_np), 5),
    }

    if save_report:
        report_path = OUTPUTS_DIR / "reports" / f"evaluation_{split}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        print(f"Saved evaluation report: {report_path}")

    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VisionCare classifier on OCT dataset")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASETS_DIR,
        help="Path to dataset root containing kermany/",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save evaluation report to outputs/reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Evaluating model on split: {args.split}")
    evaluation = run_evaluation(args.dataset_root, args.split, save_report=not args.no_save)
    print(json.dumps(evaluation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
