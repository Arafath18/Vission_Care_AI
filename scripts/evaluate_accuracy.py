"""Evaluate the trained VisionCare models on the test dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.train import load_samples_from_split
from utils.inference import VisionCarePredictor
from utils.metrics import compute_metrics
from utils.config import CLASS_LABELS, OUTPUTS_DIR


def evaluate(dataset_root: Path, max_samples: int | None = None) -> dict:
    samples = load_samples_from_split(dataset_root, "test", max_per_class=None)
    if not samples:
        raise ValueError(f"No test samples found in {dataset_root}")
    if max_samples:
        samples = samples[:max_samples]

    predictor = VisionCarePredictor()

    y_true = []
    y_pred = []
    y_prob = []
    failed = []

    for path, label in samples:
        image = cv2.imread(str(path))
        if image is None:
            failed.append(str(path))
            continue
        try:
            result = predictor.predict(image)
        except Exception as exc:
            failed.append(f"{path}: {exc}")
            continue
        y_true.append(label)
        y_pred.append(CLASS_LABELS.index(result["label"]))
        y_prob.append([result["probabilities"][cls] / 100.0 for cls in CLASS_LABELS])

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_LABELS))), target_names=CLASS_LABELS, zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_LABELS)))).tolist()

    summary = {
        "dataset_root": str(dataset_root),
        "samples_evaluated": int(len(y_true)),
        "samples_failed": len(failed),
        "failed_samples": failed[:20],
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": {
            "labels": CLASS_LABELS,
            "matrix": matrix,
        },
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VisionCare classifier accuracy on the test dataset.")
    parser.add_argument("--data", default="datasets/kermany", help="Root dataset folder")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--output", default="outputs/evaluation/accuracy_report.json", help="JSON output path")
    args = parser.parse_args()

    root = Path(args.data).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = evaluate(root, args.max_samples)
    output_path.write_text(json.dumps(summary, indent=2))

    print(f"Evaluation complete: {summary['samples_evaluated']} samples")
    print(f"Accuracy: {summary['metrics']['accuracy']:.4f}")
    print(f"Precision: {summary['metrics']['precision']:.4f}")
    print(f"Recall: {summary['metrics']['recall']:.4f}")
    print(f"F1: {summary['metrics']['f1']:.4f}")
    if "roc_auc" in summary["metrics"]:
        print(f"ROC AUC: {summary['metrics']['roc_auc']:.4f}")
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
