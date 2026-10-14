"""Evaluation metrics for model training."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            )
        except ValueError:
            metrics["roc_auc"] = 0.0
    # Sensitivity / specificity for binary case
    if len(np.unique(y_true)) == 2:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        metrics["sensitivity"] = tp / (tp + fn + 1e-8)
        metrics["specificity"] = tn / (tn + fp + 1e-8)
    return metrics
