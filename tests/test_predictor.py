"""Smoke tests for the VisionCare predictor."""

from pathlib import Path

import cv2
import numpy as np

from utils.inference import VisionCarePredictor
from utils.config import CLASS_LABELS


def test_predictor_returns_expected_fields() -> None:
    sample_path = Path(__file__).resolve().parents[1] / "datasets" / "kermany" / "test"
    image_paths = sorted([p for p in sample_path.rglob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    if not image_paths:
        raise RuntimeError("No test dataset images found for predictor smoke test.")

    image = cv2.imread(str(image_paths[0]))
    assert image is not None, f"Failed to read image: {image_paths[0]}"

    predictor = VisionCarePredictor()
    result = predictor.predict(image)

    assert result["label"] in CLASS_LABELS
    assert 0 <= result["confidence"] <= 100
    assert isinstance(result["probabilities"], dict)
    assert set(result["probabilities"]) == set(CLASS_LABELS)
    assert "seg_mask" in result
