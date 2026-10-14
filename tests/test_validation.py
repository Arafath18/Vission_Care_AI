"""Unit tests for the OCT validation helper."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.validation import validate_oct_image


class TestValidation(unittest.TestCase):
    def test_validate_oct_image_blank(self):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        valid, message = validate_oct_image(image)
        self.assertFalse(valid)
        self.assertIn("valid retinal oct", message.lower())

    def test_validate_oct_image_real_sample(self):
        data_dir = Path(__file__).resolve().parents[1] / "datasets" / "kermany" / "test" / "NORMAL"
        if not data_dir.exists():
            self.skipTest("Kermany test dataset is not present.")

        image_paths = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}])
        if not image_paths:
            self.skipTest("No sample images found in datasets/kermany/test/NORMAL")

        image = cv2.imread(str(image_paths[0]))
        self.assertIsNotNone(image, f"Failed to read image: {image_paths[0]}")
        valid, message = validate_oct_image(image)
        self.assertTrue(valid)
        self.assertEqual(message, "Valid retinal OCT scan.")


if __name__ == "__main__":
    unittest.main()
